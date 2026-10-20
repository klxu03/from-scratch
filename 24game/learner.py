from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from actor import Rollout
from env import calculator
import torch
import numpy as np
from utils.prompts import prompt
from pathlib import Path
from lora_linear import apply_lora

class Learner:
    def __init__(self, model: Qwen3_5ForCausalLM, max_output_tokens: int = 4096, kl_loss_weight: float = 0.01):
        self.model = model
        self.optimizer = torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad), lr=1e-5)
        self.max_output_tokens = max_output_tokens
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
        self.kl_loss_weight = kl_loss_weight

    def zero_grad(self) -> None:
        self.model.zero_grad()
        self.cumulated_rewards = 0

    def compute_advantages(self, rollout_group: list[Rollout]) -> np.ndarray:
        rewards = np.asarray([rollout.reward for rollout in rollout_group])
        self.cumulated_rewards += sum(rewards)
        advantages = rewards - rewards.mean()
        std = advantages.std()
        if std == 0:
            return np.zeros_like(advantages)
        return advantages / std

    def compute_action_logprobs(self, token_ids: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        print(f"Computing action logprobs, token_ids shape: {token_ids.shape}, action_mask shape: {action_mask.shape}")
        logits = self.model(input_ids = token_ids).logits[0]
        print(f"Logits shape: {logits.shape}")
        logprobs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        print(f"Logprobs shape: {logprobs.shape}")

        token_ids = token_ids.squeeze(0)
        targets = token_ids[1:]
        print(f"token_ids[1:] is dim 0, the batch axis: {targets.shape}")
        print(f"action_mask dtype: {action_mask.dtype}, ones count: {int(action_mask.sum())}")
        selected = logprobs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        print(f"Selected shape: {selected.shape}")

        return selected[action_mask[1:]]

    def accumulate_group(self, rollout_group: list[Rollout], puzzle: list[int], total_rollouts: int) -> None:
        advantages = self.compute_advantages(rollout_group)
        print(f"Computed advantages: {advantages}")

        prompt_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt(self.max_output_tokens, puzzle)}],
            tools=[calculator],
            add_generation_prompt=True,
            enable_thinking=True,
            return_tensors="pt",
        )["input_ids"]

        for i, rollout in enumerate(rollout_group):
            print(f"This rollout has {len(rollout.turns)} turns")
            token_ids = torch.cat([prompt_ids, *[turn.token_ids for turn in rollout.turns]], dim=1).to("mps")
            action_mask = torch.cat([
                torch.zeros(prompt_ids.shape[1]).bool(),
                *[
                    torch.cat([
                        torch.zeros(turn.completion_start).bool(),
                        torch.ones(turn.token_ids.shape[1] - turn.completion_start).bool(),
                    ])
                    for turn in rollout.turns
                ],
            ]).to("mps")
            action_logprobs = self.compute_action_logprobs(token_ids, action_mask)

            with torch.no_grad(), self.model.disable_lora():
                ref_logprobs = self.compute_action_logprobs(token_ids, action_mask)
            
            log_ratio = ref_logprobs - action_logprobs
            kl_loss = torch.expm1(log_ratio) - log_ratio

            loss = (-float(advantages[i]) * action_logprobs.mean() + self.kl_loss_weight * kl_loss.mean()) / total_rollouts
            loss.backward()

    def step(self) -> float:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return self.cumulated_rewards

    def create_checkpoint(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

        lora_weights = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                lora_weights[name] = p.detach().cpu()

        torch.save({
            "lora": lora_weights,
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
        params = dict(self.model.named_parameters())

        with torch.no_grad():
            for name, weight in checkpoint["lora"].items():
                params[name].copy_(weight)

        self.optimizer.load_state_dict(checkpoint["optimizer"])

if __name__ == "__main__":
    from actor import Actor
    model = Qwen3_5ForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16).to("mps")
    model = apply_lora(model)
    actor = Actor(model, 1024)
    sample_rollout = actor.collect_rollouts([[6, 6, 6, 6]], 8)[0]
    learner = Learner(model, 1024)
    learner.zero_grad()
    learner.accumulate_group(sample_rollout, [6, 6, 6, 6], 8)
    learner.step()