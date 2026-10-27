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
        self.cumulated_kl_loss = 0

        self.entropy_sum = 0.0
        self.entropy_count = 0

    def compute_advantages(self, rollout_group: list[Rollout]) -> np.ndarray:
        rewards = np.asarray([rollout.reward for rollout in rollout_group])
        self.cumulated_rewards += sum(rewards)
        advantages = rewards - rewards.mean()
        std = advantages.std()
        if std == 0:
            return np.zeros_like(advantages)
        return advantages / std

    def compute_action_logprobs(self, token_ids: torch.Tensor, action_mask: torch.Tensor, record_entropy: bool = False) -> torch.Tensor:
        print(f"Computing action logprobs, token_ids shape: {token_ids.shape}, action_mask shape: {action_mask.shape}")
        logits = self.model(input_ids = token_ids, use_cache=False).logits[0]
        if record_entropy:
            self.record_entropy(logits, action_mask)
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
            token_ids = torch.cat([prompt_ids, *[turn.token_ids.unsqueeze(0) for turn in rollout.turns]], dim=1).to(self.model.device)
            action_mask = torch.cat([
                torch.zeros(prompt_ids.shape[1], dtype=torch.bool),
                *[
                    torch.cat([
                        torch.zeros(turn.completion_start, dtype=torch.bool),
                        torch.ones(turn.token_ids.numel() - turn.completion_start, dtype=torch.bool),
                    ])
                    for turn in rollout.turns
                ],
            ]).to(self.model.device)

            with torch.no_grad(), self.model.disable_lora():
                ref_logprobs = self.compute_action_logprobs(token_ids, action_mask)

            action_logprobs = self.compute_action_logprobs(token_ids, action_mask, record_entropy=True)
            
            log_ratio = ref_logprobs - action_logprobs
            kl_loss = torch.expm1(log_ratio) - log_ratio
            self.cumulated_kl_loss += (
                kl_loss.detach().mean().item() / total_rollouts
            )

            loss = (-float(advantages[i]) * action_logprobs.mean() + self.kl_loss_weight * kl_loss.mean()) / total_rollouts
            loss.backward()

    def step(self) -> float:
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        self.optimizer.step()
        print(f"Gradient norm (before clipping): {grad_norm.item():.4f}")
        print(f"Gradient clipped: {grad_norm.item() > 1.0}")
        print(f"Cumulated KL loss: {self.cumulated_kl_loss}")

        print(
            f"Policy entropy: "
            f"{self.entropy_sum / max(self.entropy_count, 1):.4f} nats"
        )

        return self.cumulated_rewards

    def record_entropy(self, logits, action_mask):
        with torch.no_grad():
            # Action at token position t+1 is predicted by logits[t].
            positions = action_mask[1:].nonzero(as_tuple=True)[0][::64]

            if positions.numel() == 0:
                return

            # Only gather selected positions, not the full sequence.
            selected_logits = logits.detach().index_select(
                0, positions
            ).float()

            log_p = torch.log_softmax(selected_logits, dim=-1)

            entropy = -(log_p.exp() * log_p).sum(dim=-1)

            self.entropy_sum += entropy.sum().item()
            self.entropy_count += entropy.numel()

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
                params[name].copy_(weight.to(self.model.device))

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.model.device)

if __name__ == "__main__":
    from actor import Actor
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = Qwen3_5ForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16).to(device)
    model = apply_lora(model)
    actor = Actor(model, 1024)
    sample_rollout = actor.collect_rollouts([[6, 6, 6, 6]], 8)[0]
    learner = Learner(model, 1024)
    learner.zero_grad()
    learner.accumulate_group(sample_rollout, [6, 6, 6, 6], 8)
    learner.step()