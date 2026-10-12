# https://huggingface.co/learn/deep-rl-course/unit8/hands-on-cleanrl

import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Distribution, Normal, Independent
from tqdm.auto import tqdm
import time

NUM_ENVS = 8
TRAJECTORY_SIZE = 256
TOTAL_TIMESTEPS = 4_001_792
MINIBATCH_SIZE = 256
NUM_EPOCHS = 8
GAMMA = 0.99
EPS = 0.2
C_1 = 0.5
C_2 = 0.05


class Agent(nn.Module):
  def __init__(self, batch, input_dim, output_dim):
    super().__init__()
    self.batch = batch
    self.input_dim = input_dim
    self.output_dim = output_dim

    self.actor = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, output_dim),
    )
    self.actor_logstd = nn.Parameter(torch.zeros(output_dim))

    self.value = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )

    self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)

  def get_action_distribution(self, obs: torch.Tensor) -> Distribution:
    mean = self.actor(obs)
    std = self.actor_logstd.exp()
    
    return Independent(Normal(mean, std), 1)
  
  def get_value(self, obs: torch.Tensor):
    return self.value(obs).squeeze(-1)

  def precompute_advantages(self, obs: torch.Tensor, next_obs: torch.Tensor, terminated: torch.Tensor, rewards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
      old_values = self.get_value(obs) # (256, 4)
      next_values = self.get_value(next_obs) # (256, 4)
      next_values = next_values.masked_fill(terminated, 0.0)
    
    td_targets = rewards + GAMMA * next_values
    advantages = td_targets - old_values

    return advantages, td_targets

  def learn(self, data):
    data[:] = [torch.stack([torch.as_tensor(d[i]) for d in data]).flatten(0, 1) for i in range(6)]
    obs, next_obs, terminated, rewards, _, _ = data

    advantages, td_targets = self.precompute_advantages(obs, next_obs, terminated, rewards)

    for epoch in range(NUM_EPOCHS):
      indices = torch.randperm(NUM_ENVS * TRAJECTORY_SIZE)

      for start in range(0, NUM_ENVS * TRAJECTORY_SIZE, MINIBATCH_SIZE):
        mb_indices = indices[start: start + MINIBATCH_SIZE]

        mb_data = [x[mb_indices] for x in data]

        obs, next_obs, terminated, rewards, actions, old_log_probs = mb_data
   
        # PPO actor loss
        dist = self.get_action_distribution(obs)
        ratio = torch.exp(dist.log_prob(actions) - old_log_probs)
        L_CLIP = torch.minimum(ratio * advantages[mb_indices], torch.clamp(ratio, 1 - EPS, 1 + EPS) * advantages[mb_indices]).mean()

        L_S = dist.entropy().mean()

        # Critic loss
        L_VF = 0.5 * ((self.get_value(obs) - td_targets[mb_indices])**2).mean()

        loss = -L_CLIP + C_1 * L_VF - C_2 * L_S
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# vector_kwargs same step is important, otherwise 100 terminats, 101 is intermediary I'd have to write code to ignore, 102 is the starting state of next obs. Otherwise 102 becomes 101
envs = gym.make_vec("LunarLander-v3", num_envs=NUM_ENVS, vectorization_mode="async", continuous=True, enable_wind=True, vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})
agent = Agent(NUM_ENVS, 8, 2)

next_obs, infos = envs.reset()

rollout_time = 0.0
learn_time = 0.0

for update in tqdm(range(TOTAL_TIMESTEPS // (NUM_ENVS * TRAJECTORY_SIZE)), desc="Training"):
  start = time.perf_counter()

  # [obs, next_obs, terminated, reward, action, old_log_prob]
  data = []
  for step in range(0, TRAJECTORY_SIZE):
    obs = torch.as_tensor(next_obs, dtype=torch.float32)
    with torch.no_grad():
      dist = agent.get_action_distribution(obs)
      actions = dist.sample()
    next_obs, rewards, terminated, truncated, infos = envs.step(actions.cpu().numpy())

    # transition_next_obs = [obs if not truncated[i] else infos["final_obs"][i] for i, obs in enumerate(next_obs)]
    transition_next_obs = next_obs.copy()
    if truncated.any(): transition_next_obs[truncated] = np.stack(infos["final_obs"][truncated])

    # we can just store dist.log_prob(action) instead of returning entire dist to be more memory efficient. For PPO we need pi_new/pi_old = e^(pi_new_log - pi_old_log)
    data.append([obs, transition_next_obs, terminated, rewards, actions, dist.log_prob(actions)])
  rollout_time += time.perf_counter() - start

  start = time.perf_counter()
  agent.learn(data)
  learn_time += time.perf_counter() - start

total_time = rollout_time + learn_time
print(f"Rollout: {rollout_time / total_time:.1%} ({rollout_time:.1f}s)")
print(f"Learning: {learn_time / total_time:.1%} ({learn_time:.1f}s)")
