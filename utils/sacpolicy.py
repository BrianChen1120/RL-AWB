import torch
import torch.nn as nn
import copy
from gymnasium import spaces
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution


class CustomSACFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box):
        super().__init__(
            observation_space,
            features_dim=get_flattened_obs_dim(observation_space)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations.view(observations.size(0), -1)


class CustomActor(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.features_dim = get_flattened_obs_dim(observation_space)
        action_dim = action_space.shape[0]

        self.img_dim = 10800
        self.extra_dim = self.features_dim - self.img_dim

        self.img_net = nn.Sequential(
            nn.Linear(self.img_dim, 1024),   nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 256),  nn.ReLU(),
            nn.Linear(256, 128),   nn.ReLU(),
            nn.Linear(128, 64),    nn.ReLU(),
        )

        self.extra_net = nn.Sequential(
            nn.Linear(self.extra_dim, 32), nn.ReLU(),
            nn.Linear(32, 64),   nn.ReLU(),
        )

        self.mu_head = nn.Sequential(
            nn.Linear(64 + 64, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.dist = SquashedDiagGaussianDistribution(action_dim)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        wbsrgb_obs   = obs[:, :self.img_dim]
        action_obs = obs[:, self.img_dim:]

        wbsrgb_feat   = self.img_net(wbsrgb_obs)
        action_feat = self.extra_net(action_obs)

        fused = torch.cat([wbsrgb_feat, action_feat], dim=1)

        mu       = self.mu_head(fused)
        log_std  = self.log_std.expand_as(mu)
        dist     = self.dist.proba_distribution(mu, log_std)
        actions  = dist.get_actions(deterministic=deterministic)
        return actions

    def get_distribution(self, obs: torch.Tensor):
        img_feat   = self.img_net(obs[:, :self.img_dim])
        extra_feat = self.extra_net(obs[:, self.img_dim:])
        fused      = torch.cat([img_feat, extra_feat], dim=1)

        mu      = self.mu_head(fused)
        log_std = self.log_std.expand_as(mu)
        return self.dist.proba_distribution(mu, log_std)

    def set_training_mode(self, mode: bool):
        self.train(mode)

    def action_log_prob(self, obs: torch.Tensor):
        dist     = self.get_distribution(obs)
        actions  = dist.get_actions()
        log_prob = dist.log_prob(actions)
        return actions, log_prob


class CustomCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

        self.features_dim = get_flattened_obs_dim(observation_space)
        self.img_dim   = 60 * 60 * 3
        self.extra_dim = self.features_dim - self.img_dim

        self.img_net = nn.Sequential(
            nn.Linear(self.img_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )

        self.extra_net = nn.Sequential(
            nn.Linear(self.extra_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
        )

        action_dim = action_space.shape[0]

        def _make_q_head():
            return nn.Sequential(
                nn.Linear(64 + 64 + action_dim, 64), nn.ReLU(),
                nn.Linear(64, 1)
            )

        self.q1_head = _make_q_head()
        self.q2_head = copy.deepcopy(self.q1_head)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        img_feat   = self.img_net(obs[:, :self.img_dim])
        extra_feat = self.extra_net(obs[:, self.img_dim:])
        fused      = torch.cat([img_feat, extra_feat, action], dim=1)

        return self.q1_head(fused), self.q2_head(fused)

    def set_training_mode(self, mode: bool):
        self.train(mode)


class CustomSACPolicy(SACPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=CustomSACFeatureExtractor,
            **kwargs,
        )

    def make_actor(self, features_extractor=None):
        actor = CustomActor(self.observation_space, self.action_space).to(self.device)
        return actor

    def make_critic(self, features_extractor=None):
        critic = CustomCritic(self.observation_space, self.action_space).to(self.device)
        return critic