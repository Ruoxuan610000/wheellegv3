from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def joint_acc(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return joint accelerations for the selected joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


def default_joint_pos_offset(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the randomized default joint offsets if they exist."""
    asset: Articulation = env.scene[asset_cfg.name]
    if hasattr(env, "default_joint_pos_offset"):
        return env.default_joint_pos_offset[:, asset_cfg.joint_ids]
    return torch.zeros((env.num_envs, len(asset_cfg.joint_ids)), device=env.device)


def action_delay(env, term_names: list[str] | tuple[str, ...] | None = None) -> torch.Tensor:
    """Return the configured action delay steps for delayed action terms."""
    if term_names is None:
        if hasattr(env, "action_delay_steps"):
            term_names = list(env.action_delay_steps.keys())
        else:
            term_names = []

    if not term_names:
        return torch.zeros((env.num_envs, 1), device=env.device)

    delays = []
    for term_name in term_names:
        if hasattr(env, "action_delay_steps") and term_name in env.action_delay_steps:
            delays.append(env.action_delay_steps[term_name].float().unsqueeze(-1))
        else:
            delays.append(torch.zeros((env.num_envs, 1), device=env.device))
    return torch.cat(delays, dim=-1)
