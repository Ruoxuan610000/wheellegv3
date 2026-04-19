from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers import SceneEntityCfg, RewardTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def rew_track_lin_vel_xy_enhanced(env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]

    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / (10*std))-1.0

def rew_base_height_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset = env.scene[asset_cfg.name]
    target_height = env.command_manager.get_command(command_name)
    if target_height.ndim > 1:
        target_height = target_height.squeeze(-1)

    height_error = torch.square(target_height - asset.data.root_pos_w[:, 2])

    return torch.exp(-height_error / std)

def rew_nominal_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")      
):
    asset = env.scene[asset_cfg.name]
    l1 = 150
    l2 = 280
    l5 = 150

    left_forw_id = asset.data.joint_names.index("left_forw_joint")
    left_back_id = asset.data.joint_names.index("left_back_joint")
    right_forw_id = asset.data.joint_names.index("right_forw_joint")
    right_back_id = asset.data.joint_names.index("right_back_joint")

    xc_l = l1 * torch.cos(asset.data.joint_pos[:, left_forw_id]) + l2 * torch.cos(asset.data.joint_pos[:, left_back_id])
    yc_l = l1 * torch.sin(asset.data.joint_pos[:, left_forw_id]) + l2 * torch.sin(asset.data.joint_pos[:, left_back_id])
    xc_r = l1 * torch.cos(asset.data.joint_pos[:, right_forw_id]) + l2 * torch.cos(asset.data.joint_pos[:, right_back_id])
    yc_r = l1 * torch.sin(asset.data.joint_pos[:, right_forw_id]) + l2 * torch.sin(asset.data.joint_pos[:, right_back_id])

    #L0_l = torch.sqrt(torch.square(xc_l - l5/2) + torch.square(yc_l))
    #L0_r = torch.sqrt(torch.square(xc_r - l5/2) + torch.square(yc_r))

    phi0_l = torch.arctan2(yc_l, xc_l-l5/2)
    phi0_r = torch.arctan2(yc_r, xc_r-l5/2)

    return torch.square(phi0_r - phi0_l)

def symmetry_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")      
):
    asset = env.scene[asset_cfg.name]
    #l1 = 150
    #l2 = 280
    #l5 = 150

    left_forw_id = asset.data.joint_names.index("left_forw_joint")
    left_back_id = asset.data.joint_names.index("left_back_joint")
    right_forw_id = asset.data.joint_names.index("right_forw_joint")
    right_back_id = asset.data.joint_names.index("right_back_joint")

    return torch.square(asset.data.joint_pos[:, left_forw_id] + asset.data.joint_pos[:, right_forw_id]) + torch.square(asset.data.joint_pos[:, left_back_id] + asset.data.joint_pos[:, right_back_id])

def rew_leg_joint_deviation_l2(env, asset_cfg):
    robot = env.scene[asset_cfg.name]

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, int):
        joint_ids = [joint_ids]

    q = robot.data.joint_pos[:, joint_ids]
    q0 = robot.data.default_joint_pos[:, joint_ids]

    return torch.sum((q - q0) ** 2, dim=-1)


def joint_pos_near_default_penalty(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    power: float = 2.0,
    normalize: bool = False,
):
    """Penalize joints that stay too close to their default positions.

    The penalty is active only when ``|q - q0| < threshold``. Inside that band,
    the closer the joint is to the default position, the larger the penalty.

    Typical usage is to pair this term with a negative reward weight.
    """
    robot = env.scene[asset_cfg.name]

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, int):
        joint_ids = [joint_ids]

    q = robot.data.joint_pos[:, joint_ids]
    q0 = robot.data.default_joint_pos[:, joint_ids]

    dist = torch.abs(q - q0)
    margin = torch.clamp(threshold - dist, min=0.0)

    if normalize and threshold > 0.0:
        margin = margin / threshold

    if power == 1.0:
        penalty = margin
    else:
        penalty = torch.pow(margin, power)

    return torch.sum(penalty, dim=-1)

class rew_action_acc_l2(ManagerTermBase):

    def __init__(self, cfg:RewardTermCfg, env):
        super().__init__(cfg, env)
        self.prev_prev_action = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim),
            device=env.device,
            dtype=torch.float32,
        )
    
    def reset(self, env_ids):
        if env_ids is None:
            env_ids = slice(None)
        self.prev_prev_action[env_ids] = 0.0

    def __call__(self, env) -> torch.Tensor:
        action = env.action_manager.action
        prev_action = env.action_manager.prev_action
        prev_prev_action = self.prev_prev_action

        action_acc = action - 2.0 * prev_action + prev_prev_action
        self.prev_prev_action[:] = prev_action

        return torch.sum(torch.square(action_acc), dim=1)

def base_pitch_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset = env.scene[asset_cfg.name]
    _, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.square(pitch)


def rew_base_height_level_exp(
    env: ManagerBasedRLEnv,
    std_height: float,
    std_pitch: float,
    pitch_deadband: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset = env.scene[asset_cfg.name]

    target_height = env.command_manager.get_command(command_name)
    if target_height.ndim > 1:
        target_height = target_height.squeeze(-1)

    current_height = asset.data.root_link_pos_w[:, 2]
    _, pitch, _ = euler_xyz_from_quat(asset.data.root_link_quat_w)

    height_error = torch.square(target_height - current_height)

    # 允许小范围 pitch，不然策略早期太僵
    pitch_violation = torch.clamp(torch.abs(pitch) - pitch_deadband, min=0.0)
    pitch_error = torch.square(pitch_violation)

    return torch.exp(-height_error / std_height - pitch_error / std_pitch)
