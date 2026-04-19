from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg, ManagerTermBase


class randomize_inertia_independent(ManagerTermBase):
    """复刻旧代码里 independent inertia scaling。"""

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        scale_range: tuple[float, float],
    ):
        robot: Articulation = env.scene[asset_cfg.name]

        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        inertias = robot.root_physx_view.get_inertias().clone()

        # 默认对所有body做缩放；你也可以只对 base body 做
        body_ids = torch.arange(robot.num_bodies, dtype=torch.int, device="cpu")
        scales = math_utils.sample_uniform(
            scale_range[0], scale_range[1],
            (len(env_ids), len(body_ids), 1),
            device="cpu"
        )
        inertias[env_ids[:, None], body_ids] = (
            robot.data.default_inertia[env_ids[:, None], body_ids] * scales
        )
        robot.root_physx_view.set_inertias(inertias, env_ids)


class randomize_default_joint_pos_offset(ManagerTermBase):
    """给每个env采样 default joint pos offset，供 action term 使用。"""

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        offset_range: tuple[float, float],
    ):
        robot: Articulation = env.scene[asset_cfg.name]
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=robot.device)

        if not hasattr(env, "default_joint_pos_offset"):
            env.default_joint_pos_offset = torch.zeros(
                (env.scene.num_envs, robot.num_joints),
                device=robot.device,
                dtype=robot.data.default_joint_pos.dtype,
            )
            
        offsets = torch.empty(
            (len(env_ids), robot.num_joints),
            device=robot.device
        ).uniform_(offset_range[0], offset_range[1])

        # 这里挂到 env 自己的缓存上，后续 action term 使用
        env.default_joint_pos_offset[env_ids] = offsets


def randomize_action_delay(
    env,
    env_ids: torch.Tensor | None,
    min_delay: int,
    max_delay: int,
    term_names: list[str] | tuple[str, ...],
):
    """Randomize control-step delays for delayed wheelleg action terms."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    if not hasattr(env, "action_delay_steps"):
        env.action_delay_steps = {}

    delays = torch.randint(min_delay, max_delay + 1, (len(env_ids),), device=env.device, dtype=torch.int)

    for term_name in term_names:
        term = env.action_manager._terms.get(term_name)
        if term is None:
            raise ValueError(f"Action term '{term_name}' was not found.")
        if not hasattr(term, "set_delay"):
            raise ValueError(f"Action term '{term_name}' does not support delayed actions.")
        term.set_delay(delays, env_ids)

        if term_name not in env.action_delay_steps:
            env.action_delay_steps[term_name] = torch.zeros(env.num_envs, device=env.device, dtype=torch.int)
        env.action_delay_steps[term_name][env_ids] = delays
