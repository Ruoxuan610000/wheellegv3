from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.envs import mdp
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

def override_after(self, env, env_ids, old_value, *args, value=None, num_steps: int = 0, **kwargs):
    if env.common_step_counter > num_steps:
        return value
    return mdp.modify_term_cfg.NO_CHANGE

class HeightCommand(CommandTerm):
    """为机器人生成目标高度命令，shape = (num_envs, 1)."""

    cfg: "HeightCommandCfg"

    def __init__(self, cfg: "HeightCommandCfg", env):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # command buffer: (num_envs, 1)
        self._command = torch.zeros(self.num_envs, 1, device=self.device)

        # 可选 metrics，方便 log
        self.metrics["height_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: torch.Tensor):
        """给指定 env 重采样目标高度."""
        low, high = self.cfg.ranges.height
        self._command[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(low, high)

    def _update_command(self):
        """每步更新 metrics；高度命令本身这里不做动态变化."""
        # 这里默认取 root z 作为机器人当前高度
        current_height = self.robot.data.root_pos_w[:, 2]
        self.metrics["height_error"] = torch.abs(current_height - self._command[:, 0])

    def _update_metrics(self):
        """有些版本会调用这个接口；保留兼容."""
        current_height = self.robot.data.root_pos_w[:, 2]
        self.metrics["height_error"] = torch.abs(current_height - self._command[:, 0])
        

@configclass
class HeightCommandCfg(CommandTermCfg):
    """HeightCommand 的配置."""

    class_type: type = HeightCommand

    asset_name: str = MISSING
    """scene 里的机器人名字，比如 'robot'."""

    @configclass
    class Ranges:
        height: tuple[float, float] = MISSING
        """目标高度采样范围，单位 m."""

    ranges: Ranges = MISSING