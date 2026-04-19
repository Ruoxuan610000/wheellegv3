from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions import joint_actions
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.buffers import DelayBuffer
from isaaclab.utils import configclass


class JointPositionActionWithOffset(joint_actions.JointPositionAction):
    """Joint position action that also applies per-env startup joint offsets when available."""

    cfg: "JointPositionActionWithOffsetCfg"

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)

        if hasattr(self._env, "default_joint_pos_offset"):
            self._processed_actions = self._processed_actions + self._env.default_joint_pos_offset[:, self._joint_ids]

            if self.cfg.clip is not None:
                self._processed_actions = torch.clamp(
                    self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
                )


@configclass
class JointPositionActionWithOffsetCfg(JointPositionActionCfg):
    """Configuration for :class:`JointPositionActionWithOffset`."""

    class_type: type[ActionTerm] = JointPositionActionWithOffset


class JointPositionActionWithOffsetAndDelay(JointPositionActionWithOffset):
    """Joint position action with startup offsets and per-environment control delay."""

    cfg: "JointPositionActionWithOffsetAndDelayCfg"

    def __init__(self, cfg: "JointPositionActionWithOffsetAndDelayCfg", env):
        super().__init__(cfg, env)
        self._delay_buffer = DelayBuffer(cfg.max_delay, self.num_envs, device=self.device)
        self.set_delay(cfg.min_delay)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._processed_actions = self._delay_buffer.compute(self._processed_actions)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._delay_buffer.reset(env_ids)

    def set_delay(self, delay: int | torch.Tensor, env_ids=None):
        self._delay_buffer.set_time_lag(delay, env_ids)


@configclass
class JointPositionActionWithOffsetAndDelayCfg(JointPositionActionWithOffsetCfg):
    """Configuration for :class:`JointPositionActionWithOffsetAndDelay`."""

    class_type: type[ActionTerm] = JointPositionActionWithOffsetAndDelay
    min_delay: int = 0
    max_delay: int = 0


class JointVelocityActionWithDelay(joint_actions.JointVelocityAction):
    """Joint velocity action with per-environment control delay."""

    cfg: "JointVelocityActionWithDelayCfg"

    def __init__(self, cfg: "JointVelocityActionWithDelayCfg", env):
        super().__init__(cfg, env)
        self._delay_buffer = DelayBuffer(cfg.max_delay, self.num_envs, device=self.device)
        self.set_delay(cfg.min_delay)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._processed_actions = self._delay_buffer.compute(self._processed_actions)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._delay_buffer.reset(env_ids)

    def set_delay(self, delay: int | torch.Tensor, env_ids=None):
        self._delay_buffer.set_time_lag(delay, env_ids)


@configclass
class JointVelocityActionWithDelayCfg(JointVelocityActionCfg):
    """Configuration for :class:`JointVelocityActionWithDelay`."""

    class_type: type[ActionTerm] = JointVelocityActionWithDelay
    min_delay: int = 0
    max_delay: int = 0
