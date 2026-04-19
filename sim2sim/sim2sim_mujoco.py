import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def resolve_default_policy_path(isaaclab_wheelleg_log_dir: Path, fallback_policy_path: Path):
    exported_policies = sorted(isaaclab_wheelleg_log_dir.glob("*/exported/policy.pt"))
    if exported_policies:
        return exported_policies[-1]
    return fallback_policy_path


@dataclass(frozen=True)
class RunConfig:
    base_dir: Path = Path(__file__).resolve().parent
    isaaclab_wheelleg_log_dir: Path = Path("/home/li/IsaacLab/logs/rsl_rl/wheelleg")
    fallback_policy_path: Path | None = None
    model_path: Path | None = None
    policy_path: Path | None = None
    report_path: Path | None = None
    base_body_name: str = "base_link"
    policy_joint_order: tuple[str, ...] = (
        "left_forw_joint",
        "left_back_joint",
        "right_forw_joint",
        "right_back_joint",
        "left_wheel_joint",
        "right_wheel_joint",
    )
    actuator_order: tuple[str, ...] = (
        "left_forw",
        "left_back",
        "right_forw",
        "right_back",
        "left_wheel",
        "right_wheel",
    )
    obs_dim: int = 3 + 3 + 1 + 3 + 3 + 1 + 4 + 6 + 6
    act_dim: int = 6
    leg_action_dim: int = 4
    leg_action_scale: float = 0.85
    wheel_action_scale: float = 40.0
    leg_pos_clip: tuple[float, float] = (-1.25, 1.25)
    wheel_vel_clip: tuple[float, float] = (-40.0, 40.0)
    leg_kp: float = 8.0
    leg_kd: float = 0.8
    wheel_kd: float = 6e-4
    sim_time: float = 20.0
    decimation: int = 2
    cmd_x: float = 0.5
    cmd_y: float = 0.0
    cmd_yaw: float = 0.0
    height_cmd: float = 0.13
    use_training_heading_command: bool = True
    heading_control_stiffness: float = 0.35
    heading_velocity_limit: float = 1.0
    headless: bool = False
    realtime_scale: float = 0.0
    print_every: int = 200
    def __post_init__(self):
        fallback_policy_path = self.fallback_policy_path or (self.base_dir / "models" / "policy.pt")
        model_path = self.model_path or (self.base_dir / "wheellegv3" / "urdf" / "wheellegv3.xml")
        report_path = self.report_path or (self.base_dir / "outputs" / "wheelleg_tracking_report.png")
        policy_path = self.policy_path or resolve_default_policy_path(
            self.isaaclab_wheelleg_log_dir, fallback_policy_path
        )

        object.__setattr__(self, "fallback_policy_path", fallback_policy_path)
        object.__setattr__(self, "model_path", model_path)
        object.__setattr__(self, "report_path", report_path)
        object.__setattr__(self, "policy_path", policy_path)


RUN_CONFIG = RunConfig()


class Policy(nn.Module):
    """Fallback policy structure for non-TorchScript checkpoints."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, act_dim),
        )
        self.init_weights()

    def init_weights(self):
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, obs):
        return self.actor(obs)


def load_policy(policy_path: Path, obs_dim: int, act_dim: int):
    """Load a policy exported either as TorchScript or state_dict."""
    policy_path = Path(policy_path)

    try:
        policy = torch.jit.load(str(policy_path), map_location="cpu")
        policy.eval()
        return policy
    except Exception:
        pass

    policy = Policy(obs_dim, act_dim)
    state_dict = torch.load(str(policy_path), map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict):
        policy.load_state_dict(state_dict)
        policy.eval()
        return policy

    policy = state_dict
    policy.eval()
    return policy


def projected_gravity_body(model, data, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in MuJoCo model")

    rotation_world_body = data.xmat[body_id].reshape(3, 3)
    gravity_world = np.array(model.opt.gravity, dtype=np.float64)
    gravity_norm = np.linalg.norm(gravity_world)
    if gravity_norm > 1e-8:
        gravity_world = gravity_world / gravity_norm
    return rotation_world_body.T @ gravity_world


def body_velocity_local(model, data, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in MuJoCo model")

    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 1)
    ang_vel = velocity[:3]
    lin_vel = velocity[3:]
    return lin_vel, ang_vel


def build_joint_state(model, joint_names):
    qpos_indices = []
    qvel_indices = []
    default_qpos = []

    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint '{joint_name}' not found in MuJoCo model")
        qpos_adr = model.jnt_qposadr[joint_id]
        qvel_adr = model.jnt_dofadr[joint_id]
        qpos_indices.append(qpos_adr)
        qvel_indices.append(qvel_adr)
        default_qpos.append(model.qpos0[qpos_adr])

    return {
        "joint_names": tuple(joint_names),
        "qpos_indices": np.array(qpos_indices, dtype=np.int32),
        "qvel_indices": np.array(qvel_indices, dtype=np.int32),
        "default_qpos": np.array(default_qpos, dtype=np.float32),
    }


def build_actuator_state(model, actuator_names):
    actuator_ids = []

    for actuator_name in actuator_names:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            raise ValueError(f"Actuator '{actuator_name}' not found in MuJoCo model")
        actuator_ids.append(actuator_id)

    actuator_ids = np.array(actuator_ids, dtype=np.int32)
    return {
        "actuator_names": tuple(actuator_names),
        "actuator_ids": actuator_ids,
        "force_ranges": model.actuator_forcerange[actuator_ids].astype(np.float32).copy(),
    }


def get_observation(data, model, joint_state, command, last_actions, config):
    """Build the 30D policy observation in the IsaacLab training order."""
    base_lin_vel_body, base_ang_vel_body = body_velocity_local(model, data, config.base_body_name)
    base_height = data.qpos[2:3]
    gravity_body = projected_gravity_body(model, data, config.base_body_name)

    joint_pos_rel = data.qpos[joint_state["qpos_indices"]] - joint_state["default_qpos"]
    joint_vel_rel = data.qvel[joint_state["qvel_indices"]]
    leg_joint_pos_rel = joint_pos_rel[: config.leg_action_dim]

    obs = np.concatenate(
        [
            base_lin_vel_body,
            base_ang_vel_body,
            base_height,
            gravity_body,
            command["base_velocity"],
            command["height"],
            leg_joint_pos_rel,
            joint_vel_rel,
            last_actions,
        ]
    )
    obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
    return np.clip(obs, -100.0, 100.0).astype(np.float32)


def process_actions(raw_actions, joint_state, config):
    # RSL-RL clips the policy action to [-1, 1] before it reaches the action terms.
    clipped_actions = np.nan_to_num(raw_actions, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
    clipped_actions = np.clip(clipped_actions, -1.0, 1.0)

    leg_pos_targets = (
        joint_state["default_qpos"][: config.leg_action_dim]
        + config.leg_action_scale * clipped_actions[: config.leg_action_dim]
    )
    leg_pos_targets = np.clip(leg_pos_targets, *config.leg_pos_clip)

    wheel_vel_targets = np.clip(
        config.wheel_action_scale * clipped_actions[config.leg_action_dim :],
        *config.wheel_vel_clip,
    )

    return clipped_actions, leg_pos_targets.astype(np.float32), wheel_vel_targets.astype(np.float32)


def compute_motor_torques(data, joint_state, actuator_state, leg_pos_targets, wheel_vel_targets, config):
    joint_pos = data.qpos[joint_state["qpos_indices"]]
    joint_vel = data.qvel[joint_state["qvel_indices"]]

    torques = np.zeros(config.act_dim, dtype=np.float32)
    torques[: config.leg_action_dim] = (
        config.leg_kp * (leg_pos_targets - joint_pos[: config.leg_action_dim])
        - config.leg_kd * joint_vel[: config.leg_action_dim]
    )
    torques[config.leg_action_dim :] = config.wheel_kd * (
        wheel_vel_targets - joint_vel[config.leg_action_dim :]
    )

    force_low = actuator_state["force_ranges"][:, 0]
    force_high = actuator_state["force_ranges"][:, 1]
    return np.clip(torques, force_low, force_high)


def apply_motor_torques(data, actuator_state, torques):
    data.ctrl[actuator_state["actuator_ids"]] = torques
    return torques


def make_command(config):
    return {
        "base_velocity": np.array([config.cmd_x, config.cmd_y, config.cmd_yaw], dtype=np.float32),
        "height": np.array([config.height_cmd], dtype=np.float32),
    }


def base_heading_yaw(data):
    quat = data.qpos[3:7]
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def update_command(command, data, heading_target, config, control_dt):
    if not config.use_training_heading_command:
        command["base_velocity"][2] = config.cmd_yaw
        return heading_target

    heading_target = wrap_to_pi(heading_target + config.cmd_yaw * control_dt)
    heading_error = wrap_to_pi(heading_target - base_heading_yaw(data))
    command["base_velocity"][2] = np.clip(
        config.heading_control_stiffness * heading_error,
        -config.heading_velocity_limit,
        config.heading_velocity_limit,
    )
    return heading_target


def upright_tilt_angle(model, data, body_name):
    gravity_body = projected_gravity_body(model, data, body_name)
    return math.acos(float(np.clip(-gravity_body[2], -1.0, 1.0)))


def init_tracking_history():
    return {
        "time": [],
        "cmd_vx": [],
        "cmd_vy": [],
        "cmd_wz": [],
        "actual_vx": [],
        "actual_vy": [],
        "actual_wz": [],
        "cmd_height": [],
        "actual_height": [],
        "tilt_deg": [],
        "ang_vel_xy_norm": [],
        "left_wheel_vel_target": [],
        "right_wheel_vel_target": [],
        "left_wheel_vel_actual": [],
        "right_wheel_vel_actual": [],
    }


def record_tracking_metrics(history, model, data, command, joint_state, wheel_vel_targets, config):
    base_lin_vel_body, base_ang_vel_body = body_velocity_local(model, data, config.base_body_name)
    tilt_deg = math.degrees(upright_tilt_angle(model, data, config.base_body_name))
    wheel_vel_actual = data.qvel[joint_state["qvel_indices"][config.leg_action_dim :]].astype(np.float32)

    history["time"].append(float(data.time))
    history["cmd_vx"].append(float(command["base_velocity"][0]))
    history["cmd_vy"].append(float(command["base_velocity"][1]))
    history["cmd_wz"].append(float(command["base_velocity"][2]))
    history["actual_vx"].append(float(base_lin_vel_body[0]))
    history["actual_vy"].append(float(base_lin_vel_body[1]))
    history["actual_wz"].append(float(base_ang_vel_body[2]))
    history["cmd_height"].append(float(command["height"][0]))
    history["actual_height"].append(float(data.qpos[2]))
    history["tilt_deg"].append(float(tilt_deg))
    history["ang_vel_xy_norm"].append(float(np.linalg.norm(base_ang_vel_body[:2])))
    history["left_wheel_vel_target"].append(float(wheel_vel_targets[0]))
    history["right_wheel_vel_target"].append(float(wheel_vel_targets[1]))
    history["left_wheel_vel_actual"].append(float(wheel_vel_actual[0]))
    history["right_wheel_vel_actual"].append(float(wheel_vel_actual[1]))


def save_tracking_report(report_path, history, summary):
    if not history["time"]:
        print("report skipped: no tracking history collected")
        return

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    time_axis = np.asarray(history["time"], dtype=np.float32)
    cmd_vx = np.asarray(history["cmd_vx"], dtype=np.float32)
    cmd_vy = np.asarray(history["cmd_vy"], dtype=np.float32)
    cmd_wz = np.asarray(history["cmd_wz"], dtype=np.float32)
    actual_vx = np.asarray(history["actual_vx"], dtype=np.float32)
    actual_vy = np.asarray(history["actual_vy"], dtype=np.float32)
    actual_wz = np.asarray(history["actual_wz"], dtype=np.float32)
    cmd_height = np.asarray(history["cmd_height"], dtype=np.float32)
    actual_height = np.asarray(history["actual_height"], dtype=np.float32)
    tilt_deg = np.asarray(history["tilt_deg"], dtype=np.float32)
    ang_vel_xy_norm = np.asarray(history["ang_vel_xy_norm"], dtype=np.float32)
    left_wheel_vel_target = np.asarray(history["left_wheel_vel_target"], dtype=np.float32)
    right_wheel_vel_target = np.asarray(history["right_wheel_vel_target"], dtype=np.float32)
    left_wheel_vel_actual = np.asarray(history["left_wheel_vel_actual"], dtype=np.float32)
    right_wheel_vel_actual = np.asarray(history["right_wheel_vel_actual"], dtype=np.float32)

    vel_rmse = float(np.sqrt(np.mean((actual_vx - cmd_vx) ** 2 + (actual_vy - cmd_vy) ** 2)))
    yaw_rmse = float(np.sqrt(np.mean((actual_wz - cmd_wz) ** 2)))
    height_rmse = float(np.sqrt(np.mean((actual_height - cmd_height) ** 2)))
    max_tilt = float(np.max(tilt_deg))
    left_wheel_vel_rmse = float(np.sqrt(np.mean((left_wheel_vel_actual - left_wheel_vel_target) ** 2)))
    right_wheel_vel_rmse = float(np.sqrt(np.mean((right_wheel_vel_actual - right_wheel_vel_target) ** 2)))
    wheel_vel_rmse = max(left_wheel_vel_rmse, right_wheel_vel_rmse)

    fig, axes = plt.subplots(6, 1, figsize=(12, 16), constrained_layout=True)

    axes[0].plot(time_axis, cmd_vx, label="cmd vx", color="#1f77b4", linestyle="--")
    axes[0].plot(time_axis, actual_vx, label="actual vx", color="#1f77b4")
    axes[0].plot(time_axis, cmd_vy, label="cmd vy", color="#ff7f0e", linestyle="--")
    axes[0].plot(time_axis, actual_vy, label="actual vy", color="#ff7f0e")
    axes[0].set_ylabel("m/s")
    axes[0].set_title(f"Base Velocity Tracking  RMSE={vel_rmse:.4f}")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(ncol=4, fontsize=9)

    axes[1].plot(time_axis, cmd_wz, label="cmd wz", color="#2ca02c", linestyle="--")
    axes[1].plot(time_axis, actual_wz, label="actual wz", color="#2ca02c")
    axes[1].set_ylabel("rad/s")
    axes[1].set_title(f"Yaw Rate Tracking  RMSE={yaw_rmse:.4f}")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=9)

    axes[2].plot(time_axis, cmd_height, label="cmd height", color="#d62728", linestyle="--")
    axes[2].plot(time_axis, actual_height, label="actual height", color="#d62728")
    axes[2].set_ylabel("m")
    axes[2].set_title(f"Base Height Tracking  RMSE={height_rmse:.4f}")
    axes[2].grid(True, linestyle="--", alpha=0.4)
    axes[2].legend(fontsize=9)

    axes[3].plot(time_axis, tilt_deg, label="tilt angle", color="#9467bd")
    axes[3].set_ylabel("deg")
    axes[3].set_title(f"Base Link Balance  max tilt={max_tilt:.2f} deg")
    axes[3].grid(True, linestyle="--", alpha=0.4)
    balance_axis = axes[3].twinx()
    balance_axis.plot(time_axis, ang_vel_xy_norm, label="|omega_xy|", color="#8c564b", alpha=0.85)
    balance_axis.set_ylabel("rad/s")

    handles_left, labels_left = axes[3].get_legend_handles_labels()
    handles_right, labels_right = balance_axis.get_legend_handles_labels()
    axes[3].legend(handles_left + handles_right, labels_left + labels_right, fontsize=9, loc="upper right")

    axes[4].plot(time_axis, left_wheel_vel_target, label="target speed", color="#17becf", linestyle="--")
    axes[4].plot(time_axis, left_wheel_vel_actual, label="actual speed", color="#1f77b4")
    axes[4].legend(fontsize=9)
    axes[4].set_ylabel("rad/s")
    axes[4].set_title(f"Left Leg Hub Motor Speed  RMSE={left_wheel_vel_rmse:.4f}")
    axes[4].grid(True, linestyle="--", alpha=0.4)

    axes[5].plot(time_axis, right_wheel_vel_target, label="target speed", color="#bcbd22", linestyle="--")
    axes[5].plot(time_axis, right_wheel_vel_actual, label="actual speed", color="#ff7f0e")
    axes[5].legend(fontsize=9)
    axes[5].set_ylabel("rad/s")
    axes[5].set_title(f"Right Leg Hub Motor Speed  RMSE={right_wheel_vel_rmse:.4f}")
    axes[5].grid(True, linestyle="--", alpha=0.4)
    axes[5].set_xlabel("time [s]")

    fig.suptitle(
        f"WheelLeg {summary['duration_s']:.2f}s Command Tracking and Balance Report\n"
        f"unstable={summary['unstable']}  avg_body_vx={summary['avg_body_vx']:.4f}  avg_z={summary['avg_z']:.4f}  wheel_rmse={wheel_vel_rmse:.4f}",
        fontsize=14,
    )
    fig.savefig(report_path, dpi=160)
    plt.close(fig)
    print(f"report saved: {report_path}")


def simulate(model, data, policy, joint_state, actuator_state, command, config, viewer=None):
    dt = model.opt.timestep
    steps = int(config.sim_time / dt)
    control_dt = config.decimation * dt
    last_actions = np.zeros(config.act_dim, dtype=np.float32)
    wheel_vel_targets = np.zeros(config.act_dim - config.leg_action_dim, dtype=np.float32)
    heading_target = base_heading_yaw(data)

    x_history = []
    body_vx_history = []
    z_history = []
    tracking_history = init_tracking_history()
    unstable = False

    for step in range(steps):
        if step % config.decimation == 0:
            heading_target = update_command(command, data, heading_target, config, control_dt)
            obs = get_observation(
                data,
                model,
                joint_state,
                command,
                last_actions,
                config,
            )
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            with torch.no_grad():
                raw_actions = policy(obs_tensor).squeeze(0).cpu().numpy()

            last_actions, leg_pos_targets, wheel_vel_targets = process_actions(raw_actions, joint_state, config)
            torques = compute_motor_torques(
                data, joint_state, actuator_state, leg_pos_targets, wheel_vel_targets, config
            )
            apply_motor_torques(data, actuator_state, torques)

        mujoco.mj_step(model, data)
        x_history.append(float(data.qpos[0]))
        body_vx_history.append(float(body_velocity_local(model, data, config.base_body_name)[0][0]))
        z_history.append(float(data.qpos[2]))
        record_tracking_metrics(tracking_history, model, data, command, joint_state, wheel_vel_targets, config)

        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            unstable = True
            print(f"unstable step={step:5d} reason=nonfinite")
            break
        if abs(float(data.qpos[2])) > 5.0 or np.max(np.abs(data.qvel)) > 1.0e4:
            unstable = True
            print(f"unstable step={step:5d} reason=diverged z={data.qpos[2]:.4f}")
            break

        if config.print_every > 0 and step % config.print_every == 0:
            print(
                f"step={step:5d} "
                f"x={data.qpos[0]: .4f} "
                f"z={data.qpos[2]: .4f} "
                f"vx={data.qvel[0]: .4f} "
                f"tau={np.array2string(data.ctrl[actuator_state['actuator_ids']], precision=3)}"
            )

        if viewer is not None:
            if config.realtime_scale > 0.0:
                time.sleep(dt * config.realtime_scale)
            viewer.cam.lookat[:] = data.qpos[:3].astype(np.float32)
            viewer.sync()

    avg_vx = 0.0 if len(x_history) < 2 else (x_history[-1] - x_history[0]) / (len(x_history) * dt)
    avg_body_vx = float(np.mean(body_vx_history)) if body_vx_history else 0.0
    avg_z = float(np.mean(z_history)) if z_history else float(data.qpos[2])
    tilt = upright_tilt_angle(model, data, config.base_body_name)

    print(
        "summary "
        f"final_pos={np.array2string(data.qpos[:3], precision=4)} "
        f"final_quat={np.array2string(data.qpos[3:7], precision=4)} "
        f"tilt_rad={tilt:.4f} "
        f"avg_vx={avg_vx:.4f} "
        f"avg_body_vx={avg_body_vx:.4f} "
        f"avg_z={avg_z:.4f} "
        f"unstable={unstable}"
    )
    return {
        "history": tracking_history,
        "final_pos": data.qpos[:3].copy(),
        "final_quat": data.qpos[3:7].copy(),
        "duration_s": float(data.time),
        "tilt_rad": tilt,
        "avg_vx": avg_vx,
        "avg_body_vx": avg_body_vx,
        "avg_z": avg_z,
        "unstable": unstable,
    }


def main():
    config = RUN_CONFIG

    if not config.model_path.exists():
        raise FileNotFoundError(f"MuJoCo model file not found: {config.model_path}")
    if not config.policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {config.policy_path}")

    model = mujoco.MjModel.from_xml_path(str(config.model_path))
    data = mujoco.MjData(model)
    policy = load_policy(config.policy_path, config.obs_dim, config.act_dim)
    joint_state = build_joint_state(model, config.policy_joint_order)
    actuator_state = build_actuator_state(model, config.actuator_order)
    command = make_command(config)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    if hasattr(policy, "reset"):
        policy.reset()

    print(
        "start "
        f"policy={config.policy_path.name} "
        f"model={config.model_path.name} "
        f"report={config.report_path.name} "
        f"cmd={command['base_velocity'].tolist()} "
        f"height={command['height'].item():.3f} "
        f"dt={model.opt.timestep:.4f} "
        f"decimation={config.decimation} "
        f"leg_pd=({config.leg_kp:.2f},{config.leg_kd:.2f}) "
        f"wheel_kd={config.wheel_kd:.2f}"
    )

    if config.headless:
        summary = simulate(model, data, policy, joint_state, actuator_state, command, config, viewer=None)
        save_tracking_report(config.report_path, summary["history"], summary)
        return

    with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
        viewer.cam.elevation = -20
        summary = simulate(model, data, policy, joint_state, actuator_state, command, config, viewer=viewer)

    save_tracking_report(config.report_path, summary["history"], summary)


if __name__ == "__main__":
    main()
