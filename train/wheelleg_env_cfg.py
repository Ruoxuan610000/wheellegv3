import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
import isaaclab_tasks.manager_based.user.wheelleg.mdp as mdp
from .wheelleg import WHEELLEG_CFG

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR


@configclass
class WheelLegSceneCfg(InteractiveSceneCfg):

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = WHEELLEG_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")

    contact_forces = ContactSensorCfg(
        # The imported wheelleg USD contains one nested root prim (for example `wheellegv3`)
        # under `robot`, so the runtime body path is `.../robot/<root>/base_link`.
        prim_path="{ENV_REGEX_NS}/robot/.*/.*",
        history_length=3,
        track_air_time=True,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/dome_light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

@configclass
class CommandsCfg:

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        # Keep commands around longer so the low-torque wheel actuator has time to settle.
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.35,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.25, 0.25),
            heading=(-math.pi, math.pi),
        ),
    )

    height_command = mdp.HeightCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0), 
        debug_vis=False,
        ranges=mdp.HeightCommandCfg.Ranges(
            height=(0.13, 0.13),
        ),
    )

@configclass
class ActionsCfg:

    leg_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["left_forw_joint", "left_back_joint", "right_forw_joint", "right_back_joint",],
        scale=1.2,
        clip={".*": (-1.2, 1.2)},
        use_default_offset=True,
        preserve_order=True,
        )
    
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["left_wheel_joint", "right_wheel_joint"],
        scale=40.0,
        clip={".*": (-40.0, 40.0)},
        use_default_offset=False,
        preserve_order=True,
        )


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel) #, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel) #, noise=Unoise(n_min=-0.2, n_max=0.2))
        base_hight = ObsTerm(func=mdp.base_pos_z) #, noise=Unoise(n_min=-0.01, n_max=0.01))
        projected_gravity = ObsTerm(func=mdp.projected_gravity) #, noise=Unoise(n_min=-0.05, n_max=0.05))

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        height_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "height_command"})


        joint_pos = ObsTerm(func=mdp.joint_pos_rel, 
                            params={
                                "asset_cfg": SceneEntityCfg(
                                    "robot", 
                                    joint_names=["left_forw_joint", "left_back_joint", "right_forw_joint", "right_back_joint"],
                                    preserve_order=True)
                                    }
                            ) #, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, 
                            params={
                                "asset_cfg": SceneEntityCfg(
                                    "robot",
                                    joint_names=["left_forw_joint", "left_back_joint", "right_forw_joint", "right_back_joint","left_wheel_joint", "right_wheel_joint"],
                                    preserve_order=True)
                                    }
                            ) #, noise=Unoise(n_min=-1.5, n_max=1.5))

        actions = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    robot_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "mass_distribution_params": (0.0, 0.1),
            "operation": "add",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {
                "x": (-0.002, 0.002),
                "y": (-0.002, 0.002),
                "z": (0.0, 0.005),
            },
        },
    )
    
    robot_inertia = EventTerm(
        func=mdp.randomize_inertia_independent,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "scale_range": (0.95, 1.05),
        },
    )

    robot_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.98, 1.02),
            "damping_distribution_params": (0.98, 1.02),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # 若你的 actuator / action term 支持 torque scale，也可挂这里
    default_joint_offset = EventTerm(
        func=mdp.randomize_default_joint_pos_offset,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "offset_range": (-0.01, 0.01),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (0.0, 0.0),
                "roll": (-0.01, 0.01),
                "pitch": (-0.02, 0.02),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
        },
    )

    """
    reset_leg_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["left_forw_joint", "left_back_joint", "right_forw_joint", "right_back_joint"],
                preserve_order=True,
            ),
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.2, 0.2),
        },
    )

    reset_wheel_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["left_wheel_joint", "right_wheel_joint"],
                preserve_order=True,
            ),
            "position_range": (0.0, 0.0),
            "velocity_range": (-0.1, 0.1),
        },
    )
    """

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(12.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.1, 0.1),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.1, 0.1),
            },
        },
    )

@configclass
class RewardsCfg:
    is_alive = RewTerm(func=mdp.is_alive, weight=0.5)
    
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, 
        weight=3.0, 
        params={"command_name": "base_velocity", "std":0.35} 
    )

    track_lin_vel_xy_exp_enhanced = RewTerm(
        func=mdp.rew_track_lin_vel_xy_enhanced, 
        weight=0.0, 
        params={"command_name": "base_velocity", "std":0.35} 
    )

    track_yaw_rate_l2 = RewTerm(
        func=mdp.track_ang_vel_z_exp, 
        weight=0.5, 
        params={"command_name": "base_velocity", "std":0.2}
    )
    
    balance_exp = RewTerm(func=mdp.flat_orientation_l2, weight=-8.0)
 
    base_height_reward = RewTerm(
        func=mdp.rew_base_height_exp,
        weight=3.0,
        params={"command_name": "height_command", "std": 0.0005},
    )

    vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0,)

    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)

    niminal_state = RewTerm(func=mdp.symmetry_state, weight=-1.0)

    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.2)

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.1,    
    )

    leg_joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2, 
        weight=-2e-4, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_forw_joint", "left_back_joint","right_forw_joint", "right_back_joint",], preserve_order=True,)},
    )

    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    collison = RewTerm(
        func=mdp.undesired_contacts, weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", "left_forw_link", "left_back_link", "right_forw_link", "right_back_link",
                                                                       "left_forw_p_link", "left_back_p_link", "right_forw_p_link", "right_back_p_link"]), 
            "threshold": 1.0
        },
    )

    joint_pos_l2 = RewTerm(func=mdp.joint_pos_limits, weight=-1.5)
    
    action_acc_l2 = RewTerm(func=mdp.rew_action_acc_l2, weight=-0.06)

@configclass
class TerminationsCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.9},
    )

@configclass
class TerrainCurriculumsCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

@configclass
class WheelLegEnvCfg(ManagerBasedRLEnvCfg):

    scene: WheelLegSceneCfg = WheelLegSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events:EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculums = TerrainCurriculumsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 20.0
        self.scene.robot.spawn.activate_contact_sensors = True
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        #if self.scene.height_scanner is not None:
        #    self.scene.height_scanner.update_period = self.decimation * self.sim.dt

@configclass
class WheelLegFlatEnvCfg(WheelLegEnvCfg):

    def __post_init__(self):

        super().__post_init__()
        """Post initialization."""
        self.curriculums.terrain_levels = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

@configclass
class WheelLegRoughEnvCfg(WheelLegEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.curriculums = TerrainCurriculumsCfg()

        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.005, 0.02)
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.005, 0.02)
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.005, 0.02)   
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].slope_range = (0.0, 0.22)
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].slope_range = (0.0, 0.2)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.005, 0.02)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

@configclass
class WheelLegRoughEnvCfg_PLAY(WheelLegRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.curriculums = TerrainCurriculumsCfg()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
