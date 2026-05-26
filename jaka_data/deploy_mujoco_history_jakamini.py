import re
import time
import mujoco, mujoco_viewer, mujoco.viewer
import numpy as np
import torch
import onnxruntime as ort
import argparse
from enum import Enum
from collections import deque


def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def quat_to_rot6d(q):
    r, i, j, k = q[0], q[1], q[2], q[3]
    two_s = 2.0 / (r * r + i * i + j * j + k * k)
    ii = i * i
    jj = j * j
    kk = k * k
    ij = i * j
    kr = k * r
    ik = i * k
    jr = j * r
    jk = j * k
    ir = i * r
    return np.array([
        1 - two_s * (jj + kk),  # R00
        two_s * (ij - kr),  # R01
        two_s * (ij + kr),  # R10
        1 - two_s * (ii + kk),  # R11
        two_s * (ik - jr),  # R20
        two_s * (jk + ir)  # R21
    ])


def quat_apply_inverse(quat, vec):
    xyz = quat[1:]
    w = quat[0]
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)


def quat_rotate_inverse_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    a = v * np.expand_dims(2.0 * q_w ** 2 - 1.0, axis=-1)
    b = np.cross(q_vec, v, axis=-1) * np.expand_dims(q_w, axis=-1) * 2.0
    dot_product = np.expand_dims(np.einsum('...i,...i->...', q_vec, v), axis=-1)
    c = q_vec * dot_product * 2.0
    return a - b + c


def quaternion_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_inv_np(q):
    conj = quaternion_conjugate(q)
    norm_sq = np.sum(q ** 2)
    inv_norm_sq = 1 / max(norm_sq, 1e-9)
    return conj * inv_norm_sq


def quat_mul_np(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def yaw_quat(q):
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def quatToEuler(quat):
    eulerVec = np.zeros(3)
    qw, qx, qy, qz = quat
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        eulerVec[1] = np.copysign(np.pi / 2, sinp)
    else:
        eulerVec[1] = np.arcsin(sinp)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)
    return eulerVec


def subtract_frame_transforms(t01, q01, t02, q02):
    q10 = quat_inv_np(q01)
    if q02 is not None:
        q12 = quat_mul_np(q10, q02)
    else:
        q12 = q10
    return None, q12


class AnchorBody(Enum):
    WAIST_YAW_LINK = 3  # BFS index: base_link(0) -> Left_hip_pitch(1) -> Right_hip_pitch(2) -> waist_yaw(3)


anchor_body = AnchorBody.WAIST_YAW_LINK

# IsaacLab joint names (from jaka.py Khan_mini_JOINT_NAMES_DEPLOY)
# isaaclab_joint_names = [
#     "Left_hip_pitch_joint", "Right_hip_pitch_joint", "waist_yaw_joint",
#     "Left_hip_roll_joint", "Right_hip_roll_joint", "Left_shoulder_pitch_joint", "Right_shoulder_pitch_joint", "Neck_yaw_joint",
#     "Left_hip_yaw_joint", "Right_hip_yaw_joint", "Left_shoulder_roll_joint", "Right_shoulder_roll_joint", "Neck_pitch_joint",
#     "Left_knee_joint", "Right_knee_joint", "Left_shoulder_yaw_joint", "Right_shoulder_yaw_joint",
#     "Left_ankle_pitch_joint", "Right_ankle_pitch_joint", "Left_elbow_joint", "Right_elbow_joint",
#     "Left_ankle_roll_joint", "Right_ankle_roll_joint", "Left_wrist_roll_joint", "Right_wrist_roll_joint",
#     "Left_wrist_yaw_joint", "Right_wrist_yaw_joint"
# ]

isaaclab_joint_names = ['Left_hip_pitch_joint', 'Right_hip_pitch_joint', 'waist_yaw_joint',
                        'Left_hip_roll_joint', 'Right_hip_roll_joint', 'Left_shoulder_pitch_joint', 'Neck_yaw_joint',
                        'Right_shoulder_pitch_joint',
                        'Left_hip_yaw_joint', 'Right_hip_yaw_joint', 'Left_shoulder_roll_joint', 'Neck_pitch_joint',
                        'Right_shoulder_roll_joint', 'Left_knee_joint', 'Right_knee_joint', 'Left_shoulder_yaw_joint',
                        'Right_shoulder_yaw_joint',
                        'Left_ankle_pitch_joint', 'Right_ankle_pitch_joint', 'Left_elbow_joint', 'Right_elbow_joint',
                        'Left_ankle_roll_joint',
                        'Right_ankle_roll_joint', 'Left_wrist_roll_joint', 'Right_wrist_roll_joint',
                        'Left_wrist_yaw_joint', 'Right_wrist_yaw_joint']

# MuJoCo joint names (from Khan_mini_simplified.xml order)
mujoco_joint_names = [
    "Left_hip_pitch_joint", "Left_hip_roll_joint", "Left_hip_yaw_joint", "Left_knee_joint", "Left_ankle_pitch_joint",
    "Left_ankle_roll_joint",
    "Right_hip_pitch_joint", "Right_hip_roll_joint", "Right_hip_yaw_joint", "Right_knee_joint",
    "Right_ankle_pitch_joint", "Right_ankle_roll_joint",
    "waist_yaw_joint",
    "Left_shoulder_pitch_joint", "Left_shoulder_roll_joint", "Left_shoulder_yaw_joint", "Left_elbow_joint",
    "Left_wrist_roll_joint", "Left_wrist_yaw_joint",
    "Right_shoulder_pitch_joint", "Right_shoulder_roll_joint", "Right_shoulder_yaw_joint", "Right_elbow_joint",
    "Right_wrist_roll_joint", "Right_wrist_yaw_joint",
    "Neck_yaw_joint", "Neck_pitch_joint"
]

# Stiffness and Damping based on jaka.py Khan_mini_CFG
# stiffness_dict = {
#     ".*_hip_pitch_joint": 187.0,
#     ".*_hip_roll_joint": 187.0,
#     ".*_hip_yaw_joint": 187.0,
#     ".*_knee_joint": 187.0,
#     ".*_ankle_pitch_joint": 28.50124619574858,
#     ".*_ankle_roll_joint": 28.50124619574858,
#     "waist_yaw_joint": 187.0,
#     ".*_shoulder_pitch_joint": 158.0,
#     ".*_shoulder_roll_joint": 158.0,
#     ".*_shoulder_yaw_joint": 40.8,
#     ".*_elbow_joint": 40.8,
#     ".*_wrist_roll_joint": 13.4,
#     ".*_wrist_yaw_joint": 13.4,
#     "Neck_yaw_joint": 13.4,
#     "Neck_pitch_joint": 13.4,
# }
#
# damping_dict = {
#     ".*_hip_pitch_joint": 18.7,
#     ".*_hip_roll_joint": 18.7,
#     ".*_hip_yaw_joint": 18.7,
#     ".*_knee_joint": 18.7,
#     ".*_ankle_pitch_joint": 1.814445686584846,
#     ".*_ankle_roll_joint": 1.814445686584846,
#     "waist_yaw_joint": 18.7,
#     ".*_shoulder_pitch_joint": 15.8,
#     ".*_shoulder_roll_joint": 15.8,
#     ".*_shoulder_yaw_joint": 4.0,
#     ".*_elbow_joint": 4.0,
#     ".*_wrist_roll_joint": 1.3,
#     ".*_wrist_yaw_joint": 1.3,
#     "Neck_yaw_joint": 1.34,
#     "Neck_pitch_joint": 1.34,
# }

stiffness_dict = {
    ".*_hip_pitch_joint": 187.0,
    ".*_hip_roll_joint": 187.0,
    ".*_hip_yaw_joint": 187.0,
    ".*_knee_joint": 187.0,
    ".*_ankle_pitch_joint": 100,
    ".*_ankle_roll_joint": 50,
    "waist_yaw_joint": 187.0,
    ".*_shoulder_pitch_joint": 102.0,
    ".*_shoulder_roll_joint": 102.0,
    ".*_shoulder_yaw_joint": 40.8,
    ".*_elbow_joint": 40.8,
    ".*_wrist_roll_joint": 6.7,
    ".*_wrist_yaw_joint": 6.7,
    "Neck_yaw_joint": 6.7,
    "Neck_pitch_joint": 6.7,
}

damping_dict = {
    ".*_hip_pitch_joint": 18.7,
    ".*_hip_roll_joint": 18.7,
    ".*_hip_yaw_joint": 18.7,
    ".*_knee_joint": 18.7,
    ".*_ankle_pitch_joint": 2,
    ".*_ankle_roll_joint": 0.5,
    "waist_yaw_joint": 18.7,
    ".*_shoulder_pitch_joint": 10.2,
    ".*_shoulder_roll_joint": 10.2,
    ".*_shoulder_yaw_joint": 4.0,
    ".*_elbow_joint": 4.0,
    ".*_wrist_roll_joint": 0.67,
    ".*_wrist_yaw_joint": 0.67,
    "Neck_yaw_joint": 0.67,
    "Neck_pitch_joint": 0.67,
}
#

scale_dict = {
    ".*": 0.5,
}

joint_pos_config = {
    "Left_shoulder_roll_joint": -1.57,
    "Left_elbow_joint": 1.57,
    "Left_wrist_yaw_joint": 0.3,
    "Right_shoulder_roll_joint": -1.57,
    "Right_elbow_joint": 1.57,
    "Right_wrist_yaw_joint": 0.3,
}


def get_param(joint_name, param_dict):
    for pattern, value in param_dict.items():
        if pattern == ".*":
            return value
        if pattern.startswith(".*"):
            suffix = pattern[3:]
            if joint_name.lower().endswith(suffix.lower()):
                return value
        else:
            if joint_name == pattern:
                return value
    if ".*" in param_dict: return param_dict[".*"]
    raise ValueError(f"No value found for joint: {joint_name}")


def get_joint_default_pos(joint_name):
    return joint_pos_config.get(joint_name, 0.0)


isaaclab_to_mujoco_reindex = [isaaclab_joint_names.index(name) for name in mujoco_joint_names]
mujoco_to_isaaclab_reindex = [mujoco_joint_names.index(name) for name in isaaclab_joint_names]

kps = np.array([get_param(name, stiffness_dict) for name in mujoco_joint_names], dtype=np.float32)
kds = np.array([get_param(name, damping_dict) for name in mujoco_joint_names], dtype=np.float32)
action_scale = np.array([get_param(name, scale_dict) for name in mujoco_joint_names], dtype=np.float32)
default_angles = np.array([get_joint_default_pos(name) for name in mujoco_joint_names], dtype=np.float32)


# Fix bug for mujoco_viewer:
def __fix__add_marker_to_scene(self, marker):
    if self.scn.ngeom >= self.scn.maxgeom:
        raise RuntimeError("Ran out of geoms. maxgeom: %d" % self.scn.maxgeom)

    g = self.scn.geoms[self.scn.ngeom]

    mujoco.mjv_initGeom(
        g,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.ones(4)
    )
    # default values.
    # g.dataid = -1
    g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
    g.objid = -1
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    # g.texid = -1
    # g.texuniform = 0
    # g.texrepeat[0] = 1
    # g.texrepeat[1] = 1
    g.emission = 0
    g.specular = 0.5
    g.shininess = 0.5
    g.reflectance = 0
    # g.type = mujoco.mjtGeom.mjGEOM_BOX
    # g.size[:] = np.ones(3) * 0.1
    # g.mat[:] = np.eye(3)
    # g.rgba[:] = np.ones(4)

    for key, value in marker.items():
        if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
            setattr(g, key, value)
        elif isinstance(value, (tuple, list, np.ndarray)):
            attr = getattr(g, key)
            attr[:] = np.asarray(value).reshape(attr.shape)
        elif isinstance(value, str):
            assert key == "label", "Only label is a string in mjtGeom."
            if value is None:
                g.label[0] = 0
            else:
                g.label = value
        elif hasattr(g, key):
            raise ValueError("mjtGeom has attr {} but type {} is invalid".format(key, type(value)))
        else:
            raise ValueError("mjtGeom doesn't have field %s" % key)

    self.scn.ngeom += 1

    return


mujoco_viewer.MujocoViewer._add_marker_to_scene = __fix__add_marker_to_scene


def update_joint_visualization(viewer, motion_loader, t):
    """Update joint visualization spheres using body positions"""
    if t < 0 or t >= motion_loader.T:
        return

    # Get body positions from motion data
    body_pos = motion_loader.body_pos[t]  # [N, 3]
    # print(body_pos,"bbbbb")
    # Update spheres for each body (excluding the first one which is pelvis)
    if not np.isfinite(body_pos).all():
        print(f"Warning: Invalid tracking data (NaN/Inf) detected at frame {t}. Skipping visualization.")
        return
    for i in range(0, len(body_pos)):
        viewer.add_marker(
            pos=body_pos[i],
            size=[0.02, 0.02, 0.02],
            rgba=[0.8, 0.5, 0.3, 1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            label=""
        )

    # Update anchor body sphere (larger) - use torso_link position
    #print(motion_loader.anchor_body_index, "ppppp", body_pos[motion_loader.anchor_body_index])
    anchor_pos = body_pos[motion_loader.anchor_body_index]  # torso_link position
    viewer.add_marker(
        pos=anchor_pos, size=[0.05, 0.05, 0.05], rgba=[0.8, 0.5, 0.3, 1], type=mujoco.mjtGeom.mjGEOM_SPHERE, label=""
    )


class MotionLoader:
    def __init__(self, motion_file):
        data = np.load(motion_file)
        self.joint_pos = data["joint_pos"]  # [T, 27]
        # self.joint_pos[:,:]=default_angles[mujoco_to_isaaclab_reindex]
        self.joint_vel = data["joint_vel"]  # [T, 27]
        self.body_pos = data["body_pos_w"]  # [T, N, 3]
        self.body_ori = data["body_quat_w"]  # [T, N, 4]
        self.body_vel = data["body_lin_vel_w"]
        self.body_ang_vel = data["body_ang_vel_w"]
        self.fps = data["fps"]
        self.T = self.joint_pos.shape[0]
        self.anchor_body_name = "waist_yaw_Link"  # MuJoCo body name (matches XML)
        self.anchor_body_index = anchor_body.value  # 3 (Isaac Lab BFS index)


def get_command(motion_loader, t):
    if t < 0: return np.zeros(33, dtype=np.float32)
    step_idx = min(t, motion_loader.T - 1)
    joint_pos_future = motion_loader.joint_pos[step_idx]
    ref_anchor_pos = motion_loader.body_pos[step_idx][motion_loader.anchor_body_index]
    ref_anchor_lin_vel = motion_loader.body_vel[step_idx][motion_loader.anchor_body_index]
    ref_anchor_ang_vel = motion_loader.body_ang_vel[step_idx][motion_loader.anchor_body_index]
    ref_anchor_quat = motion_loader.body_ori[step_idx][motion_loader.anchor_body_index]
    ref_root_lin_vel_base = quat_apply_inverse(ref_anchor_quat, ref_anchor_lin_vel)
    ref_root_ang_vel_base = quat_apply_inverse(ref_anchor_quat, ref_anchor_ang_vel)
    rpy = quatToEuler(ref_anchor_quat)
    print(joint_pos_future[[6,11]],joint_pos_future[-2:])
    # joint_pos_future[[6,11]]=0
    # joint_pos_future[-2:]=0
    # joint_pos_future[-4]
    # print(isaaclab_to_mujoco_reindex[:12])
    # joint_pos_future[isaaclab_to_mujoco_reindex[13:]]=0.
    # joint_pos_future[:] += (0.1 - np.random.rand(27) * 0.2)*3
    # ref_anchor_pos[2:3] +=  0.01 - np.random.rand(1) * 0.02
    # rpy[:2] = (0.1 - np.random.rand(2) * 0.2)*3
    # print(ref_anchor_pos[2:3],joint_pos_future,"hhhhhhh")
    cmd = np.concatenate([
        ref_root_lin_vel_base[:2],
        ref_anchor_pos[2:3],
        rpy[:2],
        ref_root_ang_vel_base[2:3],
        joint_pos_future
    ], axis=0)
    return cmd


def motion_anchor_ori_b_future(sim_data, motion_loader, t, ref_to_robot_quat_init):
    robot_quat = sim_data.body(motion_loader.anchor_body_name).xquat.copy()  # waist_yaw_Link orientation
    step_idx = min(t, motion_loader.T - 1)
    ref_quat = motion_loader.body_ori[step_idx][motion_loader.anchor_body_index]
    future_anchor_quat_w = quat_mul_np(ref_to_robot_quat_init, ref_quat)
    _, ori_b = subtract_frame_transforms(None, robot_quat, None, future_anchor_quat_w)
    return quat_to_rot6d(ori_b)


def get_base_ang_vel(sim_data):
    return sim_data.qvel[3:6]


def get_joint_pos_rel(sim_data):
    return sim_data.qpos[7:7 + 27] - default_angles


def get_joint_vel_rel(sim_data):
    return sim_data.qvel[6:6 + 27]


def get_projected_gravity(sim_data):
    quaternion = sim_data.qpos[3:7]
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def compute_observation(sim_data, motion_loader, t, last_actions, ref_to_robot_quat_init):
    obs_dim = 126  # 33 + 6 + 3 + 3 + 27 + 27 + 27
    if t < 0: return np.zeros(obs_dim, dtype=np.float32)

    command = get_command(motion_loader, t)
    motion_anchor_ori = motion_anchor_ori_b_future(sim_data, motion_loader, t, ref_to_robot_quat_init)

    obs = np.zeros(obs_dim, dtype=np.float32)
    obs[0:33] = command
    obs[33:39] = motion_anchor_ori
    # print(command, "commmmmmand",motion_anchor_ori)
    obs[39:42] = get_projected_gravity(sim_data)
    obs[42:45] = get_base_ang_vel(sim_data) * 0.25
    obs[45:72] = get_joint_pos_rel(sim_data)[mujoco_to_isaaclab_reindex]
    obs[72:99] = get_joint_vel_rel(sim_data)[mujoco_to_isaaclab_reindex] * 0.05
    obs[99:126] = last_actions
    return obs


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy Khan MuJoCo simulation")
    # parser.add_argument("--motion_path", type=str, default="", help="Path to motion file")
    # parser.add_argument("--policy_path", type=str, default="", help="Path to policy file")
    args = parser.parse_args()

   # xml_path = "Khan_mini_simplified.xml"
    simulation_dt = 0.002
    control_decimation = 10
    num_actions = 27



    # Use specified paths or defaults
    # policy_path = args.policy_path if args.policy_path else "policy.onnx"
    # motion_path = args.motion_path if args.motion_path else "motion.npz"
    policy_path = "/home/irev/docker_data/TextOp/TextOpTracker/logs/rsl_rl/ExampleRun/2026-04-28_new/latest15k.onnx" #2026-04-21_03-34-47_base/latest.onnx"
    # 修改：指定文件夹路径
    policy_path ="/home/irev/docker_data/test_motion/latest35knew.onnx"
    motion_path = "/home/irev/docker_data/test_motion/jaka_mini_lafan/dance1_subject2/motion.npz" #jaka_mini_lafan/fallAndGetUp1_subject4/motion.npz"#"/home/irev/docker_data/TextOp/TextOpTracker/artifacts/jaka_mini_test2/Uaenet_0259/motion.npz" #"/home/irev/docker_data/TextOp/TextOpTracker/artifacts/jaka_mini_test2/homejrhanprojectsPBHC-InternalPBHC-Motiong1robotlafandance2_subject3_0_6771_cont_mask_inter05_S1643-/motion.npz"
    # motion_path="/home/irev/docker_data/TextOp/TextOpTracker/artifacts/test_mini/00_self_pico_record_185/motion.npz"
    # motion_path = "/home/irev/docker_data/test_motion/jaka/motion.npz"#
    # motion_path ="/home/irev//docker_data/TextOp/TextOpTracker/artifacts/jaka_mini_test2/Uaenet_0664/motion.npz"
    xml_path = "./source/textop_tracker/textop_tracker/assets/Khan_mini_simplified/Khan_mini_simplified_new_bigfeet.xml"
    # 0352 0890  Data10k-open_homejrhangmr_dataset_pbhc_contact_maskCMU1313_03_posespkl
    session = ort.InferenceSession(policy_path)
    obs_name = session.get_inputs()[0].name
    motion_loader = MotionLoader(motion_path)
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    viewer = mujoco_viewer.MujocoViewer(m, d)
    m.opt.timestep = simulation_dt

    if True:
        viewer.cam.lookat = d.xpos[1]
        viewer.cam.distance = 5
        viewer.cam.elevation = -30  # 正面视角，轻微向下看
   # print(d.qpos.shape)
    inner_counter =0
    # init=default_angles[mujoco_to_isaaclab_reindex].copy()
    # # init[-4:]=1.5
    # init[-2:]=0.
    #
    # print(init, "iiiiiiiiiiiiii",)
    # print(motion_loader.joint_pos[inner_counter]-init,"wwwwwwwwwwwwwwwwwww")
    d.qpos[7:7 + 27] = motion_loader.joint_pos[inner_counter][isaaclab_to_mujoco_reindex]
    print(motion_loader.joint_pos[inner_counter][isaaclab_to_mujoco_reindex],"wwwww")
    d.qpos[:3] = motion_loader.body_pos[inner_counter][0]
    d.qpos[3:7] = motion_loader.body_ori[inner_counter][0]
    print( d.qpos[-13:-10],"wwww")#-1.34431398  1.13296843  0.83139992]
    # d.qpos[-13:-10]=[-1.27290213 ,0.24230489 , 1.24240172]
    # d.qpos[-12]=0.24230489
    print(d.qpos[7:])
    frame_stack = deque(maxlen=5)
    for _ in range(5): frame_stack.append(np.zeros(126, dtype=np.float32))

    # robot_init_quat = yaw_quat(d.qpos[3:7])
    mujoco.mj_step(m, d)  # forward kinematics so body xquat is valid
    robot_init_quat = yaw_quat(d.body(motion_loader.anchor_body_name).xquat.copy())
    ref_init_quat_inv = quat_inv_np(yaw_quat(motion_loader.body_ori[inner_counter][motion_loader.anchor_body_index]))
    ref_to_robot_quat_init = quat_mul_np(robot_init_quat, ref_init_quat_inv)

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()

    counter = 0
    # inner_counter = 0
    start_time = time.time()

    while viewer.is_alive:
        tau = pd_control(target_dof_pos, d.qpos[7:7 + 27], kps, np.zeros(27), d.qvel[6:6 + 27], kds)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

        if counter % control_decimation == 0:
            print(inner_counter)

            obs = compute_observation(d, motion_loader, inner_counter, action, ref_to_robot_quat_init)
            if counter==0:
                for _ in range(5):
                    frame_stack.append(obs.copy())
            frame_stack.append(obs)

            stacked_obs = np.concatenate(list(frame_stack), axis=0)  # [5 * 126]
            reshaped_stack = stacked_obs.reshape(5, 126)

            # Match K1L's feature-major stacking: [5*33, 5*6, 5*3, 5*3, 5*27, 5*27, 5*27]
            obs_command = reshaped_stack[:, :33].reshape(-1)
            obs_anchor_ori = reshaped_stack[:, 33:39].reshape(-1)
            obs_gravity = reshaped_stack[:, 39:42].reshape(-1)
            obs_ang_vel = reshaped_stack[:, 42:45].reshape(-1)
            obs_dof_pos = reshaped_stack[:, 45:72].reshape(-1)
            obs_dof_vel = reshaped_stack[:, 72:99].reshape(-1)
            obs_last_action = reshaped_stack[:, 99:126].reshape(-1)

            obs_concat = np.concatenate([
                obs_command, obs_anchor_ori, obs_gravity, obs_ang_vel,
                obs_dof_pos, obs_dof_vel, obs_last_action
            ], axis=0)


            obs_tensor = obs_concat.reshape(1, -1)
            output = session.run(None, {obs_name: obs_tensor})
            action = output[0].squeeze()
            action=np.clip(action,-5,5)
            # print(action)
            # print(action)
            target_dof_pos = action[isaaclab_to_mujoco_reindex] * action_scale + default_angles
            inner_counter = (inner_counter + 1) % motion_loader.T

        counter += 1
        update_joint_visualization(viewer, motion_loader, inner_counter)
        if True:
            viewer.cam.lookat = d.xpos[1]
            viewer.cam.distance = 5
            viewer.cam.elevation = -30
        viewer.render()
