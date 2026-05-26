"""Jaka robot observation classes for policy inference.

The Jaka policy uses a **5-frame stack with feature-major reordering**:
each raw frame is 126-dim (33+6+3+3+27+27+27), and 5 consecutive frames
are rearranged into [5×33, 5×6, 5×3, 5×3, 5×27, 5×27, 5×27] = 630-dim.

Observation breakdown per frame (126-dim):
  [0:33]   command — ref lin_vel_base[:2], height, rpy[:2], ang_vel_z, ref_joint_pos
  [33:39]  motion_anchor_ori_b — 6D rotation of relative anchor orientation
  [39:42]  projected_gravity
  [42:45]  base_ang_vel * 0.25
  [45:72]  joint_pos_rel  (IsaacLab order)
  [72:99]  joint_vel_rel * 0.05  (IsaacLab order)
  [99:126] last_action
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .base import Observation
from sim2real.utils.math import (
    matrix_from_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse_numpy,
    yaw_quat,
)


def _quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert a single quaternion (w,x,y,z) to Euler angles (roll, pitch, yaw)."""
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    euler = np.zeros(3)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    euler[0] = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        euler[1] = np.copysign(np.pi / 2, sinp)
    else:
        euler[1] = np.arcsin(sinp)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    euler[2] = np.arctan2(siny_cosp, cosy_cosp)
    return euler


def _quat_to_rot6d(q: np.ndarray) -> np.ndarray:
    """Convert a single quaternion (w,x,y,z) to 6D rotation representation
    (first two columns of rotation matrix, row-major)."""
    r, i, j, k = q[0], q[1], q[2], q[3]
    two_s = 2.0 / (r * r + i * i + j * j + k * k)
    ii = i * i; jj = j * j; kk = k * k
    ij = i * j; kr = k * r; ik = i * k
    jr = j * r; jk = j * k; ir = i * r
    return np.array([
        1 - two_s * (jj + kk),  # R00
        two_s * (ij - kr),      # R01
        two_s * (ij + kr),      # R10
        1 - two_s * (ii + kk),  # R11
        two_s * (ik - jr),      # R20
        two_s * (jk + ir),      # R21
    ], dtype=np.float32)


def _quat_apply_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Apply inverse rotation of a single quaternion (w,x,y,z) to a vector."""
    xyz = quat[1:]
    w = quat[0]
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)


def _subtract_frame_transforms_q(q01: np.ndarray, q02: np.ndarray) -> np.ndarray:
    """Compute relative quaternion: q12 = q01^-1 * q02."""
    conj = np.array([q01[0], -q01[1], -q01[2], -q01[3]])
    norm_sq = max(np.sum(q01 ** 2), 1e-9)
    q10 = conj / norm_sq
    # quat_mul for single quats
    w1, x1, y1, z1 = q10
    w2, x2, y2, z2 = q02
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def _quat_mul_single(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two single quaternions (w,x,y,z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def _yaw_quat_single(q: np.ndarray) -> np.ndarray:
    """Extract yaw quaternion from a single quaternion (w,x,y,z)."""
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def _quat_inv_single(q: np.ndarray) -> np.ndarray:
    """Inverse of a single quaternion (w,x,y,z)."""
    conj = np.array([q[0], -q[1], -q[2], -q[3]])
    norm_sq = max(np.sum(q ** 2), 1e-9)
    return conj / norm_sq


# ───────────────────────── Observation classes ────────────────────────── #


class jaka_frame_stack(Observation):
    """5-frame history stack with feature-major reordering for the Jaka policy.

    This observation class internally computes all Jaka sub-observations
    (command, anchor_ori, gravity, ang_vel, dof_pos, dof_vel, last_action)
    and maintains a rolling deque of frames.

    On ``compute()``, the 5 × 126-dim frames are rearranged into
    feature-major order: [5×33, 5×6, 5×3, 5×3, 5×27, 5×27, 5×27] = 630.
    """

    _FRAME_DIM = 126
    _STACK_SIZE = 5
    # Slice boundaries inside each 126-dim frame
    _SLICES = [
        (0, 33),    # command
        (33, 39),   # anchor_ori
        (39, 42),   # gravity
        (42, 45),   # ang_vel
        (45, 72),   # dof_pos
        (72, 99),   # dof_vel
        (99, 126),  # last_action
    ]

    def __init__(
        self,
        anchor_body_index: int = 3,
        ang_vel_scale: float = 0.25,
        joint_vel_scale: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.anchor_body_index = anchor_body_index
        self.ang_vel_scale = ang_vel_scale
        self.joint_vel_scale = joint_vel_scale

        # Joint order mapping: the NPZ / policy uses IsaacLab order,
        # but state_processor stores joints in MuJoCo (simulation) order.
        # We build a reindex: mujoco_joint[i] → isaaclab_joint[j].
        motion_cfg = self.state_processor.motion_config
        self.isaaclab_joint_names: List[str] = list(motion_cfg.get("npz_joint_names", []))
        sim_joint_names = list(self.state_processor.joint_names)

        # mujoco_to_isaaclab: for each IsaacLab joint, find its index in sim_joint_names
        self.mujoco_to_isaaclab_reindex = [
            sim_joint_names.index(name) for name in self.isaaclab_joint_names
        ]
        self.n_joints = len(self.isaaclab_joint_names)

        # Default joint positions in IsaacLab order
        self.default_angles_isaaclab = np.zeros(self.n_joints, dtype=np.float32)
        default_joint_pos_dict = self.env.policy_config.get("default_joint_pos", {})
        for jname, jval in default_joint_pos_dict.items():
            if jname in self.isaaclab_joint_names:
                idx = self.isaaclab_joint_names.index(jname)
                self.default_angles_isaaclab[idx] = float(jval)

        # ref_to_robot_quat_init — computed on reset
        self.ref_to_robot_quat_init = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Frame buffer
        self._frame_buffer: deque = deque(maxlen=self._STACK_SIZE)
        self._is_first_frame = True

    def reset(self):
        self._frame_buffer.clear()
        for _ in range(self._STACK_SIZE):
            self._frame_buffer.append(np.zeros(self._FRAME_DIM, dtype=np.float32))
        self._is_first_frame = True

        # Compute ref_to_robot_quat_init like G1's ref_root_ori_future_b.reset()
        # but using anchor body instead of root
        motion_data = self.state_processor.motion_data
        if motion_data is not None:
            # Get motion anchor quat at t=0
            # motion_data.body_quat_w shape: [N, S, B, 4], N=1, S=num_future_steps
            # At reset, motion_t=0, so we take the first step (index 0)
            ref_anchor_quat = motion_data.body_quat_w[0, 0, self.anchor_body_index]
            ref_init_yaw = _yaw_quat_single(ref_anchor_quat)
            ref_init_yaw_inv = _quat_inv_single(ref_init_yaw)

            # Robot's current anchor body orientation (waist_yaw_Link)
            # waist_yaw_Link is rotated by waist_yaw_joint (index 12 in MuJoCo order) relative to root
            robot_quat = self.state_processor.root_quat_w.copy()
            waist_yaw_angle = self.state_processor.joint_pos[12]
            half_angle = waist_yaw_angle * 0.5
            rz = np.array([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)], dtype=np.float32)
            robot_anchor_quat = _quat_mul_single(robot_quat, rz)
            robot_init_yaw = _yaw_quat_single(robot_anchor_quat)

            self.ref_to_robot_quat_init = _quat_mul_single(robot_init_yaw, ref_init_yaw_inv)

    def update(self, data: Dict[str, Any]) -> None:
        obs = self._compute_single_frame(data)
        if self._is_first_frame:
            # Fill entire buffer with first observation (like deploy script)
            for _ in range(self._STACK_SIZE):
                self._frame_buffer.append(obs.copy())
            self._is_first_frame = False
        else:
            self._frame_buffer.append(obs.copy())

    def _compute_single_frame(self, data: Dict[str, Any]) -> np.ndarray:
        """Compute a single 126-dim observation frame."""
        sp = self.state_processor
        motion_data = sp.motion_data

        obs = np.zeros(self._FRAME_DIM, dtype=np.float32)

        if motion_data is None:
            return obs

        # Current motion step index (for single future_step=[0], index is 0)
        step_idx = 0

        # ── Command (33) ──
        anchor_idx = self.anchor_body_index
        ref_anchor_pos = motion_data.body_pos_w[0, step_idx, anchor_idx]     # [3]
        ref_anchor_quat = motion_data.body_quat_w[0, step_idx, anchor_idx]   # [4]
        ref_anchor_lin_vel = motion_data.body_lin_vel_w[0, step_idx, anchor_idx]  # [3]
        ref_anchor_ang_vel = motion_data.body_ang_vel_w[0, step_idx, anchor_idx]  # [3]
        ref_joint_pos = motion_data.joint_pos[0, step_idx]                   # [J]

        ref_root_lin_vel_base = _quat_apply_inverse(ref_anchor_quat, ref_anchor_lin_vel)
        ref_root_ang_vel_base = _quat_apply_inverse(ref_anchor_quat, ref_anchor_ang_vel)
        rpy = _quat_to_euler(ref_anchor_quat)

        obs[0:2] = ref_root_lin_vel_base[:2]
        obs[2:3] = ref_anchor_pos[2:3]
        obs[3:5] = rpy[:2]
        obs[5:6] = ref_root_ang_vel_base[2:3]
        obs[6:33] = ref_joint_pos[:self.n_joints]

        # ── Anchor Orientation (6) ──
        # motion_anchor_ori_b_future: relative anchor orientation as rot6d
        robot_quat = sp.root_quat_w.copy()
        waist_yaw_angle = sp.joint_pos[12]
        half_angle = waist_yaw_angle * 0.5
        rz = np.array([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)], dtype=np.float32)
        robot_anchor_quat = _quat_mul_single(robot_quat, rz)

        future_anchor_quat_w = _quat_mul_single(self.ref_to_robot_quat_init, ref_anchor_quat)
        ori_b = _subtract_frame_transforms_q(robot_anchor_quat, future_anchor_quat_w)
        obs[33:39] = _quat_to_rot6d(ori_b)

        # ── Projected Gravity (3) ──
        quat = sp.root_quat_w.copy()
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        obs[39] = 2 * (-qz * qx + qw * qy)
        obs[40] = -2 * (qz * qy + qw * qx)
        obs[41] = 1 - 2 * (qw * qw + qz * qz)

        # ── Base Angular Velocity (3) ──
        obs[42:45] = sp.root_ang_vel_b * self.ang_vel_scale

        # ── Joint Pos Relative (27, IsaacLab order) ──
        joint_pos_mujoco = sp.joint_pos  # MuJoCo order
        joint_pos_isaaclab = joint_pos_mujoco[self.mujoco_to_isaaclab_reindex]
        obs[45:72] = joint_pos_isaaclab - self.default_angles_isaaclab

        # ── Joint Vel (27, IsaacLab order) ──
        joint_vel_mujoco = sp.joint_vel
        joint_vel_isaaclab = joint_vel_mujoco[self.mujoco_to_isaaclab_reindex]
        obs[72:99] = joint_vel_isaaclab * self.joint_vel_scale

        # ── Last Action (27) ──
        last_action = data.get("action", np.zeros(self.n_joints, dtype=np.float32))
        obs[99:126] = last_action[:self.n_joints]

        return obs

    def compute(self) -> np.ndarray:
        """Stack 5 frames in feature-major order → 630-dim output."""
        stacked = np.array(list(self._frame_buffer), dtype=np.float32)  # [5, 126]
        parts = []
        for start, end in self._SLICES:
            parts.append(stacked[:, start:end].reshape(-1))
        return np.concatenate(parts, axis=0)