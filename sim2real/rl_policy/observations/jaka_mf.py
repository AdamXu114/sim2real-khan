"""Jaka robot observation class for the MF (multi-frame future) policy.

The MF policy uses a **5-frame history stack with feature-major reordering**
plus **5-future-step command and anchor-orientation** computed once per step.

Per-step layout (620-dim total):
  [0:155]   command — root_pos_diff_b (5×3) + root_z_mf (5) + ref_joint_pos (5×27)
  [155:185] anchor_ori — rot6d (5×6)
  [185:200] gravity — 5-frame stack (5×3)
  [200:215] ang_vel — 5-frame stack (5×3)
  [215:350] dof_pos — 5-frame stack (5×27)
  [350:485] dof_vel — 5-frame stack (5×27)
  [485:620] last_action — 5-frame stack (5×27)

History frame (87-dim):
  [0:3]    projected_gravity (using waist_yaw_Link body quat)
  [3:6]    base_ang_vel * 0.25
  [6:33]   joint_pos_rel (IsaacLab order)
  [33:60]  joint_vel_rel * 0.05 (IsaacLab order)
  [60:87]  last_action
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List

import numpy as np

from .base import Observation


# ──────────────────────── Quaternion helpers ─────────────────────────────── #

def _quat_to_rot6d(q: np.ndarray) -> np.ndarray:
    r, i, j, k = q[0], q[1], q[2], q[3]
    two_s = 2.0 / (r * r + i * i + j * j + k * k)
    ii = i * i; jj = j * j; kk = k * k
    ij = i * j; kr = k * r; ik = i * k
    jr = j * r; jk = j * k; ir = i * r
    return np.array([
        1 - two_s * (jj + kk),
        two_s * (ij - kr),
        two_s * (ij + kr),
        1 - two_s * (ii + kk),
        two_s * (ik - jr),
        two_s * (jk + ir),
    ], dtype=np.float32)


def _quat_apply_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    xyz = quat[1:]
    w = quat[0]
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)


def _quat_mul_single(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def _quat_inv_single(q: np.ndarray) -> np.ndarray:
    conj = np.array([q[0], -q[1], -q[2], -q[3]])
    norm_sq = max(np.sum(q ** 2), 1e-9)
    return conj / norm_sq


def _yaw_quat_single(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def _subtract_frame_transforms_q(q01: np.ndarray, q02: np.ndarray) -> np.ndarray:
    """Compute relative quaternion: q01^-1 * q02."""
    q10 = _quat_inv_single(q01)
    return _quat_mul_single(q10, q02)


# ─────────────────────────── Observation class ──────────────────────────── #

class jaka_frame_stack_mf(Observation):
    """5-frame history stack + 5-future-step command/anchor_ori for the MF Jaka policy.

    On ``compute()`` returns a 620-dim vector:
      command(155) + anchor_ori(30) + stacked_history(5×87, feature-major)
    """

    _FRAME_DIM = 87
    _STACK_SIZE = 5
    _NUM_FUTURE_STEPS = 5
    _COMMAND_DIM = 155   # 3*5 + 1*5 + 27*5
    _ANCHOR_ORI_DIM = 30  # 6*5

    # Slice boundaries inside each 87-dim history frame
    _SLICES = [
        (0, 3),     # gravity
        (3, 6),     # ang_vel
        (6, 33),    # dof_pos
        (33, 60),   # dof_vel
        (60, 87),   # last_action
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

        motion_cfg = self.state_processor.motion_config
        self.isaaclab_joint_names: List[str] = list(motion_cfg.get("npz_joint_names", []))
        sim_joint_names = list(self.state_processor.joint_names)

        self.mujoco_to_isaaclab_reindex = [
            sim_joint_names.index(name) for name in self.isaaclab_joint_names
        ]
        self.n_joints = len(self.isaaclab_joint_names)

        self.default_angles_isaaclab = np.zeros(self.n_joints, dtype=np.float32)
        default_joint_pos_dict = self.env.policy_config.get("default_joint_pos", {})
        for jname, jval in default_joint_pos_dict.items():
            if jname in self.isaaclab_joint_names:
                idx = self.isaaclab_joint_names.index(jname)
                self.default_angles_isaaclab[idx] = float(jval)

        self.ref_to_robot_quat_init = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self._frame_buffer: deque = deque(maxlen=self._STACK_SIZE)
        self._is_first_frame = True

    # ── Lifecycle ──────────────────────────────────────────────────────── #

    def reset(self):
        self._frame_buffer.clear()
        for _ in range(self._STACK_SIZE):
            self._frame_buffer.append(np.zeros(self._FRAME_DIM, dtype=np.float32))
        self._is_first_frame = True

        motion_data = self.state_processor.motion_data
        if motion_data is not None:
            ref_anchor_quat = motion_data.body_quat_w[0, 0, self.anchor_body_index]
            ref_init_yaw = _yaw_quat_single(ref_anchor_quat)
            ref_init_yaw_inv = _quat_inv_single(ref_init_yaw)

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
            for _ in range(self._STACK_SIZE):
                self._frame_buffer.append(obs.copy())
            self._is_first_frame = False
        else:
            self._frame_buffer.append(obs.copy())

    # ── Single history frame (87-dim) ──────────────────────────────────── #

    def _compute_single_frame(self, data: Dict[str, Any]) -> np.ndarray:
        sp = self.state_processor
        motion_data = sp.motion_data

        obs = np.zeros(self._FRAME_DIM, dtype=np.float32)
        if motion_data is None:
            return obs

        # Projected gravity using waist_yaw_Link body quat
        robot_quat = sp.root_quat_w.copy()
        waist_yaw_angle = sp.joint_pos[12]
        half_angle = waist_yaw_angle * 0.5
        rz = np.array([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)], dtype=np.float32)
        anchor_quat = _quat_mul_single(robot_quat, rz)
        qw, qx, qy, qz = anchor_quat
        obs[0] = 2 * (-qz * qx + qw * qy)
        obs[1] = -2 * (qz * qy + qw * qx)
        obs[2] = 1 - 2 * (qw * qw + qz * qz)

        # Base angular velocity (gyro)
        obs[3:6] = sp.root_ang_vel_b * self.ang_vel_scale

        # Joint positions relative to default (IsaacLab order)
        joint_pos_mujoco = sp.joint_pos
        joint_pos_isaaclab = joint_pos_mujoco[self.mujoco_to_isaaclab_reindex]
        obs[6:33] = joint_pos_isaaclab - self.default_angles_isaaclab

        # Joint velocities (IsaacLab order)
        joint_vel_mujoco = sp.joint_vel
        joint_vel_isaaclab = joint_vel_mujoco[self.mujoco_to_isaaclab_reindex]
        obs[33:60] = joint_vel_isaaclab * self.joint_vel_scale

        # Last action
        last_action = data.get("action", np.zeros(self.n_joints, dtype=np.float32))
        obs[60:87] = last_action[:self.n_joints]

        return obs

    # ── Command (155-dim) ──────────────────────────────────────────────── #

    def _compute_command(self) -> np.ndarray:
        """Build 155-dim command from 5 future motion steps.

        Structure: root_pos_diff_b (5×3) + root_z_mf (5×1) + ref_joint_pos (5×27)
        Neck yaw/pitch and wrist yaw joints are zeroed in the reference.
        """
        motion_data = self.state_processor.motion_data
        if motion_data is None:
            return np.zeros(self._COMMAND_DIM, dtype=np.float32)

        anchor_idx = self.anchor_body_index
        ref_anchor_pos_cur = motion_data.body_pos_w[0, 0, anchor_idx]
        ref_anchor_quat_cur = motion_data.body_quat_w[0, 0, anchor_idx]

        root_pos_diff_b = []
        root_z_mf = []
        motion_joint_pos = []

        for i in range(self._NUM_FUTURE_STEPS):
            ref_pos = motion_data.body_pos_w[0, i, anchor_idx]
            diff_w = ref_pos - ref_anchor_pos_cur
            diff_b = _quat_apply_inverse(ref_anchor_quat_cur, diff_w)
            root_pos_diff_b.append(diff_b)

            root_z_mf.append(ref_pos[2:3])

            joint_pos_step = motion_data.joint_pos[0, i].copy()
            # Zero out neck and wrist yaw joints (IsaacLab indices 6, 11, 25, 26)
            joint_pos_step[6] = 0
            joint_pos_step[11] = 0
            joint_pos_step[-2:] = 0
            motion_joint_pos.append(joint_pos_step)

        return np.concatenate([
            np.concatenate(root_pos_diff_b, axis=0),
            np.concatenate(root_z_mf, axis=0),
            np.concatenate(motion_joint_pos, axis=0),
        ], axis=0).astype(np.float32)

    # ── Anchor orientation (30-dim) ────────────────────────────────────── #

    def _compute_anchor_ori(self) -> np.ndarray:
        """Build 30-dim anchor orientation from 5 future motion steps (rot6d × 5)."""
        motion_data = self.state_processor.motion_data
        if motion_data is None:
            return np.zeros(self._ANCHOR_ORI_DIM, dtype=np.float32)

        sp = self.state_processor
        robot_quat = sp.root_quat_w.copy()
        waist_yaw_angle = sp.joint_pos[12]
        half_angle = waist_yaw_angle * 0.5
        rz = np.array([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)], dtype=np.float32)
        robot_anchor_quat = _quat_mul_single(robot_quat, rz)

        anchor_idx = self.anchor_body_index
        ori_b_flat = []
        for i in range(self._NUM_FUTURE_STEPS):
            ref_quat = motion_data.body_quat_w[0, i, anchor_idx]
            future_anchor_quat_w = _quat_mul_single(self.ref_to_robot_quat_init, ref_quat)
            ori_b = _subtract_frame_transforms_q(robot_anchor_quat, future_anchor_quat_w)
            ori_b_flat.append(_quat_to_rot6d(ori_b))

        return np.concatenate(ori_b_flat, axis=0).astype(np.float32)

    # ── Full observation ───────────────────────────────────────────────── #

    def compute(self) -> np.ndarray:
        command = self._compute_command()       # 155
        anchor_ori = self._compute_anchor_ori()  # 30

        stacked = np.array(list(self._frame_buffer), dtype=np.float32)  # [5, 87]
        parts = [command, anchor_ori]
        for start, end in self._SLICES:
            parts.append(stacked[:, start:end].reshape(-1))
        return np.concatenate(parts, axis=0)  # 620
