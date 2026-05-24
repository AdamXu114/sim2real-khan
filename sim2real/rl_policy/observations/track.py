from .base import Observation
from .common import sort_names_by_preferred_order

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
from sim2real.rl_policy.utils.motion import MotionData
from sim2real.utils.math import (
    matrix_from_quat,
    projected_yaw_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse_numpy,
)
from sim2real.utils.strings import resolve_matching_names


class _motion_obs(Observation):
    def __init__(
        self,
        future_steps: Optional[Union[Sequence[int], int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        selected_future_steps = future_steps
        motion_cfg = self.state_processor.motion_config
        if not motion_cfg:
            raise ValueError("policy_config.motion is required for motion observations")

        motion_future_steps = motion_cfg.get("future_steps")
        joint_names = motion_cfg.get("joint_names")
        body_names = motion_cfg.get("body_names")
        if motion_future_steps is None or joint_names is None or body_names is None:
            raise ValueError("policy_config.motion must define future_steps, joint_names, and body_names")

        self.motion_future_steps = np.asarray(motion_future_steps, dtype=int)
        if self.motion_future_steps.ndim != 1:
            raise ValueError(f"motion.future_steps must be 1D, got shape={self.motion_future_steps.shape}")
        self.available_future_steps = [int(step) for step in self.motion_future_steps.tolist()]
        if 0 not in self.available_future_steps:
            raise ValueError("motion.future_steps must include 0 to compute current observation")
        self.obs_current_step_index = int(self.available_future_steps.index(0))
        self.future_step_indices, self.future_steps = self._resolve_future_steps(selected_future_steps)
        self.selected_future_steps = self.future_steps
        self.n_future_steps = len(self.future_steps)
        self.n_selected_future_steps = self.n_future_steps
        self.joint_names = sort_names_by_preferred_order(
            joint_names,
            self.env.joint_names_simulation,
        )
        self.body_names = sort_names_by_preferred_order(
            body_names,
            self.env.body_names_simulation,
        )
        self.root_body_name = str(motion_cfg.get("root_body_name", "pelvis"))
        self.anchor_body_name = str(motion_cfg.get("anchor_body_name", "torso_link"))
        self.n_bodies = len(self.body_names)
        self._cached_motion_layout: Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]] = None

    def _resolve_future_steps(
        self,
        future_steps: Optional[Union[Sequence[int], int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if future_steps is None:
            requested_future_steps = self.available_future_steps
        elif isinstance(future_steps, (int, np.integer)):
            requested_future_steps = [int(future_steps)]
        else:
            requested_future_steps = [int(step) for step in future_steps]

        if not requested_future_steps:
            raise ValueError("future_steps must select at least one step")

        future_step_indices = []
        for step in requested_future_steps:
            if step not in self.available_future_steps:
                raise ValueError(
                    f"future step {step} not in motion.future_steps={self.available_future_steps}"
                )
            future_step_indices.append(self.available_future_steps.index(step))

        return (
            np.asarray(future_step_indices, dtype=int),
            np.asarray(requested_future_steps, dtype=int),
        )

    def _select(self, x: np.ndarray) -> np.ndarray:
        return np.take(x, self.future_step_indices, axis=1)
    
    def reset(self):
        # state processor reset handles motion timing; we only refresh cache
        self._assign_motion_views()
    
    def update(self, data: Dict[str, Any]) -> None:
        self._assign_motion_views()

    def _refresh_motion_indices(self) -> None:
        joint_names = tuple(self.state_processor.motion_joint_names)
        body_names = tuple(self.state_processor.motion_body_names)
        layout = (joint_names, body_names)
        if self._cached_motion_layout == layout:
            return
        if not joint_names or not body_names:
            raise ValueError("Motion source names are not ready")

        self._joint_indices = [joint_names.index(name) for name in self.joint_names]
        self._body_indices = [body_names.index(name) for name in self.body_names]
        self._root_body_idx = body_names.index(self.root_body_name)
        self._anchor_body_idx = body_names.index(self.anchor_body_name)
        self._cached_motion_layout = layout

    def _assign_motion_views(self):
        motion_data: MotionData = self.state_processor.motion_data
        self._refresh_motion_indices()

        self.ref_joint_pos_future = motion_data.joint_pos[:, :, self._joint_indices]
        self.ref_body_pos_future_w = motion_data.body_pos_w[:, :, self._body_indices]
        self.ref_body_quat_future_w = motion_data.body_quat_w[:, :, self._body_indices]

        self.ref_root_pos_future_w = motion_data.body_pos_w[:, :, self._root_body_idx, :]
        self.ref_root_quat_future_w = motion_data.body_quat_w[:, :, self._root_body_idx, :]

        self.ref_root_pos_w = motion_data.body_pos_w[
            :, self.obs_current_step_index, self._root_body_idx, :
        ]
        self.ref_root_quat_w = motion_data.body_quat_w[
            :, self.obs_current_step_index, self._root_body_idx, :
        ]

        self.ref_anchor_pos_w = motion_data.body_pos_w[
            :, self.obs_current_step_index, self._anchor_body_idx, :
        ]
        self.ref_anchor_quat_w = motion_data.body_quat_w[
            :, self.obs_current_step_index, self._anchor_body_idx, :
        ]


class _motion_body_obs(_motion_obs):
    def __init__(
        self,
        body_names: Optional[Union[Sequence[str], str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if body_names is None:
            body_names = self.body_names

        body_indices, matched_body_names = resolve_matching_names(
            body_names,
            self.body_names,
        )
        if not matched_body_names:
            raise ValueError("No tracking body matched for observation.")

        self.body_indices_tracking = np.asarray(body_indices, dtype=int)
        self.selected_body_names = matched_body_names
        self.n_selected_bodies = len(self.body_indices_tracking)

    def _select(self, x: np.ndarray) -> np.ndarray:
        x = super()._select(x)
        return np.take(x, self.body_indices_tracking, axis=2)


class _motion_joint_obs(_motion_obs):
    def __init__(
        self,
        joint_names: Optional[Union[Sequence[str], str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if joint_names is None:
            joint_names = self.joint_names

        joint_indices, matched_joint_names = resolve_matching_names(
            joint_names,
            self.joint_names,
        )
        if not matched_joint_names:
            raise ValueError("No tracking joint matched for observation.")

        self.joint_indices_tracking = np.asarray(joint_indices, dtype=int)
        self.selected_joint_names = matched_joint_names
        self.n_selected_joints = len(self.joint_indices_tracking)

    def _select(self, x: np.ndarray) -> np.ndarray:
        x = super()._select(x)
        return np.take(x, self.joint_indices_tracking, axis=2)

class ref_motion_phase(_motion_obs):
    def __init__(self, motion_duration_second: float, **kwargs):
        super().__init__(**kwargs)
        self.motion_steps = int(motion_duration_second * 50)
    
    def compute(self) -> np.ndarray:
        t = self.state_processor.motion_t
        ref_motion_phase = (t % self.motion_steps) / self.motion_steps
        out = ref_motion_phase.reshape(-1)
        # print(f"[obs] ref_motion_phase shape={out.shape}")
        return out


class command_ref_motion(_motion_obs):
    """Legacy deploy command block with optional history stacking.

    Per-frame feature layout matches deploy/deploy_mujoco_history_jakamini.py:
      - ref_root_lin_vel_base[:2]
      - ref_anchor_pos[2:3]
      - rpy[:2]
      - ref_root_ang_vel_base[2:3]
      - joint_pos_future
    Total per frame: 33.
    """

    def __init__(
        self,
        history_steps: Optional[Union[Sequence[int], int]] = None,
        lin_vel_scale: float = 1.0,
        height_scale: float = 1.0,
        rpy_scale: float = 1.0,
        ang_vel_scale: float = 1.0,
        joint_pos_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if history_steps is None:
            history_steps = [0]
        elif isinstance(history_steps, (int, np.integer)):
            history_steps = [int(history_steps)]
        else:
            history_steps = [int(step) for step in history_steps]

        self.history_steps = np.asarray(history_steps, dtype=int)
        if self.history_steps.ndim != 1:
            raise ValueError(f"history_steps must be 1D, got shape={self.history_steps.shape}")
        if self.history_steps.size == 0:
            raise ValueError("history_steps must not be empty")

        self.lin_vel_scale = float(lin_vel_scale)
        self.height_scale = float(height_scale)
        self.rpy_scale = float(rpy_scale)
        self.ang_vel_scale = float(ang_vel_scale)
        self.joint_pos_scale = float(joint_pos_scale)

        self._history_buffer = np.zeros((int(self.history_steps.max()) + 1, 33), dtype=np.float32)

    @staticmethod
    def _quat_to_rpy_xy(quat_wxyz: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        pitch = np.where(
            np.abs(sinp) >= 1.0,
            np.copysign(np.pi / 2.0, sinp),
            np.arcsin(sinp),
        )
        return np.stack([roll, pitch], axis=-1)

    def reset(self):
        self._history_buffer[:] = 0.0
        self.update({})

    def update(self, data: Dict[str, Any]) -> None:
        super().update(data)
        motion_data: MotionData = self.state_processor.motion_data
        step = self.obs_current_step_index

        ref_anchor_quat_curr = motion_data.body_quat_w[:, step, self._anchor_body_idx, :]
        ref_anchor_lin_vel_w = motion_data.body_lin_vel_w[:, step, self._anchor_body_idx, :]
        ref_anchor_ang_vel_w = motion_data.body_ang_vel_w[:, step, self._anchor_body_idx, :]
        ref_root_lin_vel_base = quat_rotate_inverse_numpy(ref_anchor_quat_curr, ref_anchor_lin_vel_w)
        ref_root_ang_vel_base = quat_rotate_inverse_numpy(ref_anchor_quat_curr, ref_anchor_ang_vel_w)

        ref_anchor_pos = self.ref_anchor_pos_w
        rpy_xy = self._quat_to_rpy_xy(ref_anchor_quat_curr)
        joint_pos_future = self._select(self.ref_joint_pos_future)[:, 0, :]

        cmd = np.concatenate([
            ref_root_lin_vel_base[:, :2] * self.lin_vel_scale,
            ref_anchor_pos[:, 2:3] * self.height_scale,
            rpy_xy * self.rpy_scale,
            ref_root_ang_vel_base[:, 2:3] * self.ang_vel_scale,
            joint_pos_future * self.joint_pos_scale,
        ], axis=-1).astype(np.float32)
        # print(f"[obs] command_ref_motion shape={cmd.shape}")

        self._history_buffer = np.roll(self._history_buffer, 1, axis=0)
        self._history_buffer[0] = cmd[0]

    def compute(self) -> np.ndarray:
        out = self._history_buffer[self.history_steps].reshape(-1)
        # print(f"[obs] command_ref_motion shape={out.shape}")
        return out

class ref_joint_pos_future(_motion_joint_obs):
    def compute(self) -> np.ndarray:
        out = self._select(self.ref_joint_pos_future).reshape(-1)
        # print(f"[obs] ref_joint_pos_future shape={out.shape}")
        return out

# class ref_joint_vel_future(_motion_obs):
#     def compute(self) -> np.ndarray:
#         return self.ref_joint_vel_future.reshape(-1)
    
class ref_body_pos_future_local(_motion_body_obs):
    """
    Reference body position in motion anchor frame
    """
    def update(self, data: Dict[str, Any]) -> None:
        super().update(data)
        ref_body_pos_future_w = self._select(self.ref_body_pos_future_w)
        ref_anchor_pos_w: np.ndarray = self.ref_anchor_pos_w[:, None, None, :].copy()
        ref_anchor_quat_w: np.ndarray = self.ref_anchor_quat_w[:, None, None, :]

        ref_anchor_pos_w = np.tile(
            ref_anchor_pos_w,
            (1, self.n_selected_future_steps, self.n_selected_bodies, 1),
        )
        ref_anchor_quat_w = np.tile(
            ref_anchor_quat_w,
            (1, self.n_selected_future_steps, self.n_selected_bodies, 1),
        )

        ref_anchor_pos_w[..., 2] = 0.0
        ref_anchor_quat_w = projected_yaw_quat(ref_anchor_quat_w)

        ref_body_pos_future_local = quat_rotate_inverse_numpy(
            ref_anchor_quat_w, ref_body_pos_future_w - ref_anchor_pos_w
        )
        self.ref_body_pos_future_local = ref_body_pos_future_local
    
    def compute(self):
        return self.ref_body_pos_future_local.reshape(-1)
    
class ref_body_ori_future_local(_motion_body_obs):
    """
    Reference body orientation in motion anchor frame
    """
    def update(self, data: Dict[str, Any]) -> None:
        super().update(data)
        ref_body_quat_future_w = self._select(self.ref_body_quat_future_w)
        ref_anchor_quat_w = self.ref_anchor_quat_w[:, None, None, :]

        ref_anchor_quat_w = np.tile(
            ref_anchor_quat_w,
            (1, self.n_selected_future_steps, self.n_selected_bodies, 1),
        )
        ref_anchor_quat_w = projected_yaw_quat(ref_anchor_quat_w)

        ref_body_quat_future_local = quat_mul(
            quat_conjugate(ref_anchor_quat_w),
            ref_body_quat_future_w
        )
        self.ref_body_ori_future_local = matrix_from_quat(ref_body_quat_future_local)
    
    def compute(self):
        return self.ref_body_ori_future_local[:, :, :, :2, :3].reshape(-1)

class ref_root_ori_future_b(_motion_obs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.root_quat_offset = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion

    def reset(self):
        super().reset()
        motion_root_quat_w = self.ref_root_quat_w[0]
        robot_root_quat_w = self.state_processor.root_quat_w

        motion_root_quat_w = projected_yaw_quat(motion_root_quat_w)
        robot_root_quat_w = projected_yaw_quat(robot_root_quat_w)
        self.root_quat_offset = quat_mul(motion_root_quat_w, quat_conjugate(robot_root_quat_w))

    def update(self, data: Dict[str, Any]) -> None:
        super().update(data)
        ref_root_quat_future_w = self._select(self.ref_root_quat_future_w)
        robot_root_quat_w = self.state_processor.root_quat_w
        robot_root_quat_w = quat_mul(self.root_quat_offset, robot_root_quat_w)

        robot_root_quat_w = np.tile(robot_root_quat_w, (1, self.n_future_steps, 1))

        ref_root_quat_future_b = quat_mul(
            quat_conjugate(robot_root_quat_w),
            ref_root_quat_future_w
        )
        ref_root_ori_future_b = matrix_from_quat(ref_root_quat_future_b)[:, :, :2, :3].reshape(-1)
        self._history_buffer = np.roll(self._history_buffer, 1, axis=0)
        self._history_buffer[0] = ref_root_ori_future_b[:6]

    def compute(self):
        out = self._history_buffer[self.history_steps].reshape(-1)
        # print(f"[obs] ref_root_ori_future_b shape={out.shape}")
        return out


class ref_anchor_ori_future_b(_motion_obs):
    def __init__(self, history_steps: Optional[Union[Sequence[int], int]] = None, **kwargs):
        super().__init__(**kwargs)
        if history_steps is None:
            history_steps = [0]
        elif isinstance(history_steps, (int, np.integer)):
            history_steps = [int(history_steps)]
        else:
            history_steps = [int(step) for step in history_steps]

        self.history_steps = np.asarray(history_steps, dtype=int)
        if self.history_steps.ndim != 1:
            raise ValueError(f"history_steps must be 1D, got shape={self.history_steps.shape}")
        if self.history_steps.size == 0:
            raise ValueError("history_steps must not be empty")

        self._history_buffer = np.zeros((int(self.history_steps.max()) + 1, 6), dtype=np.float32)
        self.anchor_quat_offset = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    @staticmethod
    def _quat_to_rot6d(quat_wxyz: np.ndarray) -> np.ndarray:
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
        if quat_wxyz.ndim == 1:
            quat_wxyz = quat_wxyz[None, :]
        rot = matrix_from_quat(quat_wxyz)
        return rot[:, :2, :3].reshape(-1)

    def reset(self):
        super().reset()
        motion_anchor_quat_w = projected_yaw_quat(self.ref_anchor_quat_w[0])
        robot_anchor_quat_w = projected_yaw_quat(np.asarray(self.state_processor.root_quat_w, dtype=np.float32))
        self.anchor_quat_offset = quat_mul(motion_anchor_quat_w, quat_conjugate(robot_anchor_quat_w))
        self._history_buffer[:] = 0.0
        self.update({})

    def update(self, data: Dict[str, Any]) -> None:
        super().update(data)
        ref_quat = self.ref_anchor_quat_w[0]
        future_anchor_quat_w = quat_mul(self.anchor_quat_offset, ref_quat)

        robot_quat = np.asarray(self.state_processor.root_quat_w, dtype=np.float32)
        if robot_quat.shape != (4,):
            robot_quat = robot_quat.reshape(-1)[:4].astype(np.float32)

        ref_anchor_quat_future_b = quat_mul(quat_conjugate(robot_quat), future_anchor_quat_w)
        ref_anchor_ori_future_b = self._quat_to_rot6d(ref_anchor_quat_future_b)

        self._history_buffer = np.roll(self._history_buffer, 1, axis=0)
        self._history_buffer[0] = ref_anchor_ori_future_b[:6]

    def compute(self):
        out = self._history_buffer[self.history_steps].reshape(-1)
        # print(f"[obs] ref_anchor_ori_future_b shape={out.shape}")
        return out
