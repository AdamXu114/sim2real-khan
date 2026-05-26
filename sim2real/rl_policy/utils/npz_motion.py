"""Raw NPZ motion dataset adapter for Jaka-style motion data.

The NPZ files contain per-frame motion data with fields:
  joint_pos      [T, n_joints]   – joint positions (IsaacLab order)
  joint_vel      [T, n_joints]   – joint velocities (IsaacLab order)
  body_pos_w     [T, n_bodies, 3]  – body positions in world frame
  body_quat_w    [T, n_bodies, 4]  – body orientations (w,x,y,z) in world frame
  body_lin_vel_w [T, n_bodies, 3]  – body linear velocities in world frame
  body_ang_vel_w [T, n_bodies, 3]  – body angular velocities in world frame
  fps            [1]               – frames per second
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from sim2real.rl_policy.utils.motion import MotionData


class NpzMotionDataset:
    """Direct NPZ motion loader that wraps raw ``.npz`` files into the
    :class:`MotionData` container expected by the framework.

    Unlike the ``any4hdmi`` path, this loader does **no** joint reordering
    internally – the data is stored and returned in the order present in the
    file (typically IsaacLab order for Jaka).  Any reindexing to simulation
    order must be performed by the caller / observation classes.
    """

    def __init__(
        self,
        npz_path: str,
        joint_names: List[str],
        body_names: List[str],
    ):
        npz_path = str(Path(npz_path).expanduser())
        data = np.load(npz_path)

        self.joint_pos_all: np.ndarray = data["joint_pos"].astype(np.float32)        # [T, J]
        self.joint_vel_all: np.ndarray = data["joint_vel"].astype(np.float32)        # [T, J]
        self.body_pos_w_all: np.ndarray = data["body_pos_w"].astype(np.float32)      # [T, B, 3]
        self.body_quat_w_all: np.ndarray = data["body_quat_w"].astype(np.float32)    # [T, B, 4]
        self.body_lin_vel_w_all: np.ndarray = data["body_lin_vel_w"].astype(np.float32)  # [T, B, 3]
        self.body_ang_vel_w_all: np.ndarray = data["body_ang_vel_w"].astype(np.float32)  # [T, B, 3]

        self.fps: int = int(data["fps"].item() if data["fps"].ndim > 0 else int(data["fps"]))
        self.num_steps: int = self.joint_pos_all.shape[0]

        self.joint_names: List[str] = list(joint_names)
        self.body_names: List[str] = list(body_names)

    # --------------------------------------------------------------------- #
    #  Slice API – compatible with MotionDataset.get_slice signature
    # --------------------------------------------------------------------- #

    def get_slice(
        self,
        motion_ids: np.ndarray,
        starts: np.ndarray,
        steps: np.ndarray,
    ) -> MotionData:
        """Return a :class:`MotionData` slice.

        Parameters
        ----------
        motion_ids : ndarray [N]
            Ignored (single-motion dataset), kept for API compatibility.
        starts : ndarray [N]
            Per-batch starting frame index.
        steps : ndarray [S]
            Offsets relative to *starts* to gather (e.g. ``[0]`` for current
            frame only, or ``[-2, -1, 0, 1, 2]`` for a window).

        Returns
        -------
        MotionData
            Fields shaped ``[N, S, ...]`` matching the existing convention.
        """
        starts = np.asarray(starts, dtype=np.int64)           # [N]
        steps = np.asarray(steps, dtype=np.int64)              # [S]
        # Build index grid [N, S]
        idx = starts[:, None] + steps[None, :]                 # [N, S]
        idx = np.clip(idx, 0, self.num_steps - 1)

        return MotionData(
            joint_pos=self.joint_pos_all[idx],                 # [N, S, J]
            joint_vel=self.joint_vel_all[idx],                 # [N, S, J]
            body_pos_w=self.body_pos_w_all[idx],               # [N, S, B, 3]
            body_quat_w=self.body_quat_w_all[idx],             # [N, S, B, 4]
            body_lin_vel_w=self.body_lin_vel_w_all[idx],       # [N, S, B, 3]
            body_ang_vel_w=self.body_ang_vel_w_all[idx],       # [N, S, B, 3]
        )