"""Rot6D <-> rotation matrix in X-VLA's *interleaved* layout.

X-VLA's EE6D 20D action space (`models/action_hub.py:EE6DActionSpace`) stores
rotation as 6D *interleaved*:

    Encode: rotation_matrix_to_6d(R) = [R[0,0], R[0,1], R[1,0], R[1,1], R[2,0], R[2,1]]
    Decode: 6d_to_matrix(d) — Gram-Schmidt using a1 = d[0::2], a2 = d[1::2]

This is **NOT** the standard "first-two-columns" Rot6D used elsewhere — confusing
the two breaks both encode and decode. See
`third_party/X-VLA/evaluation/SoftFold-Agilex/deploy/utils/rotation.py` for the
reference implementation we match here.

Used by `serve_policy_xvla.py` to:
  * convert 7D world-frame EE pose proprio (xyz + quat_wxyz) → interleaved Rot6D
    (after T_world_base inversion, in arm-base frame)
  * convert 20D EE6D action chunks (arm-base frame) → 16D `action_kind="ee"`
    chunks per `docs/deployment/multimodal_inference_protocol.md` §A.2.2.
"""

from __future__ import annotations

import numpy as np


def rotation_matrix_to_interleaved_6d(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → 6D interleaved representation.

    Layout: [R[0,0], R[0,1], R[1,0], R[1,1], R[2,0], R[2,1]] (rows × first two cols).
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"R must be (..., 3, 3); got {R.shape}")
    out = np.stack([R[..., 0, :2], R[..., 1, :2], R[..., 2, :2]], axis=-2)
    return out.reshape(*R.shape[:-2], 6)


def interleaved_6d_to_rotation_matrix(d: np.ndarray) -> np.ndarray:
    """6D interleaved representation → 3x3 rotation matrix via Gram-Schmidt.

    Input layout: [R[0,0], R[0,1], R[1,0], R[1,1], R[2,0], R[2,1]] (..., 6)
    """
    if d.shape[-1] != 6:
        raise ValueError(f"d must be (..., 6); got {d.shape}")
    a1 = d[..., 0:5:2]
    a2 = d[..., 1:6:2]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-12)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


if __name__ == "__main__":
    # Round-trip check
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(200):
        R = Rotation.random(random_state=rng).as_matrix()
        R_back = interleaved_6d_to_rotation_matrix(rotation_matrix_to_interleaved_6d(R))
        err = float(np.linalg.norm(R - R_back, "fro"))
        max_err = max(max_err, err)
    print(f"Rot6D interleaved round-trip max Frobenius err: {max_err:.2e}  (should be ~1e-7)")
