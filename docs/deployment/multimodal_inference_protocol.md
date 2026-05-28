# Multimodal Inference Protocol — Depth + EE Pose

**Status**: implemented 2026-05-18, inference-side only (no training pipeline changes).
**Scope**: sim01 deployment (`/data1/tim/workspace/deepdive_kai0`).

This document describes the extended deployment protocol that lets the KAI0
WebSocket inference stack host external VLA models beyond the legacy
`RGB → 14-dim joint` pipeline. New capabilities:

1. **Depth input** — D435 top_head, single channel, float32 meters.
2. **EE pose input** — dual-arm 7-dim (xyz + quat wxyz), world frame.
3. **EE pose action output** — Cartesian-trained models; IK runs inside the
   ROS2 node before driving joint-controlled hardware.
4. **Always-on EE pose ROS2 publishers** — regardless of model output type,
   `/policy/actions_ee_{left,right}` always carry pose data (FK of the joints
   in joint-mode; the model's raw output in EE-mode).
5. **Runtime switchable execution mode** — `joint` vs `ee_pose` selected
   at start time, decides whether the server's `actions` chunk is expected
   as 14-dim joints or 16-dim Cartesian.

This document is split into two halves:

- **Part A: User Guide** — how to launch / select modes / verify behavior.
- **Part B: Developer Guide** — how to implement a compatible WebSocket
  server (or model wrapper) that this stack will drive correctly.

---

## Quick-reference matrix

| Mode                          | Server output                        | What the ROS2 node does                                 | Hardware sees       |
|-------------------------------|--------------------------------------|----------------------------------------------------------|---------------------|
| `--execution-mode joint`      | `actions: [H, 14]`, `action_kind=joint` | publish joints to `/master/joint_*`; FK→EE on `/policy/actions_ee_*` | 14-dim joint cmd    |
| `--execution-mode ee_pose`    | `actions: [H, 16]`, `action_kind=ee`    | IK chunk → 14-dim joints → `/master/joint_*`; raw EE on `/policy/actions_ee_*` | 14-dim joint cmd (post-IK) |

The legacy default (no flags) is `joint` + depth/EE disabled, byte-identical to
pre-2026-05-18 behavior.

---

# Part A: User Guide

## A.1 Launching

The standard entry points (`start_autonomy.sh`, `start_autonomy_from_ckpt.sh`,
`start_policy_node.sh`) accept three new CLI flags:

```
--execution-mode {joint, ee_pose}     # action representation; default: joint
--enable-depth-input                  # pack D435 depth into obs; default: off
--enable-ee-pose-input                # pack dual-arm EE pose into obs; default: off
```

### A.1.1 Examples

```bash
# Legacy joint-only ckpt (unchanged from pre-2026-05-18):
./start_scripts/start_autonomy_from_ckpt.sh /path/to/ckpt

# Joint-output model that also consumes RGB + depth + EE proprioception:
./start_scripts/start_autonomy_from_ckpt.sh /path/to/ckpt \
    --enable-depth-input --enable-ee-pose-input

# Full Cartesian model (16-dim EE+gripper action):
./start_scripts/start_autonomy_from_ckpt.sh /path/to/ckpt \
    --execution-mode ee_pose \
    --enable-depth-input --enable-ee-pose-input

# Same, but only the policy node (when cameras + arms are already running):
./start_scripts/start_policy_node.sh --mode websocket \
    --execution-mode ee_pose --enable-ee-pose-input
```

### A.1.2 What gets published

Independent of `execution_mode`, the following ROS2 topics are published at
publish_rate (30 Hz default):

| Topic                                      | Type                         | Content                                       |
|--------------------------------------------|------------------------------|-----------------------------------------------|
| `/master/joint_left`                       | `sensor_msgs/JointState`     | left arm 6 joints + gripper (rad / m)         |
| `/master/joint_right`                      | `sensor_msgs/JointState`     | right arm 6 joints + gripper                  |
| `/policy/actions`                          | `sensor_msgs/JointState`     | full 14-dim action                            |
| `/policy/action_chunk`                     | `std_msgs/Float32MultiArray` | full predicted chunk (Rerun viz)              |
| **NEW** `/policy/actions_ee_left`          | `geometry_msgs/PoseStamped`  | left EE in world frame (xyz + quat_wxyz)      |
| **NEW** `/policy/actions_ee_right`         | `geometry_msgs/PoseStamped`  | right EE                                      |
| **NEW** `/policy/actions_gripper_left`     | `std_msgs/Float32`           | left gripper opening (m)                      |
| **NEW** `/policy/actions_gripper_right`    | `std_msgs/Float32`           | right gripper opening (m)                     |

PoseStamped `frame_id` is `"world"`. EE pose is consistent with whatever
joints were actually sent to hardware (in joint-mode it's a direct FK; in
ee-mode it's a round-trip via IK).

The new EE/gripper publishers are only active when at least one extended flag
is on. With all flags off, they stay silent (no PiperIK initialized).

### A.1.3 Verifying

After launch, the node prints one line summarizing the active configuration:

```
[policy_inference] execution_mode=joint | depth_in=False | ee_pose_in=False
```

Verify EE pose stream:

```bash
ros2 topic hz /policy/actions_ee_left            # expect ~30 Hz when executing
ros2 topic echo /policy/actions_ee_left --once   # inspect first frame
```

In `ee_pose` execution mode, every inference cycle logs IK timing:

```
IK ee→joint: 50 steps in 42ms
```

Failed IK steps are reported as a per-chunk count and held to the previous
good joint configuration:

```
IK chunk: 2/50 left + 0/50 right failed (held last good joint on failure)
```

---

## A.2 Coordinate Conventions

### A.2.1 EE pose representation

- **Frame**: `world` (sim01 right-handed, +z up, origin between the two arms).
- **Position**: meters.
- **Rotation**: unit quaternion `[w, x, y, z]` — **`w` first**, scalar-first
  convention. NOT the scipy / ROS default which is xyzw.
- **EE link**: URDF `gripper_base` link (one fixed link past `link6`, +0.1358 m
  along link6's local +z). This differs from `calib/piper_fk.py`'s `link6`
  convention — see §B.5.

Per-arm 7-dim vector layout:

```
[x, y, z, qw, qx, qy, qz]
```

### A.2.2 Action chunk layout

**Joint mode** (`action_kind="joint"`), 14-dim per timestep:

```
idx:  0    1    2    3    4    5    6     7    8    9   10   11   12   13
dim: L_j0 L_j1 L_j2 L_j3 L_j4 L_j5 L_grip R_j0 R_j1 R_j2 R_j3 R_j4 R_j5 R_grip
units: rad rad  rad  rad  rad  rad   m    rad rad  rad  rad  rad  rad    m
```

**EE mode** (`action_kind="ee"`), 16-dim per timestep:

```
idx:  0  1  2  3  4  5  6   7       8  9 10 11 12 13 14    15
dim:  Lx Ly Lz Lqw Lqx Lqy Lqz Lgrip Rx Ry Rz Rqw Rqx Rqy Rqz Rgrip
units: m  m  m  ─   ─  ─   ─    m    m  m  m  ─   ─   ─   ─    m
```

The chunk is `[H, 14]` or `[H, 16]` where H = action_horizon (50 default).

### A.2.3 Calibration source

`config/calibration.yml` holds `T_world_baseL` and `T_world_baseR` (4×4
homogeneous, meters). The policy node loads these at startup whenever any
extended flag is on. These transforms are produced by
`calib/solve_calibration.py` (DANIILIDIS hand-eye, see §calib README).

---

# Part B: Developer Guide

This section describes the contracts a custom WebSocket server (or any AI
working with this stack) must honor.

## B.1 WebSocket Protocol

### B.1.1 Transport

- **Protocol**: WebSocket over TCP, no compression, no message size limit.
- **Serialization**: msgpack with numpy extension (`msgpack_numpy`). All
  ndarray frames travel as-is — no JSON, no base64.
- **Endpoint**: a single endpoint (default `ws://<host>:8000`). One
  observation per send, one action response per recv.
- **Health check**: HTTP `GET /healthz` returns 200 "OK\n" on the same port.

### B.1.2 Handshake

Immediately after `connect()` the server MUST send one msgpack-packed dict —
this is the **initial metadata packet**. Recommended fields:

```python
{
    "action_kind": "joint",       # or "ee"; tells client what slot the
                                  # actions dict will populate.
    "action_dim":   14,           # 14 for joint, 16 for ee.
    "action_horizon": 50,         # chunk length H.
    "obs_keys": [                 # which obs dict keys the model needs.
        "images.top_head",
        "images.hand_left",
        "images.hand_right",
        "state",
        "prompt",
        # Optional, only if the model uses them:
        # "depth_top_head",
        # "ee_pose_left",
        # "ee_pose_right",
    ],
    "model_name": "your_model_v1",
}
```

The client (`policy_inference_node`) does NOT today error if metadata is
absent — it uses `result.get("action_kind", "joint")` per-response as the
authoritative tag. Sending metadata is recommended for diagnostics and
future capability negotiation, but not strictly required for correctness.

### B.1.3 Per-inference loop

After the handshake, the client sends a msgpack-packed observation dict and
expects a msgpack-packed result dict in response. Loop forever.

**Request** (client → server, every inference cycle):

```python
{
    "images": {
        "top_head":   np.uint8 array, shape (3, H, W) CHW,    # required
        "hand_left":  np.uint8 array, shape (3, H, W),         # required
        "hand_right": np.uint8 array, shape (3, H, W),         # required
    },
    "state": np.float32 array, shape (14,),                   # required, joint
                                                              # rad/m mix as in A.2.2
    "prompt": str,                                            # required
    # Optional (only present if user passed --enable-* flags):
    "depth_top_head":  np.float32 array, shape (H, W),        # meters
    "ee_pose_left":    np.float32 array, shape (7,),          # world, xyz+quat_wxyz
    "ee_pose_right":   np.float32 array, shape (7,),
}
```

H, W = 480, 640 for RGB and depth (D435 native at 640×480, sent uncropped).
Models usually resize internally to 224×224.

**Response** (server → client, every inference cycle):

```python
{
    "actions":     np.ndarray, shape (H_chunk, action_dim),   # required
    "action_kind": "joint" | "ee",                            # required when ee
                                                              # (absent → joint)
    "server_timing": {                                        # optional but
        "infer_ms": float,                                    # populated by the
        "prev_total_ms": float,                               # framework, used
    },                                                        # for diagnostics
}
```

Notes:

- `actions.shape[-1]` is the model's *padded* action_dim (e.g. 32 for pi05);
  the policy wrapper (`AgilexOutputs`) slices to the first 14 or 16 based on
  `action_kind`. Don't pad your response yourself — emit the meaningful dims
  and `AgilexOutputs` (or your own wrapper) handles the slice.
- `action_kind` MUST match the server's metadata declaration if metadata was
  sent. Mismatches log a warning and follow the per-response tag.

### B.1.4 Error handling

If `infer()` raises, the server sends a single string frame containing the
traceback, then closes with code 1011 (INTERNAL_ERROR). The client raises
`RuntimeError` with the traceback embedded.

## B.2 ROS2 Subscribed Topics (Observation Sources)

The policy node assembles obs from these incoming topics. To produce equivalent
data offline (e.g. for batch evaluation), match these conventions:

| Topic                                                | Type                       | Format                                          |
|------------------------------------------------------|----------------------------|-------------------------------------------------|
| `/camera_f/camera/color/image_raw`                   | `sensor_msgs/Image`        | RGB or BGR; node JPEG-roundtrips to BGR then→RGB|
| `/camera_l/camera/color/image_raw`                   | `sensor_msgs/Image`        | same                                            |
| `/camera_r/camera/color/image_raw`                   | `sensor_msgs/Image`        | same                                            |
| `/camera_f/camera/aligned_depth_to_color/image_raw`  | `sensor_msgs/Image`        | uint16 millimeters → node converts to float32 m |
| `/puppet/joint_left`                                 | `sensor_msgs/JointState`   | `position[7]` = 6 joints (rad) + gripper (m)    |
| `/puppet/joint_right`                                | `sensor_msgs/JointState`   | same                                            |
| `/policy/execute`                                    | `std_msgs/Bool`            | runtime execute toggle (observe-only when False)|

EE pose is computed inside the node via FK (`PiperIK.fk_homogeneous` over the
URDF chain + `T_world_baseL/R` from calibration). The node never subscribes
to an external EE pose topic.

## B.3 Internal Data Flow

```
                ┌────────── policy_inference_node ──────────┐
                │                                            │
RGB topics ──→  │  _get_observation():                       │  msgpack  WebSocket
JointState ──→  │    pack state[14] + images[3 CHW]          │ ────────→ ┌──────────────┐
Depth topic ──→ │    if enable_depth:    +depth_top_head     │           │   server     │
                │    if enable_ee_pose:  +ee_pose_{l,r}      │           │ (this stack  │
                │                       (via PiperFK + calib) │           │  or external)│
                │                                            │ ←──────── │              │
                │  result.action_kind:                       │  msgpack  └──────────────┘
                │    "joint" → use actions[H, 14] directly   │              actions chunk
                │    "ee"    → _ik_chunk_to_joint(actions)   │              + action_kind
                │              → [H, 14]                     │
                │                                            │
                │  StreamActionBuffer (joint domain):        │
                │    smoothing + RTC + jump protection       │
                │                                            │
                │  _publish_action() at 30 Hz:               │
                │    pop [14] → /master/joint_{l,r}          │ ──→ arm_teleop_node ──→ CAN bus
                │    /policy/actions (full [14])             │
                │    FK([:6]) per arm →                      │
                │      /policy/actions_ee_{left,right}       │ ──→ external monitors
                │      /policy/actions_gripper_{left,right}  │
                └────────────────────────────────────────────┘
```

Key invariants:

1. **The joint domain is canonical.** StreamActionBuffer always operates on
   the 14-dim joint chunk. EE→joint conversion (IK) happens once per chunk
   *before* buffering. EE topics are always FK of the actually-sent joints.
2. **Hardware sees only joints.** `/master/joint_*` is the only topic
   `arm_teleop_node` consumes. The IK module lives one node upstream.
3. **Backward compat is structural.** With all extended flags off, the new
   code paths short-circuit (`self._ik` is None, EE publishers stay silent,
   obs dict has only the legacy keys). Bit-identical to pre-2026-05-18.

## B.4 Server-side Policy Wrapper Schema

If you build a server inside this stack (using `openpi.serving`), the
`AgilexInputs` / `AgilexOutputs` wrappers in
`kai0/src/openpi/policies/agilex_policy.py` handle the conversion between
the WebSocket obs dict and the model's expected tensor layout.

### B.4.1 AgilexInputs

```python
@dataclasses.dataclass(frozen=True)
class AgilexInputs(transforms.DataTransformFn):
    action_dim: int                # padded action dim (e.g. 32 for pi05)
    model_type: _model.ModelType   # PI0 / PI0_FAST / PI0_RTC / PI05 / PI05_RTC
    mask_state: bool = False       # debug: zero out state input
    enable_depth: bool = False     # require depth_top_head in obs
    enable_ee_pose: bool = False   # require ee_pose_{left,right} in obs
```

Output dict keys:

```python
{
    "image":      {"base_0_rgb": (H, W, 3) uint8, "left_wrist_0_rgb": ..., "right_wrist_0_rgb": ...},
    "image_mask": {"base_0_rgb": True/False, ...},
    "state":      (action_dim,) float, padded with zeros past dim 14,
    "prompt":     str,
    # When enable_depth:
    "depth":      {"base_0_depth": (H, W) float32, meters},
    # When enable_ee_pose:
    "ee_pose":    {"left": (7,) float32, "right": (7,) float32},
    # Plus action_mask, actions, frame_index, episode_length, etc. as applicable.
}
```

Models needing depth or EE pose should read from these sibling top-level keys
(NOT from `state` — `state` stays 14-dim joint to preserve compatibility with
legacy ckpts).

### B.4.2 AgilexOutputs

```python
@dataclasses.dataclass(frozen=True)
class AgilexOutputs(transforms.DataTransformFn):
    action_kind: str = "joint"  # or "ee"; SET AT POLICY CREATION TIME
```

`action_kind` is a property of how the model was trained, not the runtime
data. Don't try to infer it from `actions.shape[-1]` — that's always the
padded action_dim. Set it explicitly when wiring a new TrainConfig:

```python
# In your TrainConfig (e.g. kai0/src/openpi/training/config.py):
TrainConfig(
    ...
    data=LeRobotAgilexDataConfig(
        repack_transforms=...,
        # default joint output:
        data_transforms=transforms.Group(
            inputs=[AgilexInputs(action_dim=32, model_type=ModelType.PI05)],
            outputs=[AgilexOutputs()],
        ),
    ),
    # Or, for an ee-mode model:
    data=LeRobotAgilexDataConfig(
        data_transforms=transforms.Group(
            inputs=[AgilexInputs(action_dim=32, model_type=ModelType.PI05,
                                 enable_ee_pose=True, enable_depth=True)],
            outputs=[AgilexOutputs(action_kind="ee")],
        ),
    ),
    policy_metadata={"action_kind": "ee", "action_dim": 16,
                     "obs_keys": ["images.*", "state", "depth_top_head",
                                  "ee_pose_left", "ee_pose_right"]},
)
```

The `policy_metadata` dict flows through to the WebSocket initial packet
(`websocket_policy_server.py:54`).

## B.5 Kinematics: PiperIK and PiperFK Conventions

There are **two** FK conventions in the codebase, used for different purposes:

| Module                       | EE link        | Used by                                       |
|------------------------------|----------------|-----------------------------------------------|
| `calib/piper_fk.py`          | `link6`        | Rerun 3D visualization (`world/{arm}/link_i`) |
| `calib/piper_ik.py` (NEW)    | `gripper_base` | obs `ee_pose_*` + EE pose publishers + IK     |

The two differ by a fixed +0.1358 m translation in link6's local +z direction.
**For the multimodal protocol (this document) you always work in the
`gripper_base` convention.** Wrap your model's training data the same way if
you're training a new EE-mode policy — otherwise the model will operate in
the wrong frame and IK at deploy time will fail.

### B.5.1 PiperIK API

```python
from calib.piper_ik import PiperIK

ik = PiperIK()                                   # loads calib/piper_local.urdf

# FK (gripper_base, in arm base frame):
T = ik.fk_homogeneous(q6_rad)                    # 4×4 numpy
xyz, quat_wxyz = ik.fk_xyz_quat(q6_rad)          # ((3,), (4,))

# IK (seeded, returns success flag):
q6, ok = ik.solve(target_pos, target_quat_wxyz, q_seed,
                  tol_pos=5e-3, tol_rot=5e-2)
# or directly from 4×4 matrix:
q6, ok = ik.solve_mat(T_target, q_seed)
```

Seed handling is critical: ikpy uses scipy `least_squares` which only finds a
local minimum. **Always pass `q_seed` ≈ current joint state** (or the previous
chunk-step's solution for trajectory tracking). Failure modes:

- `ok=False, q6=q_seed.copy()` → solver hit `x0 is infeasible` (seed outside
  URDF joint bounds) or the FK-roundtrip check exceeded tolerance.
- Caller decides hold-vs-extrapolate; the policy node holds the last good
  joint configuration (no extrapolation), see `_ik_chunk_to_joint`.

### B.5.2 World ↔ Base composition

The policy node applies `T_world_baseL/R` (from `config/calibration.yml`)
around PiperIK so the model sees a single unified world frame:

```python
# obs (FK):
T_base_ee   = ik.fk_homogeneous(q6_per_arm)
T_world_ee  = T_world_base @ T_base_ee
# action (IK):
T_world_ee  = build_pose(model_output_xyz, model_output_quat)
T_base_ee   = inv(T_world_base) @ T_world_ee
q6, ok = ik.solve_mat(T_base_ee, q_seed)
```

If you implement an external server, your model is free to use whichever
frame matches its training data — but the obs you receive will always be in
the world frame above, and the actions you emit will be interpreted as world
frame too. If your model trained on a different frame, do the conversion
inside your server's `infer()`.

## B.6 ROS2 Parameters (full list of new params)

Declared in `ros2_ws/src/piper/scripts/policy_inference_node.py:309-332`.

| Parameter             | Type   | Default              | Purpose                                          |
|-----------------------|--------|----------------------|--------------------------------------------------|
| `execution_mode`      | string | `"joint"`            | `joint` or `ee_pose`. Sanity check vs server.    |
| `enable_depth_input`  | bool   | `false`              | Pack depth into obs.                             |
| `enable_ee_pose_input`| bool   | `false`              | Pack EE pose into obs.                           |
| `urdf_path`           | string | auto-resolved        | Override URDF for PiperIK init.                  |
| `calibration_yaml`    | string | auto-resolved        | YAML with `transforms.T_world_baseL/R`.          |

Defaults preserve legacy behavior. Setting `execution_mode=ee_pose` without
also turning on any modality flag is allowed (and logged) — it just means
the inbound observation only has joint state but the outbound actions are
expected to be EE Cartesian.

## B.7 Implementing a Compatible Server From Scratch

Minimal Python skeleton for a server that hosts a Cartesian-output VLA model
on this stack:

```python
import asyncio
import websockets.asyncio.server as server
from openpi_client import msgpack_numpy
import numpy as np

class MyEEServer:
    def __init__(self, model):
        self.model = model
        self.metadata = {
            "action_kind":    "ee",
            "action_dim":      16,
            "action_horizon":  50,
            "obs_keys": [
                "images.top_head", "images.hand_left", "images.hand_right",
                "state", "prompt",
                "depth_top_head", "ee_pose_left", "ee_pose_right",
            ],
            "model_name": "my_ee_vla_v1",
        }

    async def handler(self, ws):
        packer = msgpack_numpy.Packer()
        await ws.send(packer.pack(self.metadata))
        while True:
            obs = msgpack_numpy.unpackb(await ws.recv())
            # obs has keys: images (dict of 3 CHW uint8 arrays),
            #               state (14 float), prompt (str),
            #               depth_top_head (H, W float32 m),
            #               ee_pose_left/right ((7,) float32 xyz+quat_wxyz).

            # Your model: returns [H_chunk, 16] EE+gripper actions.
            ee_chunk = self.model.infer(
                rgb=obs["images"],
                depth=obs["depth_top_head"],
                state=obs["state"],
                ee_l=obs["ee_pose_left"],
                ee_r=obs["ee_pose_right"],
                prompt=obs["prompt"],
            )  # np.ndarray, shape (50, 16) float32

            result = {
                "actions":     ee_chunk,
                "action_kind": "ee",
            }
            await ws.send(packer.pack(result))

async def main():
    srv = MyEEServer(model=MyModel())
    async with server.serve(srv.handler, "0.0.0.0", 8000,
                            compression=None, max_size=None) as s:
        await s.serve_forever()

asyncio.run(main())
```

Run on the same host (or any host reachable from sim01) on a port that
matches `--port` passed to `start_policy_node.sh --mode websocket`.

## B.8 Failure Modes & Debug Hints

| Symptom                                                           | Likely cause                                                            | Fix                                                                                              |
|-------------------------------------------------------------------|-------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| `Server returned action_kind=ee but PiperIK not loaded`           | Launched without any extended flag, server emits ee actions             | Add `--execution-mode ee_pose` or any `--enable-*` flag at start time                            |
| `IK chunk: N/50 left+M/50 right failed`                           | Model emits poses outside reachable workspace, OR wrong frame convention | Check model output is in world frame + URDF gripper_base convention (§B.5). Verify q_seed sanity. |
| `enable_depth=True but obs missing key 'depth_top_head'`          | Client/server schema mismatch                                            | Make sure server populates depth key whenever metadata declares it required                      |
| `/policy/actions_ee_left` topic has no publishers                 | No extended flag passed, PiperIK never initialized                       | Add at least `--enable-ee-pose-input` for monitoring                                              |
| `PiperIK init` log shows `Failed to init PiperIK`                 | URDF not found, or ikpy not installed in system Python                  | Verify `pip install --user --break-system-packages ikpy` for `/usr/bin/python3`                  |
| Jump protection rejects every action with `Δ=large°`              | IK solution far from current joints (workspace boundary)                | Reduce action magnitude; check IK seed sanity; investigate near-singularity poses                |

Useful log filters during a run:

```bash
ros2 run piper policy_inference_node 2>&1 | grep -E 'IK|PiperIK|execution_mode|action_kind'
```

---

# Part C: Reference

## C.1 File Layout

| File                                                            | Role                                                       |
|-----------------------------------------------------------------|------------------------------------------------------------|
| `calib/piper_ik.py`                                             | PiperIK class (FK + IK, URDF gripper_base)                 |
| `calib/piper_fk.py`                                             | PiperFK class (DH-based, link6 EE; Rerun viz only)         |
| `calib/piper_local.urdf`                                        | Piper 6-DOF URDF + gripper                                 |
| `config/calibration.yml`                                        | Hand-eye + world←base transforms                           |
| `kai0/src/openpi/policies/agilex_policy.py`                     | AgilexInputs / AgilexOutputs schema                        |
| `kai0/src/openpi/serving/websocket_policy_server.py`            | WS server                                                   |
| `kai0/packages/openpi-client/src/openpi_client/websocket_client_policy.py` | WS client                                          |
| `ros2_ws/src/piper/scripts/policy_inference_node.py`            | Main deployment node                                       |
| `ros2_ws/src/piper/launch/autonomy_launch.py`                   | ROS2 launch wiring                                         |
| `start_scripts/start_autonomy.sh`                               | Top-level CLI entry                                        |
| `start_scripts/start_autonomy_from_ckpt.sh`                     | CLI wrapper that reads ckpt's `train_config.json`          |
| `start_scripts/start_policy_node.sh`                            | Standalone policy node CLI (cameras + arms must run separately) |

## C.2 What Is NOT Yet Supported

- **Wrist depth (D405)** — disabled by `config/camera_depth_flags.py` for USB
  bandwidth + flicker reasons. Re-enabling needs verification; see
  `docs/deployment/realsense_anti_flicker_2026-04-27.md`.
- **EE pose recording in autonomy dataset** —
  `autonomy_recorder_node.py` still writes the legacy 14-dim parquet
  schema. EE poses appear only on live ROS2 topics, not in recorded episodes.
- **Runtime hot-swap of execution_mode** — current implementation is
  start-time only. Toggling requires restart. Adding a `ros2 param set`
  callback is straightforward but not done.
- **Multi-model servers / dynamic ckpt swap** — `serve_policy.py` still
  binds one ckpt per process. External servers (per §B.7) are free to do
  whatever they want internally.
- **EE pose subscribed from external source** — the node always FK-computes
  EE pose from `/puppet/joint_*`. If you wanted to inject EE pose from a
  vision tracker, you'd need a small node-side change.

## C.3 Verification Checklist (hardware-required)

Before declaring a new ckpt deployable:

1. **Legacy regression** — launch without flags, compare published actions
   against a recorded baseline. Expect bit-identical behavior.
2. **EE pose self-consistency** — record `/policy/actions` and
   `/policy/actions_ee_left`, verify `FK(joint[:7]) == ee_left` within
   1 mm / 0.1° at each timestep.
3. **IK round-trip** (ee mode) — record both `/policy/actions_ee_*` and
   `/master/joint_*`; offline verify `FK(joint) ≈ ee` within 5 mm (the
   default IK tolerance).
4. **Depth pipeline** — confirm `obs['depth_top_head']` shape `(480, 640)
   float32` with values in 0.3–1.5 m range at the desk.
5. **Mode-mismatch warning** — launch with `--execution-mode ee_pose` but a
   joint-output ckpt; expect a warning log line, but inference continues
   based on the per-response `action_kind` (server is the authority).
6. **No-flag run** — re-run the existing `task_a_new_smooth_800` idle-table
   test; jitter signature should match the 2026-05-15 baseline.

---

**Last updated**: 2026-05-18
**Implementation commit**: see `git log -- calib/piper_ik.py` for the first
appearance of this feature set.
