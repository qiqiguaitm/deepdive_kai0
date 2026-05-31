#!/usr/bin/env python3
"""
推理服务器测试 (统一入口) — 完全由 server metadata 驱动, 与具体模型解耦。

本工具只认 `docs/deployment/multimodal_inference_protocol.md` 协议, 不含任何
模型 (pi05 / XVLA / ...) 专属逻辑:
  • 连上后读 server 握手 metadata:
      action_kind ∈ {joint, ee}  → 决定用哪套质量校验 (14D 关节 / 16D world EE)
      obs_keys: [...]            → 决定合成 obs 时补哪些可选模态
  • 可选模态合成器 (OPTIONAL_SYNTH) 按 obs_keys 自动启用:
      depth_top_head    (H,W) float32 m
      ee_pose_left/right (7,) world xyz+quat_wxyz
  任何符合该协议的 server 都能被本工具直接测, 无需改代码。

模式:
  --check latency   延迟基准测试 (合成 payload)
  --check quality   输出质量验证 (数值范围 / 一致性 / 敏感性 / 平滑性)
  --check all       全部检查

前置: serve_policy.py / serve_policy_xvla.py / 任意协议兼容 server 在运行

Usage:
  # joint server (默认 :8000) — metadata 自动判别
  python3 start_scripts/kai/test_inference_server.py --check all
  # XVLA ee server (:8003) — 从 metadata 自动识别 ee + 自动补 ee_pose
  python3 start_scripts/kai/test_inference_server.py --check all --port 8003
  # metadata 缺失时手动指定协议 / 模态
  python3 start_scripts/kai/test_inference_server.py --port 8003 --mode ee --with ee_pose
"""
import argparse
import time
import sys
import numpy as np

sys.path.insert(0, 'packages/openpi-client/src')
sys.path.insert(0, 'kai0/packages/openpi-client/src')
from openpi_client import websocket_client_policy

# Piper 关节限位 (radians) — 来源: 官方 URDF piper_description.urdf
JOINT_LIMITS = [
    (-2.618, 2.618),   # joint 0
    ( 0.000, 3.140),   # joint 1 (非对称)
    (-2.967, 0.000),   # joint 2 (非对称)
    (-1.745, 1.745),   # joint 3
    (-1.220, 1.220),   # joint 4
    (-2.0944, 2.0944), # joint 5
    ( 0.000, 0.035),   # gripper (rad, URDF 值)
]
ALL_LIMITS = JOINT_LIMITS + JOINT_LIMITS

# ── EE-mode 16D 布局 (multimodal_inference_protocol.md §A.2.2) ────────────────
#   L: [0:3]=xyz  [3:7]=quat_wxyz  [7]=grip   R: [8:11]=xyz [11:15]=quat [15]=grip
EE_XYZ_L = slice(0, 3); EE_QUAT_L = slice(3, 7); EE_GRIP_L = 7
EE_XYZ_R = slice(8, 11); EE_QUAT_R = slice(11, 15); EE_GRIP_R = 15
EE_XYZ_IDX = [0, 1, 2, 8, 9, 10]   # 双臂 xyz 列, 用于位移类指标
EE_POS_BOUND = 2.0                  # |x|,|y|,|z| 超过即告警 (m)
EE_GRIP_RANGE = (-0.05, 0.12)       # 夹爪开合合理区间 (m), 含 SoftFold 负值闭合
QUAT_NORM_TOL = 0.05                # |‖quat‖-1| 容差

# 协议定义的相机/状态分辨率 (B.1.3): RGB/depth 原生 640×480
RGB_HW = (480, 640)


# ══════════════════════════════════════════════════════════════════════════════
# 协议 obs 合成 — 由 metadata.obs_keys 驱动, 与模型解耦
# ══════════════════════════════════════════════════════════════════════════════

def _norm_quat_wxyz(q):
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else np.array([1, 0, 0, 0], dtype=np.float32)


# 合成 ee_pose 基准 (world, xyz + quat_wxyz). 朝下姿态 ≈ 绕 x 转 180° → wxyz[0,1,0,0]。
_EE_BASE = {
    'ee_pose_left':  np.array([0.45, 0.22, 0.22, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    'ee_pose_right': np.array([0.45, -0.22, 0.22, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
}


def _synth_ee_pose(key, randomize=False, override=None):
    """合成一条 7D world ee_pose (xyz + 单位 quat_wxyz)。"""
    base = _EE_BASE[key] if override is None else np.asarray(override, dtype=np.float32)
    xyz, quat = base[:3].copy(), base[3:7].copy()
    if randomize:
        xyz = xyz + np.random.uniform(-0.05, 0.05, 3).astype(np.float32)
        quat = quat + np.random.uniform(-0.1, 0.1, 4).astype(np.float32)
    return np.concatenate([xyz, _norm_quat_wxyz(quat)]).astype(np.float32)


def _synth_depth(randomize=False):
    """合成 D435 top_head 深度 (H,W) float32 米, 桌面 0.3-1.5m 量级。"""
    if randomize:
        return np.random.uniform(0.3, 1.5, RGB_HW).astype(np.float32)
    return np.full(RGB_HW, 0.6, dtype=np.float32)


# obs_key → 合成器。新增模态只需在此登记, 校验/构造逻辑无需改。
OPTIONAL_SYNTH = {
    'depth_top_head':  lambda rnd, ov: _synth_depth(rnd),
    'ee_pose_left':    lambda rnd, ov: _synth_ee_pose('ee_pose_left', rnd, ov),
    'ee_pose_right':   lambda rnd, ov: _synth_ee_pose('ee_pose_right', rnd, ov),
}


def build_payload(extras=(), img=None, state=None, prompt='Flatten and fold the cloth.',
                  randomize=False, overrides=None):
    """构造一帧协议 obs。

    extras: 要附带的可选 obs_key 集合 (OPTIONAL_SYNTH 的键)。
    overrides: {obs_key: value} 显式覆盖某模态 (敏感性测试用)。
    """
    overrides = overrides or {}
    if img is None:
        img = np.random.randint(0, 255, (*RGB_HW, 3), dtype=np.uint8)
    if state is None:
        state = np.array([0.0, 0.3, -0.5, 0.0, 0.5, 0.0, 0.03,
                          0.0, 0.3, -0.5, 0.0, 0.5, 0.0, 0.03], dtype=np.float32)
    payload = {
        'images': {'top_head': img, 'hand_left': img, 'hand_right': img},
        'state': state,
        'prompt': prompt,
    }
    for key in extras:
        synth = OPTIONAL_SYNTH.get(key)
        if synth is not None:
            payload[key] = synth(randomize, overrides.get(key))
    return payload


# ══════════════════════════════════════════════════════════════════════════════
# Latency benchmark (协议无关; 自动附带 extras)
# ══════════════════════════════════════════════════════════════════════════════

def check_latency(policy, extras, rounds=20, warmup=3):
    print('=' * 60)
    print(f'延迟基准测试 ({rounds} rounds, {warmup} warmup, extras={list(extras) or "—"})')
    print('=' * 60)

    for i in range(warmup):
        t0 = time.monotonic()
        r = policy.infer(build_payload(extras))
        dt = (time.monotonic() - t0) * 1000
        shape = r['actions'].shape if 'actions' in r else 'N/A'
        print(f'  warmup {i+1}: {dt:.0f}ms  shape={shape}  kind={r.get("action_kind", "joint")}')

    latencies = []
    for i in range(rounds):
        payload = build_payload(
            extras,
            img=np.random.randint(0, 255, (*RGB_HW, 3), dtype=np.uint8),
            state=np.random.randn(14).astype(np.float32),
            randomize=True,
        )
        t0 = time.monotonic()
        policy.infer(payload)
        latencies.append((time.monotonic() - t0) * 1000)
        print(f'  round {i+1:2d}: {latencies[-1]:.0f}ms')

    lat = np.array(latencies)
    print(f'\n  avg={lat.mean():.0f}ms  std={lat.std():.0f}ms  '
          f'p50={np.median(lat):.0f}ms  p95={np.percentile(lat, 95):.0f}ms  '
          f'p99={np.percentile(lat, 99):.0f}ms  max={lat.max():.0f}ms')
    print(f'  throughput: {1000/lat.mean():.1f} infer/s')

    ok = lat.mean() < 300
    tag = 'PASS' if ok else ('MARGINAL' if lat.mean() < 500 else 'FAIL')
    print(f'  结论: {tag} (要求 < 300ms, 实际 {lat.mean():.0f}ms)')
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# Quality checks — joint (14D)
# ══════════════════════════════════════════════════════════════════════════════

def _check_range(actions):
    violations = 0
    for dim in range(14):
        lo, hi = ALL_LIMITS[dim]
        margin = 0.5
        below = (actions[:, dim] < lo - margin).sum()
        above = (actions[:, dim] > hi + margin).sum()
        if below or above:
            violations += below + above
            jn = f"{'L' if dim < 7 else 'R'}_j{dim % 7}"
            print(f'    [WARN] {jn}: {below} below {lo-margin:.2f}, {above} above {hi+margin:.2f}')
    return violations == 0


def check_quality_joint(policy, extras):
    print('=' * 60)
    print('推理质量验证 (joint 14D)')
    print('=' * 60)
    results = {}

    policy.infer(build_payload(extras))  # warmup

    # 1. 形状和范围
    print('\n--- Test 1: Action 形状和数值范围 ---')
    r = policy.infer(build_payload(extras))
    actions = r['actions']
    print(f'  Shape: {actions.shape}  dtype: {actions.dtype}')
    print(f'  范围: [{actions.min():.4f}, {actions.max():.4f}]  均值: {actions.mean():.4f}')
    for dim in range(14):
        jn = f"{'L' if dim < 7 else 'R'}_j{dim % 7}"
        v = actions[:, dim]
        print(f'  {jn}: [{v.min():.3f}, {v.max():.3f}] mean={v.mean():.3f}')
    range_ok = _check_range(actions) and actions.std() > 0.001
    results['shape_range'] = range_ok
    print(f'  → {"PASS" if range_ok else "WARN"}')

    # 2. 一致性
    print('\n--- Test 2: 一致性 (同一输入 x5) ---')
    fixed = build_payload(
        extras,
        img=np.random.randint(0, 255, (*RGB_HW, 3), dtype=np.uint8),
        state=np.array([0.1, 0.5, -0.8, 0, 0.6, -0.1, 0.04,
                        -0.1, 0.5, -0.8, 0, 0.6, 0.1, 0.04], dtype=np.float32),
    )
    all_a = np.stack([policy.infer(fixed)['actions'] for _ in range(5)])
    std_across = all_a.std(axis=0).mean()
    print(f'  跨次 std: {std_across:.4f} rad  max偏差: {np.abs(all_a - all_a.mean(0)).max():.4f}')
    results['consistency'] = std_across < 0.3
    print(f'  → {"PASS" if results["consistency"] else "WARN"} (std < 0.3)')

    # 3. 敏感性
    print('\n--- Test 3: 敏感性 (不同状态) ---')
    s_a = np.zeros(14, dtype=np.float32)
    s_b = np.array([0.5, 1.0, -1.0, 0.3, 0.8, -0.5, 0.05,
                    -0.5, 1.0, -1.0, -0.3, 0.8, 0.5, 0.05], dtype=np.float32)
    img = np.random.randint(0, 255, (*RGB_HW, 3), dtype=np.uint8)
    diff = np.abs(policy.infer(build_payload(extras, img, s_a))['actions']
                  - policy.infer(build_payload(extras, img, s_b))['actions']).mean()
    results['sensitivity'] = diff > 0.01
    print(f'  mean diff: {diff:.4f} rad → {"PASS" if results["sensitivity"] else "FAIL"}')

    # 4. 平滑性
    print('\n--- Test 4: 时序平滑性 ---')
    a = policy.infer(fixed)['actions']
    step_diffs = np.abs(np.diff(a, axis=0))
    results['smoothness'] = step_diffs.max() < 0.5
    print(f'  max jump: {step_diffs.max():.4f} rad  mean: {step_diffs.mean():.4f}')
    print(f'  → {"PASS" if results["smoothness"] else "WARN"} (max < 0.5)')

    _timing_test(policy, build_payload(extras), results)
    return _summarize(results)


# ══════════════════════════════════════════════════════════════════════════════
# Quality checks — ee (16D, world EE+gripper)
# ══════════════════════════════════════════════════════════════════════════════

def _check_ee_one_arm(actions, label, xyz_sl, quat_sl, grip_i):
    """检查单臂 EE 输出: xyz 工作空间 + quat 单位模长 + grip 区间。返回 ok。"""
    xyz = actions[:, xyz_sl]; quat = actions[:, quat_sl]; grip = actions[:, grip_i]
    qnorm = np.linalg.norm(quat, axis=1)
    pos_bad = int((np.abs(xyz) > EE_POS_BOUND).sum())
    quat_err = float(np.abs(qnorm - 1.0).max())
    grip_bad = int(((grip < EE_GRIP_RANGE[0]) | (grip > EE_GRIP_RANGE[1])).sum())
    print(f'  [{label}] xyz x[{xyz[:,0].min():.3f},{xyz[:,0].max():.3f}] '
          f'y[{xyz[:,1].min():.3f},{xyz[:,1].max():.3f}] z[{xyz[:,2].min():.3f},{xyz[:,2].max():.3f}] m')
    print(f'        |quat|∈[{qnorm.min():.4f},{qnorm.max():.4f}] (err={quat_err:.4f})  '
          f'grip[{grip.min():.4f},{grip.max():.4f}] m')
    ok = True
    if pos_bad:
        print(f'    [WARN] {pos_bad} 帧 |xyz| > {EE_POS_BOUND}m (疑似越界/坐标系错)'); ok = False
    if quat_err > QUAT_NORM_TOL:
        print(f'    [WARN] quat 非单位 (err {quat_err:.4f} > {QUAT_NORM_TOL})'); ok = False
    if grip_bad:
        print(f'    [WARN] {grip_bad} 帧 grip 越界 {EE_GRIP_RANGE} m'); ok = False
    return ok


def check_quality_ee(policy, extras):
    print('=' * 60)
    print('推理质量验证 (ee 16D, world [xyz,quat_wxyz,grip]×2)')
    print('=' * 60)
    results = {}

    policy.infer(build_payload(extras))  # warmup

    # 1. 形状 + 每臂 xyz/quat/grip 合法性
    print('\n--- Test 1: Action 形状 + EE 数值合法性 ---')
    r = policy.infer(build_payload(extras))
    actions = r['actions']
    print(f'  Shape: {actions.shape}  dtype: {actions.dtype}  kind: {r.get("action_kind")}')
    if actions.shape[-1] != 16:
        print(f'    [FAIL] ee-mode 期望 action_dim=16, 实际 {actions.shape[-1]}')
        results['shape_range'] = False
    else:
        okL = _check_ee_one_arm(actions, 'L', EE_XYZ_L, EE_QUAT_L, EE_GRIP_L)
        okR = _check_ee_one_arm(actions, 'R', EE_XYZ_R, EE_QUAT_R, EE_GRIP_R)
        results['shape_range'] = okL and okR and actions.std() > 1e-4
    print(f'  → {"PASS" if results.get("shape_range") else "WARN"}')

    # 2. 一致性 (flow-matching 随机初始化 → 本身非确定, 仅 WARN 不 FAIL)
    print('\n--- Test 2: 一致性 (同一输入 x5) ---')
    fixed = build_payload(extras, img=np.random.randint(0, 255, (*RGB_HW, 3), dtype=np.uint8))
    all_a = np.stack([policy.infer(fixed)['actions'] for _ in range(5)])
    pos_std = all_a[:, :, EE_XYZ_IDX].std(axis=0).mean()
    print(f'  xyz 跨次 std: {pos_std*1000:.2f} mm  (flow-matching 随机噪声→非确定, 供参考)')
    results['consistency'] = pos_std < 0.05
    print(f'  → {"PASS" if results["consistency"] else "WARN"} (xyz std < 50mm; flow-matching 偏高属正常)')

    # 3. 敏感性 (不同 ee_pose + state → 输出应不同)
    print('\n--- Test 3: 敏感性 (不同 ee_pose/state) ---')
    img = np.random.randint(0, 255, (*RGB_HW, 3), dtype=np.uint8)
    pa = build_payload(extras, img, np.zeros(14, dtype=np.float32))
    pb = build_payload(
        extras, img, np.full(14, 0.04, dtype=np.float32),
        overrides={
            'ee_pose_left':  _EE_BASE['ee_pose_left'] + np.array([0.1, -0.1, 0.1, 0, 0, 0, 0], np.float32),
            'ee_pose_right': _EE_BASE['ee_pose_right'] + np.array([0.1, 0.1, 0.1, 0, 0, 0, 0], np.float32),
        },
    )
    diff = np.abs(policy.infer(pa)['actions'][:, EE_XYZ_IDX]
                  - policy.infer(pb)['actions'][:, EE_XYZ_IDX]).mean()
    results['sensitivity'] = diff > 1e-3
    print(f'  xyz mean diff: {diff*1000:.3f} mm → {"PASS" if results["sensitivity"] else "FAIL"}')

    # 4. 平滑性 (相邻步 xyz 跳变, m)
    print('\n--- Test 4: 时序平滑性 (xyz, m) ---')
    a = policy.infer(fixed)['actions']
    pos_steps = np.abs(np.diff(a[:, EE_XYZ_IDX], axis=0))
    results['smoothness'] = pos_steps.max() < 0.1
    print(f'  max xyz jump: {pos_steps.max()*1000:.2f} mm  mean: {pos_steps.mean()*1000:.2f} mm')
    print(f'  → {"PASS" if results["smoothness"] else "WARN"} (max < 100mm)')

    _timing_test(policy, build_payload(extras), results)
    return _summarize(results)


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _timing_test(policy, payload, results):
    print('\n--- Test 5: Server timing ---')
    r = policy.infer(payload)
    if 'server_timing' in r:
        st = r['server_timing']
        print(f'  infer_ms: {st.get("infer_ms", "N/A")}')
        results['timing'] = st.get('infer_ms', 999) < 200
    else:
        print('  No server_timing in response')
        results['timing'] = True


def _summarize(results):
    print('\n' + '=' * 60)
    all_pass = all(results.values())
    for k, v in results.items():
        print(f'  {k}: {"PASS" if v else "FAIL"}')
    print(f'\n  总结: {"PASS" if all_pass else "NEEDS REVIEW"}')
    return all_pass


def resolve_action_kind(metadata, mode_flag):
    """按 --mode 覆盖 / metadata.action_kind / action_dim / 默认 joint 解析。"""
    if mode_flag != 'auto':
        return mode_flag
    md = metadata or {}
    kind = md.get('action_kind')
    if kind in ('joint', 'ee'):
        return kind
    if md.get('action_dim') == 16:
        return 'ee'
    return 'joint'


def resolve_extras(metadata, action_kind, with_flag):
    """决定要附带哪些可选 obs_key。

    优先用 metadata.obs_keys 声明的 (与协议一致, 自适配任意 server);
    缺失时退化: --with 显式指定 > ee-mode 默认补 ee_pose。
    """
    md = metadata or {}
    if with_flag:
        return [k for k in with_flag if k in OPTIONAL_SYNTH]
    declared = md.get('obs_keys')
    if declared:
        return [k for k in declared if k in OPTIONAL_SYNTH]
    # metadata 未声明 obs_keys: ee 协议至少需要 ee_pose
    return ['ee_pose_left', 'ee_pose_right'] if action_kind == 'ee' else []


def main():
    parser = argparse.ArgumentParser(description='推理服务器测试 (协议驱动, joint / ee)')
    parser.add_argument('--check', choices=['latency', 'quality', 'all'], default='all')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--mode', choices=['auto', 'joint', 'ee'], default='auto',
                        help='action 协议; auto=按 server metadata 判别')
    parser.add_argument('--with', dest='with_extras', default='',
                        help='强制附带的可选模态, 逗号分隔 (depth_top_head,ee_pose_left,'
                             'ee_pose_right); 留空则按 metadata.obs_keys 自动决定')
    parser.add_argument('--rounds', type=int, default=20, help='延迟测试轮数')
    args = parser.parse_args()

    print(f'Connecting to ws://{args.host}:{args.port}...')
    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = policy.get_server_metadata()
    print(f'Connected. metadata: {metadata}\n')

    action_kind = resolve_action_kind(metadata, args.mode)
    with_flag = [s.strip() for s in args.with_extras.split(',') if s.strip()]
    extras = resolve_extras(metadata, action_kind, with_flag)
    print(f'[test] action_kind={action_kind} (mode={args.mode}) | extras={extras or "—"} '
          f'→ {"16D world EE 校验" if action_kind == "ee" else "14D 关节校验"}\n')

    ok = True
    if args.check in ('latency', 'all'):
        ok &= check_latency(policy, extras, rounds=args.rounds)
        print()
    if args.check in ('quality', 'all'):
        ok &= check_quality_ee(policy, extras) if action_kind == 'ee' else check_quality_joint(policy, extras)

    # WebsocketClientPolicy has no close(); guard so cleanup never masks the result.
    if hasattr(policy, 'close'):
        policy.close()
    return 0 if ok else 1


if __name__ == '__main__':
    exit(main())
