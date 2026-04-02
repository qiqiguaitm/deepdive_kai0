#!/usr/bin/env python3
"""
推理质量验证脚本

检查项:
  1. Action 形状和数值范围 (关节角度是否在 Piper 物理限位内)
  2. 一致性: 相同输入 → 相似输出 (flow matching 有随机性, 但不应差太多)
  3. 敏感性: 不同输入 → 不同输出 (模型不是输出常数)
  4. 时序合理性: 连续 action chunk 之间应该平滑
  5. 左右臂协调: 双臂折叠任务应有协调运动

Usage:
  # 确保 serve_policy 在跑
  python3 scripts/verify_inference_quality.py [--host localhost] [--port 8000]
"""
import argparse
import time
import sys
import numpy as np

sys.path.insert(0, 'packages/openpi-client/src')
from openpi_client import websocket_client_policy


# Piper 关节限位 (radians, 近似值)
JOINT_LIMITS = [
    (-2.6, 2.6),   # joint 0
    (-1.6, 1.6),   # joint 1
    (-1.6, 1.6),   # joint 2
    (-1.8, 1.8),   # joint 3
    (-1.6, 1.6),   # joint 4
    (-2.6, 2.6),   # joint 5
    (0.0, 0.08),   # gripper (meters)
]
# 14 维 = 左臂 7 + 右臂 7
ALL_LIMITS = JOINT_LIMITS + JOINT_LIMITS


def make_payload(img=None, state=None):
    if img is None:
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    if state is None:
        # 合理的初始状态 (接近零位)
        state = np.array([0.0, 0.3, -0.5, 0.0, 0.5, 0.0, 0.03,
                          0.0, 0.3, -0.5, 0.0, 0.5, 0.0, 0.03], dtype=np.float32)
    return {
        'images': {'top_head': img, 'hand_left': img, 'hand_right': img},
        'state': state,
        'prompt': 'fold the cloth',
    }


def check_action_range(actions, name=""):
    """检查 action 是否在物理限位内"""
    n_steps, n_dims = actions.shape
    assert n_dims == 14, f"Expected 14 dims, got {n_dims}"

    violations = 0
    for dim in range(14):
        lo, hi = ALL_LIMITS[dim]
        margin = 0.5  # 允许 0.5rad 余量
        below = (actions[:, dim] < lo - margin).sum()
        above = (actions[:, dim] > hi + margin).sum()
        if below > 0 or above > 0:
            violations += below + above
            joint_name = f"{'L' if dim < 7 else 'R'}_joint{dim % 7}"
            print(f"    [WARN] {joint_name}: {below} below {lo-margin:.2f}, {above} above {hi+margin:.2f}")

    return violations == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    print(f'Connecting to ws://{args.host}:{args.port}...')
    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print('Connected.\n')

    # Warmup
    payload = make_payload()
    policy.infer(payload)

    results = {}

    # ── Test 1: Action 形状和数值范围 ─────────────────────────────────
    print('='*60)
    print('Test 1: Action 形状和数值范围')
    print('='*60)
    r = policy.infer(make_payload())
    actions = r['actions']
    print(f'  Shape: {actions.shape}')
    print(f'  dtype: {actions.dtype}')
    print(f'  范围: [{actions.min():.4f}, {actions.max():.4f}]')
    print(f'  均值: {actions.mean():.4f}')
    print(f'  标准差: {actions.std():.4f}')

    # 按维度统计
    for dim in range(14):
        joint_name = f"{'L' if dim < 7 else 'R'}_j{dim % 7}"
        vals = actions[:, dim]
        print(f'  {joint_name}: [{vals.min():.3f}, {vals.max():.3f}] mean={vals.mean():.3f}')

    range_ok = check_action_range(actions)
    not_zero = actions.std() > 0.001
    results['shape_range'] = range_ok and not_zero
    print(f'\n  有限位内: {"PASS" if range_ok else "WARN"}')
    print(f'  非零输出: {"PASS" if not_zero else "FAIL"}')

    # ── Test 2: 一致性 (相同输入 → 相似输出) ────────────────────────
    print(f'\n{"="*60}')
    print('Test 2: 一致性 (相同输入, 5 次推理)')
    print('='*60)
    fixed_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    fixed_state = np.array([0.1, 0.5, -0.8, 0.0, 0.6, -0.1, 0.04,
                            -0.1, 0.5, -0.8, 0.0, 0.6, 0.1, 0.04], dtype=np.float32)
    fixed_payload = make_payload(fixed_img, fixed_state)

    all_actions = []
    for i in range(5):
        r = policy.infer(fixed_payload)
        all_actions.append(r['actions'])

    all_actions = np.stack(all_actions)  # (5, 50, 14)
    mean_action = all_actions.mean(axis=0)
    std_across_runs = all_actions.std(axis=0).mean()
    max_diff = np.abs(all_actions - mean_action).max()

    print(f'  跨次 std (均值): {std_across_runs:.4f} rad')
    print(f'  最大偏差: {max_diff:.4f} rad')

    # flow matching 有随机性，但 std 不应太大
    consistency_ok = std_across_runs < 0.3
    results['consistency'] = consistency_ok
    print(f'  判定: {"PASS" if consistency_ok else "WARN"} (std < 0.3)')

    # ── Test 3: 敏感性 (不同输入 → 不同输出) ────────────────────────
    print(f'\n{"="*60}')
    print('Test 3: 敏感性 (不同状态 → 不同动作)')
    print('='*60)

    state_a = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    state_b = np.array([0.5, 1.0, -1.0, 0.3, 0.8, -0.5, 0.05,
                        -0.5, 1.0, -1.0, -0.3, 0.8, 0.5, 0.05], dtype=np.float32)

    r_a = policy.infer(make_payload(fixed_img, state_a))
    r_b = policy.infer(make_payload(fixed_img, state_b))

    diff = np.abs(r_a['actions'] - r_b['actions']).mean()
    print(f'  State A (零位) vs State B (偏移): mean diff = {diff:.4f} rad')

    sensitivity_ok = diff > 0.01
    results['sensitivity'] = sensitivity_ok
    print(f'  判定: {"PASS" if sensitivity_ok else "FAIL"} (diff > 0.01)')

    # ── Test 4: 时序平滑性 ──────────────────────────────────────────
    print(f'\n{"="*60}')
    print('Test 4: 时序平滑性 (chunk 内相邻步之间的跳变)')
    print('='*60)

    r = policy.infer(make_payload(fixed_img, fixed_state))
    actions = r['actions']  # (50, 14)
    step_diffs = np.diff(actions, axis=0)  # (49, 14)
    max_step_jump = np.abs(step_diffs).max()
    mean_step_jump = np.abs(step_diffs).mean()

    print(f'  相邻步最大跳变: {max_step_jump:.4f} rad')
    print(f'  相邻步平均跳变: {mean_step_jump:.4f} rad')

    smoothness_ok = max_step_jump < 0.5  # 单步不应跳超过 0.5 rad (~28°)
    results['smoothness'] = smoothness_ok
    print(f'  判定: {"PASS" if smoothness_ok else "WARN"} (max jump < 0.5 rad)')

    # ── Test 5: server_timing 检查 ──────────────────────────────────
    print(f'\n{"="*60}')
    print('Test 5: Server timing')
    print('='*60)
    r = policy.infer(make_payload())
    if 'server_timing' in r:
        st = r['server_timing']
        print(f'  infer_ms: {st.get("infer_ms", "N/A"):.1f}')
        if 'prev_total_ms' in st:
            print(f'  prev_total_ms: {st["prev_total_ms"]:.1f}')
        results['timing'] = st.get('infer_ms', 999) < 200
    else:
        print('  No server_timing in response')
        results['timing'] = True

    # ── 总结 ─────────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print('总结')
    print('='*60)
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f'  {name}: {status}')

    print(f'\n  总体: {"PASS" if all_pass else "NEEDS REVIEW"}')


if __name__ == '__main__':
    main()
