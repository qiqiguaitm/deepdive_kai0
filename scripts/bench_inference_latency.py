#!/usr/bin/env python3
"""
Policy Server 推理延迟基准测试

模拟真实推理请求：发送 3 张 RGB 图像 (224x224) + 14 维关节状态 + prompt，
测量端到端推理延迟（网络 + 模型前向传播 + 后处理）。

kai0 推理要求:
  - inference_rate: 3-4 Hz (每 250-333ms 一次推理)
  - publish_rate: 30 Hz (action chunk 插值发布)
  - 单次推理返回 chunk_size=50 步动作，覆盖 50/30 ≈ 1.67s

所以只要单次推理 < 300ms 就满足在线控制需求。

Usage:
  python3 scripts/bench_inference_latency.py [--host localhost] [--port 8000] [--rounds 20]
"""
import argparse
import time
import numpy as np
import sys
sys.path.insert(0, '/data1/tim/workspace/deepdive_kai0/kai0/packages/openpi-client/src')

from openpi_client import websocket_client_policy, image_tools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=3)
    args = parser.parse_args()

    print(f'Connecting to ws://{args.host}:{args.port} ...')
    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print(f'Connected. Server metadata: {policy.get_server_metadata()}')

    # 构造模拟 payload (与真实推理一致)
    dummy_img_224 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_state = np.random.randn(14).astype(np.float32)

    payload = {
        'images': {
            'top_head': dummy_img_224,
            'hand_left': dummy_img_224,
            'hand_right': dummy_img_224,
        },
        'state': dummy_state,
        'prompt': 'fold the cloth',
    }

    # Warmup
    print(f'\nWarmup ({args.warmup} rounds)...')
    for i in range(args.warmup):
        t0 = time.monotonic()
        result = policy.infer(payload)
        t1 = time.monotonic()
        action_shape = result['actions'].shape if 'actions' in result else 'N/A'
        print(f'  warmup {i+1}: {(t1-t0)*1000:.0f}ms  action_shape={action_shape}')

    # Benchmark
    print(f'\nBenchmark ({args.rounds} rounds)...')
    latencies = []
    for i in range(args.rounds):
        # 每轮用不同随机数据模拟不同观测
        payload['state'] = np.random.randn(14).astype(np.float32)
        payload['images']['top_head'] = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        t0 = time.monotonic()
        result = policy.infer(payload)
        t1 = time.monotonic()

        lat = (t1 - t0) * 1000
        latencies.append(lat)
        print(f'  round {i+1:2d}: {lat:.0f}ms')

    latencies = np.array(latencies)
    print(f'\n{"="*50}')
    print(f'推理延迟统计 ({args.rounds} rounds)')
    print(f'{"="*50}')
    print(f'  avg:  {latencies.mean():.0f} ms')
    print(f'  std:  {latencies.std():.0f} ms')
    print(f'  min:  {latencies.min():.0f} ms')
    print(f'  p50:  {np.median(latencies):.0f} ms')
    print(f'  p95:  {np.percentile(latencies, 95):.0f} ms')
    print(f'  p99:  {np.percentile(latencies, 99):.0f} ms')
    print(f'  max:  {latencies.max():.0f} ms')
    print(f'  throughput: {1000/latencies.mean():.1f} infer/s')

    print(f'\n在线控制要求: < 300ms/次 (inference_rate=3Hz)')
    if latencies.mean() < 300:
        print(f'结论: PASS - 满足在线控制需求 ({latencies.mean():.0f}ms avg)')
    elif latencies.mean() < 500:
        print(f'结论: MARGINAL - 勉强可用，建议降低 inference_rate ({latencies.mean():.0f}ms avg)')
    else:
        print(f'结论: FAIL - 推理过慢，需要优化 ({latencies.mean():.0f}ms avg)')

    policy.close()


if __name__ == '__main__':
    main()
