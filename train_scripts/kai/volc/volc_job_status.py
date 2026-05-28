#!/usr/bin/env python3
"""经 gf0 查火山 ML Platform 任务状态 (用 volcengine.base.Service 直调 OpenAPI, 绕开 SDK 5.0.27 反序列化 bug).

需要 env: VOLC_AK / VOLC_SK。
见 docs/deployment/training_ops/submission/gf0_control_plane.md §5.6.c.5。
"""
import json, os
from volcengine.ApiInfo import ApiInfo
from volcengine.Credentials import Credentials
from volcengine.ServiceInfo import ServiceInfo
from volcengine.base.Service import Service


def get_svc(region):
    si = ServiceInfo('open.volcengineapi.com', {'Accept': 'application/json'},
                     Credentials(os.environ['VOLC_AK'], os.environ['VOLC_SK'], 'ml_platform', region), 5, 5)
    return Service(si, {
        'ListResourceQueues': ApiInfo('POST', '/', {'Action': 'ListResourceQueues', 'Version': '2024-07-01'}, {}, {}),
        'ListJobs':           ApiInfo('POST', '/', {'Action': 'ListJobs',           'Version': '2024-07-01'}, {}, {}),
        'GetJob':             ApiInfo('POST', '/', {'Action': 'GetJob',             'Version': '2024-07-01'}, {}, {}),
    })


if __name__ == "__main__":
    # 示例: 查 Robot-North-H20 当前 running
    svc = get_svc('cn-beijing')
    r = svc.json('ListJobs', {}, json.dumps(
        {"ResourceQueueId": "q-20260516104642-khch9", "PageSize": 30, "State": "Running"}).encode())
    for j in json.loads(r)['Result'].get('List', []):
        print(j['Name'], j['Status']['State'], j['CreateTime'])
