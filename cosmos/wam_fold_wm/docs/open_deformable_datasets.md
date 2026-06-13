# 开源柔性物体操作数据集清单(2026-06-12 调研,已验证可下载性)

> 用途标记:**CT** = 协同训练 Cosmos3 FD 世界模型;**EV** = held-out 评测;**MIX** = 预训练式混合;**AF** = 无动作纯视频。
> 我们自有数据:wam_fold_v1 去重 8,610 eps / 82.3 h(AgileX Piper 双臂 14 维关节,LeRobot v2.1)。

## Tier 1 — 大规模真机布料数据(带动作,杠杆最大)

| 数据集 | 规模(布料部分) | 本体/动作 | 格式/许可 | 链接 | 用途 |
|---|---|---|---|---|---|
| **AgiBot World Beta 叠衣任务**(经 BAAI-DataCube 按任务转 LeRobot v3,免 gate) | **≈24k+ eps**:折短裤 8,233 + 3,057 + 2,063 + 1,792;**叠 T 恤 task_570 = 2,208 ep/174 GB**;挂衣 2,064;叠毛巾 987+400 等 | Genie-1 双臂人形,头部鱼眼+腕部相机 | LeRobot v3;**CC BY-NC-SA**(继承上游) | hf.co/datasets/BAAI-DataCube/AgiBotWorld-Beta_G1_task_570_Fold_the_T-shirt_on_the_field(同 org 浏览其余) | **CT 首选** |
| **lerobot/full_folding**(Unfolding Robotics, 2026-02) | **5,688 eps / 14.1M 帧**,单任务叠 T 恤;另有精选子集 high_quality_folding 1,200 eps | OpenArms 双臂 | LeRobot v3;**Apache-2.0** | hf.co/datasets/lerobot/full_folding | **CT,任务最贴近** |
| **RoboCOIN**(BAAI, 2025-11) | AgileX Cobot Magic 叠毛巾系列:fold_towel_brown **387 ep/18.6 GB** + purple/blue/tray_twice/storage 等;另 R1_Lite fold/hang_clothes | **AgileX 双臂(与我们同厂系)**,关节空间,多相机 | LeRobot 扩展;**Apache-2.0** | hf.co/datasets/RoboCOIN/Agilex_Cobot_Magic_fold_towel_brown | **CT+EV,本体最接近** |
| RoboMIND v1.2/v2.0 | 107k 轨迹中含 **AgileX Cobot Magic 10,629 条**(3 RGB),内含皱毛巾双臂折叠任务 | 多本体 | h5/LeRobot;gated,Apache-2.0 | hf.co/datasets/x-humanoid-robomind/RoboMIND | CT/MIX |
| Galaxea Open-World / G0 | 500+ h 移动双臂(布料占比中等,含叠毛巾) | R1-Lite 双臂+躯干+底盘 | **LeRobot v2.1(与我们同版本)**;gated,CC BY-NC-SA | hf.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset | CT/MIX |
| lerobot/xvla-soft-fold | **1,542 eps / 2.85M 帧** 叠衣(X-VLA 两小时 100% 成功 demo 背后数据) | HF 元数据标 franka(单臂,注意) | LeRobot v3;Apache-2.0 | hf.co/datasets/lerobot/xvla-soft-fold | CT(本体不同) |

## Tier 2 — 小规模真机布料(评测/补充)

- **Piper 原生社区集(零本体差距 held-out!)**:Stone-Chern/piper_towel_yellow_fold(112 ep,**LeRobot v2.1**)、Ishan-Axibo/bimanual_piper_folding(150 ep,v3)、intuitioncore/piper_fold_the_yellow_towel_2/3、mjung11/nc_bi_piper_folding_mj_A(20 ep)——**EV 首选**
- Unitree G1_Dex1_Fold_Towel(灵巧手叠毛巾,Apache)/ Z1_Dual_Dex1_FoldClothes(83 ep)/ unitreeh1_fold_clothes(38 ep)
- TrossenRoboticsCommunity/aloha_fold_tshirt(21 ep,ALOHA 14 维关节)、lerobot/aloha_static_towel(50 ep)
- UTokyo xArm bimanual towel-fold(OXE,70 ep,双臂 14 维 EE,**CC BY 4.0**)
- **Flat'n'Fold**(Glasgow 2024):**1,212 人类 + 887 机器人 demo,44 件衣物 8 类,皱团→展平→折叠全流程**,多视角 RGB-D+点云;Baxter 遥操作 — CT/EV(流程覆盖最全)
- DeformPAM(ICRA 2025,T 恤展开,点云+原语动作+偏好标签)、UniFolding(CoRL 2023)
- villekuosmanen/fold_clothes_dining(40 ep,ARX5)、UMI bimanual cloth folding LeRobot 转换(<1k ep)
- RoboChallenge task_table30v2_fold_the_clothes(2026-04)、BEHAVIOR Robot Suite "Lay clothes out"(98 demos,MIT)

## Tier 3 — 混合大库(布料占比小,MIX 用)

DROID(76k/350h,含少量叠毛巾,CC BY 4.0)、BridgeData V2(60k,含叠布)、RH20T(110k,含折叠,非商用)、OXE 子集(utokyo_xarm_bimanual / berkeley_autolab_ur5 / berkeley_cable_routing 绳缆)、AgiBotWorld Challenge 2026(gated,有世界模型赛道)。

## 仿真数据/基准(带动作)

- **DexGarmentLab**(NeurIPS 2025):Isaac Sim,**2,500+ 衣物、15 个双臂灵巧手任务**(折/挂/抛/穿),单 demo 自动扩增(HALO)— 仿真增广多样性候选
- GarmentLab(NeurIPS 2024,20 任务 9,000+ 资产)、VR-Folding(CVPR 2023,人手 VR 折叠 4D 衣物状态,MIT,可下载)
- FabricFlowNet(SoftGym 双臂折叠,20k 样本随机动作集)、DeformableRavens(布/绳/袋 12 任务,2021)
- ⚠️ **RoboTwin 2.0 没有衣物任务**(尽管是主流双臂仿真基准);SoftGym 需自己生成 demo
- 资产库(无动作):ClothesNet(4,400 衣物网格)、Cloth3D、GarmentCodeData

## 其他柔性物体(绳/面团等)

RoboCook/RoboCraft(真机面团+工具)、**PGND 数据**(RSS 2025:绳/布/毛绒+机器人动作的稀疏视角 RGB-D,**专为柔性动力学模型评测设计** — EV)、PhysTwin(人手交互,非机器人)、PokeFlex(体积形变+力)、Berkeley Cable Routing。

## 无动作视频(视频侧混合,AF)

- **Something-Something v2**:220,847 段,含 "Folding/Unfolding something" 类 — 人手-布料形变动态最佳 AF 源(Qualcomm 自定义许可)
- Ego4D / Ego-Exo4D(3,670 h 第一人称,含叠衣场景)、EPIC-KITCHENS-100(CC BY-NC)
- Verleysen et al. 2020 人类叠衣多视角视频(Sci Data, doi 10.1038/s41597-020-00604-0)
- CLOTH4D(合成 4D 穿衣人体)、Cloth-splatters/folding-meshes(MuJoCo 布料网格轨迹,2026-05)

## 已宣布但未开源(不要指望)

ALOHA Unleashed(ShirtEasy 5,345 + ShirtMessy 3,313 挂衣 demo,**真机数据从未放出**)、SpeedFolding、Figure Helix laundry、π0/π0.5 折叠语料(均闭源)。

## ModelScope 可用性排查(2026-06-12 实测,本机直连验证)

> 工具:`/mnt/pfs/p46h4f/cosmos/.ms-tool/bin/modelscope download --dataset <id> --include "<pattern>" --local_dir <dir>`
> **`--include` 局部下载已实测可用**(RoboCOIN fold_clothes 只拉 meta/ 成功,4.9 MB)。

| 数据集 | ModelScope ID | 只下叠衣部分? | 布料部分体量 | 格式/许可 |
|---|---|---|---|---|
| **AgiBot World Beta(LeRobot v2 转换)⭐首选** | `amap_cvlab/AgiBotWorld-Beta_Lerobot_v2`(7 TB) | ✅ per-task tar.gz,`--include "task_570*"` 等 | **12/13 个布料任务全在**:362=200.4G、570(叠T恤)=188.5G、414=177.1G、599=167.6G、561=110.9G、555=78.4G、444=60.6G、477=55.5G、520=52.5G、509=35.9G、658=31.7G、681=10.9G(仅缺 351);**纯叠衣 ~1.0 TB,全布料 ~1.17 TB** | **LeRobot v2(最贴近我们 v2.1)**/ CC-BY-NC-SA |
| AgiBot World Beta(DataCube LeRobot v3) | `BAAI_DataCube/AgiBotWorld-Beta_G1_task_*` | ✅ 天然按任务分仓 | MS 上有 6 个叠衣仓:362=412G、570=186.6G、561=106.2G、477=53.9G、520=52.1G、509=34.6G + 414 挂衣 171.5G(599 等缺) | LeRobot v3 / **Apache** |
| **RoboCOIN** | `RoboCOIN/<task>` 全组织在 MS | ✅ 天然按任务分仓 | `Cobot_Magic_fold_clothes`=29.7G(**实测 meta:LeRobot v2.1 同版本!AgileX 双臂,584 ep/774k 帧@30fps,front+双腕三相机布局与我们相同**,带 subtask 标注与 EEF 位姿);fold_towel_brown=20G/blue=10.8G/tray_twice=3.8G/blue_tray=1G;fold_short_sleeve_white=0.7G;R1_Lite_fold_clothes=18.7G | LeRobot v2.1 / **Apache** |
| **RoboMIND 2.0** | `X-Humanoid/RoboMIND2.0-Agilex`(13.3 TB)等按本体分仓 | ✅ `--include "data/agilex/fold_clothes/*"`(已确认该目录存在) | fold_clothes 单任务目录(体量待拉 meta 确认) | Apache |
| lerobot/full_folding | `lerobot/full_folding` | 整仓即单任务(5,688 ep),无需筛 | 全量(MS size 字段未索引,HF 显示 14.1M 帧);附 sarm_progress.parquet | LeRobot v3 / Apache |
| lerobot/xvla-soft-fold | `lerobot/xvla-soft-fold` | 整仓即单任务 | 53.6 G | LeRobot v3 / Apache |
| Unitree 系 | `unitreerobotics/G1_Fold_Towel`(10.8G)、`Z1_Dual_Dex1_FoldClothes_Dataset`(6.2G)、`Z1_DualArm_FoldClothes_Dataset`(6.2G)、`lerobot/unitreeh1_fold_clothes`(1G) | 整仓即单任务 | 见左 | Apache |
| Galaxea G0 | `Galaxea/Galaxea-Open-World-Dataset`(5.1 TB) | ⚠️ 文件树 API 要求登录(gated),需 MS 账号 token 后按任务 tar 筛选 | 待授权后确认 | CC-BY-NC-SA |
| **MS 独有新发现(HF 清单之外)** | `HaoranLiCASIA/IROS2025-Agilex-Fold-Towel-All`(1,070G)/`IROS2025-Agilex-Fold-Short`(654G)/`AGX_fold_clothes_challenge_orig`(1,291G);`liyixuan2026/X-VLA-Fold-Clothes-Lerobot`(2,112G)/`cobot_magic_fold_towel`(58G);`hehehaha1215/fold_clothes`(835G);`zdhscdj/casbot_fold_shirt`(109G) | 多为整仓单任务族 | 见左 | 均 Apache(质量待抽查) |

**meta 验货结论(2026-06-12)**:
- ⭐ **IROS2025-Agilex-Fold-Towel-All:格式孪生!** LeRobot **v2.1**、robot=aloha、**1,228 ep/118 万帧@30fps**、相机 key `cam_high/cam_left_wrist/cam_right_wrist` 与 wam_fold_v1 一字不差、state/action 均 [14]。→ 已启动全量下载。
- ⭐ **X-VLA-Fold-Clothes 完整版**:同样 v2.1/aloha/三相机/[14],1,211 ep/235 万帧,但 **fps=50**(接入需时间降采样到 30 或 conditioning_fps=50 超出 Cosmos3 信封,须降采样)。→ 暂缓,等其子集 xvla-soft-fold 质量验证后再决定(2.1T ≈ 4 天带宽)。
- ❌ casbot_fold_shirt / hehehaha1215/fold_clothes / IROS2025-Fold-Short:无标准 meta 目录,且登录态下 tree API 仍报错(疑似需逐仓授权)→ 搁置。

**下载命令模板**(沿用 `wam_fold_policy/setup/download_cosmos3_models.sh` 的绕代理方式):
```bash
unset http_proxy https_proxy; export no_proxy='*' MODELSCOPE_DOMAIN=www.modelscope.cn
MS=/mnt/pfs/p46h4f/cosmos/.ms-tool/bin/modelscope
# 例:只下 AgiBot 叠 T 恤(LeRobot v2,~189G)
"$MS" download --dataset amap_cvlab/AgiBotWorld-Beta_Lerobot_v2 --include "task_570*" --local_dir <dest>
# 例:RoboCOIN 叠衣(整仓 29.7G,v2.1)
"$MS" download --dataset RoboCOIN/Cobot_Magic_fold_clothes --local_dir <dest>
# 例:RoboMIND 只下 AgileX 叠衣任务
"$MS" download --dataset X-Humanoid/RoboMIND2.0-Agilex --include "data/agilex/fold_clothes/*" --local_dir <dest>
```

## 对我们的落地建议

1. **协同训练包(把布料视频量翻 ~4 倍)**:AgiBotWorld 叠衣系列(~24k ep,注意 NC 许可)+ lerobot/full_folding(5.7k,Apache)+ RoboCOIN AgileX 毛巾系列(~1-2k,Apache)+ RoboMIND AgileX 子集。全部 LeRobot 格式,接入 `WamFoldLeRobotDataset` 的跨 rig 域机制(每源一个 domain_id + 独立归一化)即可。
2. **零本体差距 held-out 评测**:Piper 社区集(Stone-Chern 112 ep 还是同版本 v2.1)→ 直接测跨场景泛化;UTokyo xArm / aloha_fold_tshirt 测跨本体;PGND 测动力学保真。
3. **视频侧 AF 混合**(呼应"杠杆在视频侧"):SSv2 折叠类 + Ego4D 叠衣片段,提升形变外观多样性。
4. **许可注意**:AgiBot/Galaxea/RoboMIND 为 gated + CC BY-NC-SA(非商用);lerobot/RoboCOIN/Unitree 系 Apache-2.0 最干净。
