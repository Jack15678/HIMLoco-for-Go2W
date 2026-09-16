# S10 直接速度命令：model_1500 +500 轮最终评估

**该方向改善了持续转向的平均跟踪误差，但没有消除振荡，也不是原 model_1500 的全面替代。** 双引擎 8 个平地 yaw 阶段的 RMSE 全部下降，8 cm 障碍倒车明显改善；同时，8 cm 上台阶出现一个提前终止初态，Gym 直行航向和末次停车、MuJoCo 倒车尾段速度有退化。准确新增 500 轮后已停止，没有调参、追加训练或合并停车方向。

## 1. 改变的是命令生成方式

**策略原本已经输入 vx、vy、yaw 角速度。** 当前帧第 6:9 列就是这三个量的缩放值。原训练的 `heading_command=True` 会采样目标航向，在 reset 和每步 callback 中计算 `clip(0.5 * wrap(target_heading - heading), -2, 2)`，把结果作为 yaw 速度输入策略；接近目标航向时，这个速度会减小。

本方向唯一配方改动为 **S10 `commands.heading_command=False`**，复用现有 yaw 速度直接采样路径：范围仍为 `[-1, 1] rad/s`，在原 10 秒重采样或 episode reset 之前保持，不随航向误差缩小。reset 仍重新采样，但不再用航向误差覆盖新 yaw。vx/vy 范围、原 20% 高速组、线速度小命令归零、课程和所有奖励均保留。

网络输入维度、16 维动作、6 帧历史、checkpoint 形状及优化计算未改。PD 2.5 ms × decimation 8，策略和历史周期 20 ms，delay_stride 2；腿 80/2、轮 0/.6、腿动作尺度 .125/.25/.25 和轮速尺度 5 均未改。没有手动重置 std 或调整学习率。

结果与“训练命令分布影响持续转向能力”的假设相符，但**不能据此认定 heading 生成方式是全部转向振荡的唯一原因**。对照是既有 model_1500；本次包含继续学习 500 轮和课程重建，并没有额外做同预算、保留 heading 模式的续训对照。

## 2. 来源、完整恢复与准确预算

| 项目 | 结果 |
| --- | --- |
| 基础 main | `e232fb97bbd9ae9a5648fe7e55587ca8c27d129b` |
| 本方向分支 | `JackNcodex/s10-direct-velocity-1500-plus500` |
| 固定训练/预检 SHA | **`fb5744ebba2d9b00d3e6b77722f4e82438edf6bf`** |
| 输入 checkpoint 原训练 SHA | `85c786f90f4774a90e2cb6b883fc3d5c7a051f0a` |
| 固定评估 SHA | **`49500212be70ffd99a67c34d2bfdc93d3fd132ac`** |
| 样本设置 | seed 1，4096 环境 × 48 步/轮 |
| 迭代 | **1500 → 2000，新增 500** |
| 环境步 | **294,912,000 → 393,216,000，新增 98,304,000** |
| PPO / HIM Adam 步数 | **两套均 30,000 → 40,000，各新增 10,000** |
| checkpoint / 日志 | 11 个：1500、1550、…、2000；连续 500 条迭代日志 |
| 两套初始 LR | 从 checkpoint 恢复，均 `.000170859375` |
| 两套最终 LR | 均 `.000170859375`，期间按原 adaptive 规则变化 |
| 数值核验 | 全部 11 个模型与优化器状态有限，计数正确 |

CPU 和真实 Gym 均调用生产 `HIMOnPolicyRunner.load(load_optimizer=True)`，逐元素核对 **全部 30 个模型状态张量及两套完整 Adam 状态**，包括 actor、critic、HIM encoder/target/prototypes、std、Adam 动量与步数。训练 runner 在相同核验通过后才执行一次 **`learn(500)`**，没有把累计 2000 当新增预算。

真实 Gym 零更新检查包含 512 环境 × 48 步，共 24,576 个样本、29 次真实 reset。actor/critic 保存观测误差、reset 历史误差均为 0；ratio 最大偏差 `2.77e-5`，clip fraction 为 0，KL 与数值 floor 的差为 0。PPO/HIM update 和 optimizer step 调用数均为 0。纯 yaw ±.8、零 yaw、零命令、复合 vx/vy/yaw 在航向 −3/−1/0/1/3 rad 下均保持，实际观测命令误差为 0，历史每次只推进一帧。

原 checkpoint **不含模拟器、RNG 和完整课程状态**。本次按 seed 1 新建环境、随机化并重建课程，没有物理状态无缝续训。保存配置与原配方核对后，仅 `heading_command` 不同。

停车任务全部 GPU 工作结束后才启动本方向。训练容器于北京时间 **08:18:25–09:04:21** 运行（包含 Gym gate 和环境初始化），exit 0、无 OOM、无重启；runner 新增训练计时 **2662.08 秒**。评估于 **09:23:28–10:10:53** 完成，同样 exit 0、无 OOM、无重启。服务器源码一直固定、干净、只读挂载，旧训练与评估产物未覆盖；结束后已确认 GPU 空闲。

末 20 轮：回合长度 19.324 秒，地形等级 2.536，KL 均值 .015816，clip fraction .19138，PPO/HIM LR 均值 .000249882。LR 的迭代末值范围为 `1e-5–.000384434`；最终 std 范围为 hipx .3270–.3483、hipy .2118–.2197、knee .2449–.2630、wheel .7169–.7675。已记录 KL/clip fraction、小批次 LR、两套 LR、奖励分量和课程统计，未改变优化规则。训练回报不作为不同命令分布之间的行为优劣依据。

[CPU 恢复核验](../artifacts/s10-direct-velocity-20260916/cpu_restore.json) · [真实 Gym 零更新核验](../artifacts/s10-direct-velocity-20260916/zero_gate.json) · [训练恢复核验](../artifacts/s10-direct-velocity-20260916/training/resume_verification.json) · [最终预算和全部 checkpoint 核验](../artifacts/s10-direct-velocity-20260916/training/training_verification.json) · [训练曲线](../artifacts/s10-direct-velocity-20260916/training/training_curves.png)

## 3. 固定评估与证据范围

完全复用 [s10-follow-terrain-v1 协议](S10_OBSFIX_EVALUATION_PROTOCOL_20260916.md)：10 秒 Gym 17 初态及 MuJoCo 名义站姿；90 秒 Gym/MuJoCo 平地；Gym 10 类地形 × 3 固定初态，各最长 60 秒。对照为 [model_1500 已有标准评估](S10_OBSFIX_1500_REVIEW_20260916.md)，没有重跑或更换基线。

两条平地均完成 90 秒；30 条地形中 **29 条完成 60 秒，1 条按原倾角门槛提前结束**。32 条扩展轨迹共 **776,103 个 PD 样本**通过有限值、连续时序、指令时间表、下发/观测一致性检查。12 段扩展视频与对应轨迹时长匹配，2 段站姿视频均为 10 秒。已检查两引擎曲线、10 类实际地形画面、8 cm 推进画面及提前终止轨迹。

以下 `原 → 新` 均指 **model_1500 → 本方向续训 model_2000**。均值取阶段末 5 秒，RMSE 取完整阶段。首次 90% 是第一次沿目标方向越过目标幅度的 90%；稳定时间另要求进入 ±10% 误差带并连续保持至少 1 秒。

## 4. 持续转向：误差改善，稳定仍未达标

### 左右阶段均值与 RMSE

速度单位 rad/s。复合阶段同时要求 vx=.6 m/s。

| 引擎 | yaw 目标 | 尾段 yaw 均值：原 → 新 | 整段 yaw RMSE：原 → 新 | RMSE 降幅 |
| --- | --- | --- | --- | ---: |
| Gym | +.8，纯转向 | +.5722 → +.6750 | .2646 → .1684 | 36.3% |
| Gym | −.8，纯转向 | −.6334 → −.7166 | .2294 → .1315 | 42.7% |
| Gym | +.6，复合 | +.4817 → +.5026 | .1670 → .1409 | 15.6% |
| Gym | −.6，复合 | −.4387 → −.4873 | .2058 → .1543 | 25.1% |
| MuJoCo | +.8，纯转向 | +.7171 → +.8128 | .1733 → .1061 | 38.8% |
| MuJoCo | −.8，纯转向 | −.7642 → −.8124 | .1832 → .1160 | 36.7% |
| MuJoCo | +.6，复合 | +.5174 → +.5571 | .1572 → .1031 | 34.4% |
| MuJoCo | −.6，复合 | −.5594 → −.5618 | .1370 → .1055 | 23.0% |

Gym 纯 yaw 的尾段线速度夹带也减小：正转 vx −.0261→−.0141，负转 +.0491→+.0291 m/s，但仍有横向运动。复合段的 vx 均值，Gym 正/负转分别为 .5391→.5279、.6238→.5678；MuJoCo 为 .6018→.5940、.7069→.6385 m/s，需与 yaw 改善分开看。

### 振荡幅度、响应和稳定时间

振荡峰峰值为尾段 `max−min`，去均值 RMS 为该尾段标准差，单位均 rad/s；排除切换瞬间后仍可见明显振荡。完整阶段及尾段统计都保存在对照 JSON 中。

| 引擎 / 目标 | 尾段峰峰值：原 → 新 | 尾段去均值 RMS：原 → 新 | 首次 90% 时间：原 → 新（秒） |
| --- | --- | --- | --- |
| Gym +.8 | .6379 → .6397 | .1329 → .1020 | .2725 → .3775 |
| Gym −.8 | .7683 → .5123 | .1280 → .0724 | .6275 → .2075 |
| Gym 复合 +.6 | .6666 → .5734 | .1041 → .0840 | .0750 → .1925 |
| Gym 复合 −.6 | .8351 → .6280 | .1231 → .1079 | .2000 → .0575 |
| MuJoCo +.8 | .7554 → .5799 | .1442 → .0993 | .2500 → .1650 |
| MuJoCo −.8 | .9706 → .4810 | .1547 → .0994 | .2300 → .1125 |
| MuJoCo 复合 +.6 | .5269 → .4592 | .1075 → .0777 | .1800 → .1125 |
| MuJoCo 复合 −.6 | .6721 → .5692 | .1124 → .0910 | .4150 → .2000 |

**两模型、两引擎的全部 8 个 yaw 阶段均未达到 ±10% 连续 1 秒稳定带。** 新模型不是稳定转向通过。8 个去均值 RMS 均下降，7 个峰峰值下降；Gym 正向纯 yaw 峰峰值略增。Gym 两个正向阶段的首次 90% 反而更慢，其他 6 个阶段更快。

多地形中，在两模型均进入转向阶段的 **29 条同初态轨迹**上，正/负 .6 yaw 的平均 RMSE 分别为 **.3870→.2534、.3819→.2537**；阶段均值为 **+.2535→+.3873、−.2655→−.3886**。这 58 个阶段同样均未达到 1 秒稳定带。另一个新模型初态已提前终止，未进入两段转向，不将其作为改善样本；原基线全部 30 条都能进入转向。

### 展开航向与 XY 轨迹

![相同 90 秒协议的双引擎对照](../artifacts/s10-direct-velocity-20260916/direct_vs_1500.png)

图中黑虚线为速度命令或 yaw 命令积分。**Gym 直行航向退化不能被转向 RMSE 的改善掩盖**：15–25 秒 vx=1、yaw=0 时，累计偏航由 +.1372 变为 **−.4057 rad**，尾段 yaw 均值由 +.0163 变为 **−.0613 rad/s**；25 秒世界 y 由 +.196 变为 **−2.004 m**。

完整 90 秒最终展开航向：Gym **+.0275→−.9413 rad**，MuJoCo **−.9799→−.1935 rad**。终点 XY 分别为 Gym (7.794,3.106)→(9.310,2.487) m，MuJoCo (9.854,2.492)→(8.008,1.126) m。XY 用于展示相同速度时间表下的轨迹差异，协议没有另设位置目标或重置拼接。

## 5. 直行和停车是否退化

### 直行三段

| 引擎 / vx 目标 | 尾段 vx：原 → 新（m/s） | 整段 RMSE：原 → 新 | 1 秒稳定带时间：原 → 新（秒） |
| --- | --- | --- | --- |
| Gym +.5 | .4670 → .4462 | .0500 → .0649 | 1.170 → 未达 |
| Gym +1 | .9154 → .9290 | .0939 → .0838 | .200 → .535 |
| Gym −.5 | −.4447 → −.4947 | .1748 → .1502 | 未达 → .3725 |
| MuJoCo +.5 | .4953 → .4669 | .0359 → .0454 | .1975 → .220 |
| MuJoCo +1 | 1.0746 → 1.0175 | .0784 → .0427 | .1375 → .1675 |
| MuJoCo −.5 | −.5116 → −.5866 | .1907 → .1835 | .3925 → 未达 |

Gym .5 m/s 阶段和 MuJoCo .5 m/s 阶段平均跟踪变差。MuJoCo 倒车虽整段 RMSE 略降，但尾段超速且失去稳定带，不能单凭 RMSE 认定改善。Gym 1 m/s 的速度误差减小，同时存在上一节的航向退化。

### 站姿及停止段位移

| 场景 | Gym：原 → 新（cm） | MuJoCo：原 → 新（cm） |
| --- | --- | --- |
| 10 秒独立站姿，名义初态 | 1.281 → .539 | 8.516 → 4.115 |
| 35–40 秒停车 | 5.631 → 4.468 | 13.318 → 7.560 |
| 80–90 秒停车 | **3.916 → 6.557** | 18.634 → 7.071 |

Gym 17/17 和 MuJoCo 站姿均通过原姿态/接触门槛。Gym 17 个初态位移范围由 .712–1.938 cm 变为 .303–1.984 cm，名义初态改善不等于所有扰动初态都改善。

Gym 最后停车段总位移增大，但最后 5 秒 vx 仍接近 0（+.000083→−.000009 m/s）；总位移包含切入停车时的制动过程。MuJoCo 独立站姿尾段 vx −.01656→−.00414，末次停车尾段 vx −.02437→−.01492 m/s，仍有倒退漂移。本方向没有加入停车采样或奖励补丁。

## 6. 多地形：转向改善伴随 8 cm 上台阶退化

前行最大 x 只统计前 35 秒；提前结束的轨迹只统计实际观测窗口。表内为三个固定初态的范围，RMSE 单位 m/s。

| 地形 | 新模型满 60 秒 | 前行最大 x：原 → 新（m） | 倒车 vx RMSE：原 → 新 |
| --- | ---: | --- | --- |
| 粗糙 ±1 cm | 3/3 | 12.26–12.31 → 11.63–11.77 | .110–.114 → .090–.092 |
| 粗糙 ±2 cm | 3/3 | 11.73–12.03 → 11.26–11.58 | .126–.127 → .101–.121 |
| 上坡 8° | 3/3 | 13.04–13.21 → 13.75–13.80 | .130–.131 → .099–.103 |
| 下坡 8° | 3/3 | 11.17–11.58 → 11.46–11.53 | .098–.099 → .104–.107 |
| 上台阶 5 cm | 3/3 | 11.99–12.07 → 12.41–12.42 | .105–.109 → .094–.096 |
| **上台阶 8 cm** | **2/3** | **11.81–11.90 → 1.00–7.60（含终止初态）** | .108–.111 → .092–.114（仅 2 条到达倒车） |
| 下台阶 5 cm | 3/3 | 11.14–11.27 → 10.80–10.86 | .115–.120 → .101–.106 |
| 下台阶 8 cm | 3/3 | 10.46–10.65 → 10.04–10.21 | .438–.440 → .415–.445 |
| 障碍 5 cm | 3/3 | 8.81–11.30 → 11.55–11.66 | .129–.140 → .120–.121 |
| 障碍 8 cm | 3/3 | 8.36–10.59 → 9.09–9.18 | .672–.780 → .123–.347 |

### 8 cm 上台阶明确退化

- 原 model_1500 三个初态在 8.305–8.388 秒进入目标区，20 秒机身 x=3.62–3.65 m，低速阶段尾段 vx=.268–.276 m/s，均能持续并通过台阶。
- 新模型三个初态首次承载车轮进入目标网格为 **9.358、18.770、14.353 秒**；20 秒机身 x 仅 **1.765、.983、.906 m**，低速阶段尾段 vx=.121、.077、.058 m/s。
- **种子 2 在 20.2575 秒因倾角 45.033° 超过门槛而结束。** 底盘接触力最大值为 0，相对地面高度仍大于 .383 m，因此本次终止原因是倾角，不是已经发生底盘碰撞。其承载车轮接触跨度仅 x=1.001–1.248 m；曲线显示长期停在首阶附近并发生侧向偏移。
- 另两个初态完成 60 秒且车轮覆盖到末阶区；截至 35 秒，机身最大 x 仅 7.596/4.418 m。不能用这两个存活初态掩盖第三个失败，或将“触及末阶”直接等同完整速度跟随。

[提前终止初态完整曲线](../artifacts/s10-direct-velocity-20260916/assessments/checkpoint_2000/stairs_up_8cm/trial_2/following.png) · [该初态原始轨迹](../artifacts/s10-direct-velocity-20260916/assessments/checkpoint_2000/stairs_up_8cm/trial_2/trace.npz)

### 8 cm 障碍倒车改善，低速进入仍有代价

35–50 秒目标 vx=−.6 m/s：

| 固定种子 | 整段 vx RMSE：原 → 新 | 尾段 vx：原 → 新 | 新模型首次 1 秒稳定带（秒） |
| ---: | --- | --- | ---: |
| 0 | .7800 → .3470 | −.0208 → −.2560 | 1.3925 |
| 1 | .6721 → .1335 | −.0575 → −.5442 | 1.5525 |
| 2 | .7462 → .1233 | −.0329 → −.5482 | 1.5350 |

原三个初态均未达到持续稳定带；新模型均曾达到，但种子 0 后段又受阻，尾段 −.256 仍明显不足。倒车中的正向速度峰值由 1.83–1.87 m/s 降到 .420–1.336 m/s，尚未完全消失。

低速进入则更迟：三个初态承载车轮首次进入目标区从 **8.36–8.54 秒延后到 15.59–16.42 秒**。新模型前行最大 x=9.09–9.18 m，车轮覆盖到 x≈9.30 m；原两个较好的初态能推进更远。改善主要在倒车跟踪，不能概括为所有障碍能力都提高。5 cm 障碍的三个初态低速推进和倒车均更一致。

下 8 cm 台阶的倒车依然弱，三个尾段 vx 仅 −.160、−.306、−.241 m/s；即使之前曾进入 1 秒稳定带，整段 RMSE 仍为 .415–.445。

![8 cm 场景固定种子 0 在 20/30/35 秒的真实 Gym 画面](../artifacts/s10-direct-velocity-20260916/assessments/checkpoint_2000/terrain_8cm_progress.jpg)

## 7. 视频、checkpoint 和可复核证据

所有多地形代表视频固定种子 0，保留完整可用轨迹、真实 Gym 地形、世界方向固定的跟随相机及 cmd/actual 叠加。上台阶失败发生在种子 2，按原协议没有改选代表视频；其完整曲线和轨迹在上节单独列出。站姿 Gym 视频沿用保存轨迹的 MuJoCo 可视化来源，扩展地形视频来自真实 Gym 相机。

| 内容 | 视频 |
| --- | --- |
| 10 秒站姿 | [Gym](../artifacts/s10-direct-velocity-20260916/assessments/checkpoint_2000/stand_run/standing/gym_standing.mp4) · [MuJoCo](../artifacts/s10-direct-velocity-20260916/assessments/checkpoint_2000/stand_run/standing/mujoco_standing.mp4) |
| 90 秒平地 | [Gym](../artifacts/s10-direct-velocity-20260916/assessments/checkpoint_2000/flat_gym/representative.mp4) · [MuJoCo](../artifacts/s10-direct-velocity-20260916/assessments/checkpoint_2000/flat_mujoco/representative.mp4) |
| 粗糙地 | [±1 cm](../artifacts/s10-direct-velocity-20260916/assessments/checkpoint_2000/rough_1cm/representative.mp4) · [±2 cm](../artifacts/s10-direct-velocity-20260916/assessments/checkpoint_2000/rough_2cm/representative.mp4) |
| 坡道 | [上坡 8°](../artifacts/s10-direct-velocity-20260916/assessments/checkpoint_2000/slope_up_8deg/representative.mp4) · [下坡 8°](../artifacts/s10-direct-velocity-20260916/assessments/checkpoint_2000/slope_down_8deg/representative.mp4) |
| 上台阶 | [5 cm](../artifacts/s10-direct-velocity-20260916/assessments/checkpoint_2000/stairs_up_5cm/representative.mp4) · [8 cm](../artifacts/s10-direct-velocity-20260916/assessments/checkpoint_2000/stairs_up_8cm/representative.mp4) |
| 下台阶 | [5 cm](../artifacts/s10-direct-velocity-20260916/assessments/checkpoint_2000/stairs_down_5cm/representative.mp4) · [8 cm](../artifacts/s10-direct-velocity-20260916/assessments/checkpoint_2000/stairs_down_8cm/representative.mp4) |
| 障碍 | [5 cm](../artifacts/s10-direct-velocity-20260916/assessments/checkpoint_2000/obstacles_5cm/representative.mp4) · [8 cm](../artifacts/s10-direct-velocity-20260916/assessments/checkpoint_2000/obstacles_8cm/representative.mp4) |

- [新 model_2000.pt](../artifacts/s10-direct-velocity-20260916/training/model_2000.pt) · [完整训练配置](../artifacts/s10-direct-velocity-20260916/training/config.json) · [完整启动命令](../artifacts/s10-direct-velocity-20260916/start_training.ps1)
- [原始评估汇总](../artifacts/s10-direct-velocity-20260916/assessments/checkpoint_2000/assessment_complete.json) · [全量行为对照和振荡统计](../artifacts/s10-direct-velocity-20260916/behavior_comparison.json)
- [轨迹与扩展视频核验](../artifacts/s10-direct-velocity-20260916/assessments/checkpoint_2000/verification.json) · [站姿视频核验](../artifacts/s10-direct-velocity-20260916/assessments/checkpoint_2000/standing_video_verification.json)
- [10 类地形实际画面](../artifacts/s10-direct-velocity-20260916/assessments/checkpoint_2000/terrain_contact_sheet.jpg) · [Gym 单独跟随曲线](../artifacts/s10-direct-velocity-20260916/assessments/checkpoint_2000/flat_gym/trial_0/following.png) · [MuJoCo 单独跟随曲线](../artifacts/s10-direct-velocity-20260916/assessments/checkpoint_2000/flat_mujoco/following.png)
- [训练容器最终状态](../artifacts/s10-direct-velocity-20260916/inspect_training_final.json) · [评估容器最终状态](../artifacts/s10-direct-velocity-20260916/assessments/inspect_final.json)

输入完整 checkpoint（只读）：

- 本地：`D:/Desktop/Code/HIMLoco-for-Go2W/artifacts/s10-obsfix-scratch2000-20260916/model_1500.pt`
- 服务器：`/data/HIMLoco-S10-resume-git-20260915/artifacts/s10-obsfix-scratch2000-20260916/S10_HIM/Sep15_16-39-52_s10_obsfix_scratch2000_20260916/model_1500.pt`

本方向输出 checkpoint：

- 本地：`C:/Users/Lenovo/.codex/worktrees/0381/HIMLoco-for-Go2W/artifacts/s10-direct-velocity-20260916/training/model_2000.pt`
- 服务器：`/data/HIMLoco-S10-direct-velocity-git-20260916/artifacts/s10-direct-velocity-20260916/training/model_2000.pt`

本方向评估服务器目录：`/data/HIMLoco-S10-eval-git-20260916/artifacts/s10-direct-velocity-assessments/checkpoint_2000`。完整模型、日志、轨迹和视频保留在忽略目录，未加入源码提交。

**结论：持续 yaw 分布值得与部署需求保持一致，本次数据支持它对平均跟踪和部分振荡幅度有帮助；但稳定转向仍未解决，且伴随明确越障/直行代价。本方向按既定预算结束，保留独立结果，不与停车改动合并，也不自动替换原模型。**
