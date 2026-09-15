# S10 直接速度命令：model_1500 独立续训 500 轮

## 试验假设与唯一配方改动

策略原本就接收 **vx、vy、yaw 角速度**：actor 当前帧第 6:9 列为这三个量的缩放值。原 `heading_command=True` 改变的是环境的命令生成：采样目标航向，reset 和每步 callback 计算 `clip(0.5 * wrap(target_heading - heading), -2, 2)`，再把结果作为 yaw 速度输入策略。因此机器人转向接近目标航向时，训练要求的 yaw 速度会减小。

本分支只把 S10 的 `commands.heading_command` 改为 `False`，使用现有直接采样路径。yaw 在原配置 `[-1, 1] rad/s` 内采样，随航向变化保持不变，直到原有的重采样或 episode reset。vx/vy 原有采样方式、20% 高速组、线速度小命令归零、三轴范围、10 秒重采样和课程均保留。reset 仍重新采样，但不会把新采样 yaw 覆盖成航向误差。

actor/critic 输入、动作头、6 帧历史、checkpoint 形状均不变。奖励、停车采样、PD、动作尺度、std、学习率规则和优化计算均不变。这是在验证训练命令分布与持续角速度评估不匹配的假设，不能预先认定它是全部转向振荡的唯一原因。

## 固定来源与预算

- 基础提交：`e232fb97bbd9ae9a5648fe7e55587ca8c27d129b`；独立分支 `JackNcodex/s10-direct-velocity-1500-plus500`，不合并停车方向。
- 输入：原观测修复后训练的 `s10-obsfix-scratch2000-20260916/model_1500.pt` 与同目录 `config.json`；原训练 SHA `85c786f90f4774a90e2cb6b883fc3d5c7a051f0a`。
- 通过生产 `HIMOnPolicyRunner.load(load_optimizer=True)` 恢复 actor、critic、HIM encoder/target/prototypes、std、两套 Adam 和 checkpoint LR；逐元素比较整个模型及优化器状态。
- 起点：iter 1500，环境步 294,912,000，两套 Adam 均 30,000 步。**同一 runner 调用 `learn(500)`**，终点 iter 2000，环境步 393,216,000，两套 Adam 均 40,000 步。
- 新增预算：500 轮、98,304,000 环境步、每套优化器 10,000 步。每 50 轮保留 checkpoint，记录已有 KL/clip fraction/小批次 LR、两套 LR、std、奖励分量和课程统计。
- 4096 环境 × 48 步，seed 1；PD 2.5 ms × decimation 8，策略和历史 20 ms，delay_stride 2。沿用腿 80/2、轮 0/.6、动作尺度 .125/.25/.25、轮速尺度 5。
- 原 checkpoint 不包含模拟器、RNG 或完整课程状态。以 seed 1 新建环境和随机化、重建课程；不宣称物理状态无缝续训。

## 预检和运行入口

CPU 检查实际 sampler/callback 方法、PPO/HIMPPO 观测复制和现有 update_stats。`check_s10_direct_velocity.py --evidence PATH --output FILE` 额外在 CPU 上用生产 loader 核对完整恢复；仅为 CPU 存储映射替换 `torch.load` 的返回，不执行任何更新。

GPU 排队任务：停车方向 `01a0a736-96d7-7730-bcfc-3ea2da6ae95c` 优先。必须等其全部训练和评估容器退出，且 GPU 空闲后才执行真实 Gym 预检和训练。CPU-only 容器不传 `--gpus`，显式 `NVIDIA_VISIBLE_DEVICES=void`。

服务器使用独立普通 Git 检出，宿主核验完整 SHA 和干净工作树，源码和输入 checkpoint 只读挂载。镜像 `him-gym:py38-torch201-cu118` 不含 git，入口接收宿主已核验的 `--source-commit`。不 `pip -e`，使用 `PYTHONPATH`。

```sh
python legged_gym/scripts/check_s10_rollout_consistency.py \
  --task s10 --headless --num_envs 512 --evidence /evidence \
  --output /output/zero_gate.json --source-commit FULL_SHA \
  --checkpoint 1500 --direct-velocity
python legged_gym/scripts/resume_s10_direct_velocity.py \
  --task s10 --headless --num_envs 4096 --seed 1 --resume --max_iterations 500 \
  --evidence /evidence --output /output/training --source-commit FULL_SHA
```

Gym 预检复用真实 48 步 rollout 一致性检查：actor/critic 观测保存、reset 历史、旧新策略 KL/ratio、优化器零更新。额外覆盖纯 yaw ±.8、零 yaw、零指令、复合 vx/vy/yaw；改变航向到 −3/−1/0/1/3 rad 时验证命令保持、观测一致和历史每次只推进一帧。检查真实 reset 和 10 秒边界 callback 都不覆盖采样 yaw。

## 固定评估与结论边界

复用 `s10-follow-terrain-v1`，评估检出固定 `49500212be70ffd99a67c34d2bfdc93d3fd132ac`：原 10 秒 Gym17/MuJoCo 站姿，90 秒 Gym/MuJoCo 平地，Gym 10 地形 × 3 初态各 60 秒。使用独立输出目录和新的 model_2000；不覆盖旧 model_2000 或评估结果。

对照 model_1500 已有标准评估，重点比较 yaw ±.8 和复合 ±.6 的左右分段均值、RMSE、振荡幅度、首次 90%、±10% 连续 1 秒稳定时间、展开航向和 XY。同时报告直行、停车、8 cm 台阶及障碍倒车是否退化。振荡幅度从同一保存轨迹额外统计峰峰值和去均值 RMS，不改变任何测试指令或通过口径。能力不足只记录，不追加训练、调参或第三组试验。

当前为实施和排队说明；实际运行 SHA、零更新报告、最终预算、曲线视频与行为结论在运行后补充。训练总回报不代替行为比较。
