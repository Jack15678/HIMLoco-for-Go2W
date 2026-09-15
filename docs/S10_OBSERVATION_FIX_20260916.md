# S10 观测保存修复：真实 Gym 零训练检查通过

**最终通过 GPU 验证的源码 SHA：`85c786f90f4774a90e2cb6b883fc3d5c7a051f0a`。**
修复及检查脚本已提交推送 main；本报告是验证后的纯文档提交。服务器仍停在上述已验证 SHA，后续从零训练可固定使用该版本。本任务没有启动训练或创建训练任务。

## 修复与范围

`HIMPPO.act()` 和 `PPO.act()` 原先引用环境的 actor/critic 观测；两个 runner 均先 `env.step()`，再 `process_env_step()`，最后 storage 才 `copy_()`。环境在这期间对终止行执行历史清零，使旧动作、概率和存储观测不匹配。

两处 `act()` 现在分别保存 `obs.clone()` 和 `critic_obs.clone()`。已检查两个 runner、两个 storage、现有检查脚本调用者以及环境重置/观测计算路径。环境重置、动作执行、终止标记和 HIM 下一时刻 critic 观测处理均保持原样。

候选配方保持：PD .0025、decimation 8、策略/历史/奖励周期 20 ms、delay_stride 2；腿 80/2、轮 0/.6；动作缩放 .125/.25/.25/5；初始 std `[.3,.3,.3,.6]*4`。未改奖励、停车、课程、学习率规则或探索参数；未做 A/B，未用子代理。

## 验证结果

本地 PyTorch 2.5.1 CPU 回归通过：PPO/HIMPPO × 独立/共用 critic 观测共四种场景，每种两环境、两步；覆盖终止清零及正常原地推进，核对保存观测、动作、均值、旧概率和价值，并确认模型不变、优化器无 step。Python 编译与改动空白检查通过。

服务器 RTX 4090，镜像 `him-gym:py38-torch201-cu118`，真实 Isaac Gym GPU PhysX。只读加载旧 `model_1000.pt` 及两套 Adam 作诊断。原训练配置逐项比较通过，仅环境数改为 512；沿用 runner 初始化重置和随机回合年龄，无强制终止或预热。执行正常采样顺序，共 48 策略步，无参数更新。

| 指标 | 全部样本 | 终止样本 | 非终止样本 |
| --- | ---: | ---: | ---: |
| 样本数 | 24,576 | 29 | 24,547 |
| 保存 actor 观测最大绝对差 | 0 | 0 | 0 |
| 更新前平均 KL | .00016021728515625 | .00016021728515625 | .00016021728515625 |
| 概率比最小值 | .9999790788 | .9999928474 | .9999790788 |
| 概率比最大值 | 1.0000209808 | 1.0000061989 | 1.0000209808 |

- 原环境旧观测仍有 29 行被重置改写，恰好全部为终止样本；保存观测被改写的行数为 **0**。
- 保存 critic 观测最大绝对差 **0**；新回合历史帧最大绝对值 **0**；保存动作及下一时刻 critic 观测逐元素相等。
- KL 与现有 16 维公式 `+1e-5` 数值底值的最大差 **0**；clip fraction **0**。重算动作均值最大误差 `4.52995e-6`，概率比最大偏离 1 为 `2.09808e-5`。
- PPO/HIM 更新入口调用 **0/0**，优化器 step 新增 **0/0**；两套 Adam 的 step 集合检查前后均为 `{20000}`，完整优化器状态逐元素相等。
- actor、critic、HIM encoder/target/prototypes、std 共 **30 个模型状态张量、575,364 个参数元素**全部逐元素不变。
- 容器退出码 **0**；验证后服务器检出干净、无运行容器。旧模型仅用于诊断，后续从零训练不得加载它。

## 固定版本及启动记录

本地 main 提交推送后，服务器在无运行训练、检出干净时 fetch，并 `git switch --detach` 到完整 SHA；启动前后均核对 SHA 和工作区状态。GitHub 直连超时后，通过 SSH 端口转发复用本地代理完成 Git 获取。部分克隆检出曾因下载缺失对象阻塞，留下的单个检查脚本已保存在服务器 artifacts 的 `partial-checkout/`，没有覆盖用户修改。

两次采样前预检失败已保留：镜像无 git（`7511f47320b267c57218f42660694ab5a93bfd5f`），以及 Gym 单精度 dt 与 Python 浮点精确比较失败（`346137a6ba1c2e748c4c8bbb7bb3740e4b55eb31`）。均未开始策略采样或参数更新；检查脚本修正都在本地提交推送，再由服务器 fetch/检出新 SHA，没有在服务器编辑源码。

最终容器 `s10-observation-fix-verified-20260916` 的源码和模型均为只读挂载。宿主机核验 SHA，传给无 git 的镜像；完整挂载、参数和退出状态见证据中的 `container_inspect.json`。容器内执行：

```sh
python -u legged_gym/scripts/check_s10_rollout_consistency.py \
  --task s10 --headless --num_envs 512 --evidence /evidence \
  --output /output/consistency.json \
  --source-commit 85c786f90f4774a90e2cb6b883fc3d5c7a051f0a
```

CPU 回归入口：`PYTHONPATH=rsl_rl python legged_gym/scripts/check_ppo_observation_snapshot.py`。

## 证据路径

- 本地：`D:\Desktop\Code\HIMLoco-for-Go2W\artifacts\s10-observation-fix-20260916\`。
- 服务器：`/data/HIMLoco-S10-resume-git-20260915/artifacts/s10-observation-fix-20260916/`。
- [结果 JSON](../artifacts/s10-observation-fix-20260916/consistency.json)、[完整 GPU 输出](../artifacts/s10-observation-fix-20260916/consistency.log)、[CPU 回归](../artifacts/s10-observation-fix-20260916/cpu-regression.log)。
- 同目录还有 `source_commit.txt`、`container_inspect.json`、空的 `server_status_after.txt` 及两次预检失败日志。

只提交源码、检查脚本和本报告。日志/JSON 留在忽略的 artifacts；未提交 checkpoint、视频或训练产物。原本地未提交内容及 a772 工作区保持不变。
