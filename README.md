# Embodied RSI

把 [EmbodiedJev](https://github.com/FBddcz/embodied-jev) 的机械臂仿真与 [RSI-Jev](https://github.com/Shanghua-Gao/RSI-Jev) 的可训练决策模型结合起来，建立可测量的持续学习闭环。

**观察 → 选择技能 → 本地执行 → 记录结果 → 纠正与训练 → 独立评测 → 达标晋级。**

![训练监控 dashboard，来自真实本机实验](docs/assets/dashboard.png)

## 当前可运行能力

- MuJoCo / Franka Panda：搬运入盘、堆叠、越障搬运，真实接触与物理成功判定。
- 固定技能候选集合；只过滤越界目标。模型选择技能，代码生成航点并执行 IK。
- 示范采集、模型访问状态的纠正采集、仿真分支软标签、整轨迹数据隔离、历史数据回放。
- 两个明确区分的训练后端：CPU `compact` 参考学习器；GPU `rsi` 上游决策头／文本塔微调。
- 独立场景配对评测、bootstrap 区间、风险和退化门槛、检查点哈希绑定的版本晋级。
- 实时 dashboard：损失、验证准确率、梯度、学习率、吞吐、GPU/CPU/内存、异常事件和实验记录；可启动和停止实验。

已运行两轮真实仿真学习闭环，以及固定版本 RSI-Jev 0.8B 的 20 步 GPU 微调与检查点重载验证。具体结果见 [验证记录](docs/VALIDATION.md)。**Compact 是 CPU 参考模型，与 RSI-Jev 模型分别报告。**

视觉训练、完整 LIBERO、任务奖励强化学习和真机执行是后续阶段，见 [实施路线](docs/ROADMAP.md)。当前的学习能力是结构化仿真状态下的技能选择，不能据此宣称图像控制或真机泛化。

## 安装

需要 Python 3.11+、Git 和 [uv](https://docs.astral.sh/uv/)。无需 Node.js，dashboard 不使用外部 CDN。

```bash
git clone https://github.com/gaohuan2020/embodied-rsi.git
cd embodied-rsi
python3 scripts/bootstrap.py
```

脚本根据 `upstreams.lock.json` 下载精确 commit 到忽略的 `vendor/`，创建 `.venv` 并安装项目、仿真和测试依赖。核心版本固定在 `constraints-simulation.txt`，已验证的 GPU 包版本记录在 `docs/environment-rsi.json`。

需要训练 RSI-Jev 时另外安装独立环境：

```bash
python3 scripts/bootstrap.py --rsi
```

此步骤安装 PyTorch；模型权重在首次 RSI 训练时下载。GPU 微调建议先使用小模型验证。已验证平台为 Linux / aarch64 / NVIDIA GB10；其他 CUDA 平台需自行验证内核、精度和资源用量。

## 启动 dashboard

```bash
.venv/bin/embodied-rsi dashboard --port 8091
```

打开 **http://127.0.0.1:8091**，点击「启动实验」，选择「完整学习闭环」。默认运行三个任务、每任务 30 个场景、两轮采集与训练。

启用 dashboard 的 RSI 训练按钮：

```bash
.venv/bin/embodied-rsi dashboard --port 8091 --rsi-python "$PWD/.venv-rsi/bin/python"
```

dashboard 是本地工具；远程查看使用 SSH 端口转发。数据、权重和 SQLite 记录均在本机 `artifacts/`，不会提交到 Git。页面重启仍可读取实验记录；正在运行的 CLI 作业也会持续写入监控。

## 两轮学习闭环

```bash
.venv/bin/embodied-rsi cycle --rounds 2 --episodes 30 --steps 500
```

第一轮从规则示范开始。第二轮让第一轮模型自行选择动作，在其实际访问的状态上收集教师纠正，再与历史数据混合训练。每轮分配不同种子，数据划分由场景组的稳定哈希决定。

这条命令执行采集和训练，**不会自动晋级模型**。晋级必须另做未见场景的配对评测。当前教师纠正来自上游规则策略，尚未接入人工遥操作或大模型教师。

单独采集与构建数据集：

```bash
.venv/bin/embodied-rsi collect --episodes 30 --seed-start 0
.venv/bin/embodied-rsi dataset \
  --episodes-files artifacts/runs/COLLECTION_ID/episodes.jsonl \
  --output artifacts/datasets/robot-v1
.venv/bin/embodied-rsi train --dataset artifacts/datasets/robot-v1 --steps 500
```

将 `COLLECTION_ID` 替换为采集命令输出的 ID。想使用仿真分支标签，采集时加入 `--label-mode rollout`；会增加采集成本。分支未来结果仅进入标签，模型输入不会获得这些结果。

## RSI-Jev 训练与接入

使用单出口文本版本；默认 `v3.0-2b`。首期先训练决策头，`--tune-tower` 可开放文本塔，嵌入保持冻结。

```bash
.venv-rsi/bin/embodied-rsi train --backend rsi \
  --dataset artifacts/datasets/robot-v1 \
  --checkpoint v3.0-2b --steps 100 --batch-size 2
```

正式实验应以 `--revision` 固定 Hugging Face 检查点 commit。已经验证的小模型命令：

```bash
.venv-rsi/bin/embodied-rsi train --backend rsi \
  --dataset artifacts/datasets/robot-v1 \
  --checkpoint v1.0-0.8b \
  --revision f9248caceb89caf2e6c968ea33bf0d6eb7f957b0 \
  --steps 20 --batch-size 2
```

训练复用上游 `Case`、编码器、候选重排和监督目标；用本仓库训练循环逐步记录监控。输出为上游可加载的 `tower.safetensors / scorer.safetensors / meta.json`，保存后重新加载并比较预测。模型微调后不沿用父模型的旧校准。

RSI 后端初版支持文本、单出口监督微调。图像／多出口版本会明确拒绝，不会静默修改架构。RSI 概率目前未在机器人数据上校准；CPU 参考模型使用开发集温度校准，但其概率也不等于任务成功率。

训练好的模型通过上游服务连接 EmbodiedJev：

```bash
.venv-rsi/bin/rsi-jev serve artifacts/runs/TRAIN_ID/checkpoint --port 8000
```

在 EmbodiedJev 中选择「结构化决策 API」，地址填 `http://127.0.0.1:8000/v1/systemone`，模型名使用服务实际返回的名称或 `jev-latest`。也可直接用本仓库的 `collect --backend rsi --checkpoint ...` 收集微调模型轨迹。

## 配对评测与发布

```bash
.venv/bin/embodied-rsi evaluate --backend compact \
  --checkpoint artifacts/runs/CANDIDATE_ID/checkpoint \
  --champion artifacts/runs/PARENT_ID/checkpoint \
  --dataset artifacts/datasets/robot-v1 \
  --episodes 30 --seed-start 10000
.venv/bin/embodied-rsi promote --report artifacts/runs/EVALUATION_ID/report.json
```

RSI 评测用 `.venv-rsi/bin/embodied-rsi` 和 `--backend rsi`。省略 `--champion` 时与规则教师比较。`--intervention` 可以加入中途物体位移，单列为扰动协议。

冻结的初版门槛：至少 90 个配对场景、每任务至少 30 局；成功率提升至少 5 个百分点且配对提升的 95% bootstrap 下界大于零；每个旧任务退化不超过 2 个百分点；碰撞、失抓和动作拒绝总数不增加。未达标会拒绝更新 `champion.json`。这些是初期工程门槛，有限样本和部分碰撞监测不能保证真机安全。

模型和控制器改动分开评测。当前报告只适用于本仓库固定技能协议；每次发布前冻结候选、感知、控制器、种子及预算。测试结果用于发布决策，不用于调参；后续版本应轮换未见发布场景。

## 训练检测

| 检测 | 行为 |
|---|---|
| NaN / Inf 损失或梯度 | 中止更新，记录失败；非有限指标不写成 JSON 数值 |
| 梯度范数 > 100 | 记录警告；训练梯度裁剪阈值为 1 |
| 损失超过最近均值 4 倍且 > 2 | 记录损失突增警告 |
| 验证损失 > 训练损失 2 倍 | Compact 记录泛化差距警告 |
| 已开始训练后 120 秒无指标 | 记录训练停滞警告 |
| 主机内存 > 90%、显存 > 95%、温度 > 85°C | 记录资源警告；设备未报告的显存显示未知 |
| 运行中的实验 60 秒无事件 | dashboard 标记心跳过期 |

SQLite WAL 支持训练写入和 dashboard 同时读取。CLI 中断会保存已采集轨迹并记录取消状态。已有数据集不可覆盖；文件校验和及跨场景分组检查在每次训练前执行。

## 开发与许可证

```bash
.venv/bin/ruff check src tests scripts
.venv/bin/pytest -q
```

结构与接口见 [架构说明](docs/ARCHITECTURE.md)。GitHub Actions 会执行代码检查和真实 MuJoCo 单局测试。GPU 模型训练与浏览器端到端测试在本机单独验证。

本仓库代码使用 MIT。两个上游的代码、模型权重、机器人资产和数据各自遵循上游许可证；模型权重不包含在本仓库。没有发布真机可执行接口。
