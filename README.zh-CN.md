# ⚡ 源序天线锻造平台 Source Sequence Antenna Forge (YAF)

[English](README.md) | [中文](README.zh-CN.md)

**AI 驱动的天线发明平台** —— 自动探索、生成、优化、验证此前不存在的新型天线拓扑。

## 快速演示 —— 一条命令，一张真实仿真图

下面这张 PNG 是 `scripts/demo_wow.py` 跑出来的，**图上每一个数字都来自真实的
NEC2 求解**（通过 `necpp` Python 绑定的矩量法）—— 无 mock、无解析降级，求解器
装不上就显式抛 `SolverUnavailable`，绝不伪造输出。复现一次只需要一条命令：

```bash
python3 scripts/demo_wow.py    # → docs/assets/dipole_demo.png
```

![偶极子真实 NEC2 演示](docs/assets/dipole_demo.png)

三个子图分别是：(1) 输入阻抗 R(f) / X(f)，并标出谐振点（X → 0）；(2) E 面极坐标
方向图，标注 NEC 实测峰值增益 2.13 dBi；(3) S11(f) / VSWR(f)，并阴影标出 −10 dB
带宽。角注框给出逐项的"实测 vs 教科书"对比（教科书半波偶极子参考值：R ≈ 73 Ω、
G ≈ 2.15 dBi）。

闭环逆向设计 —— 让真实 NEC2 进入优化回路：

```bash
python3 scripts/demo_inverse_design.py    # → docs/assets/inverse_design_convergence.png
```

![逆向设计收敛过程](docs/assets/inverse_design_convergence.png)

对偶极子长度做黄金分割搜索，14 轮迭代 / **16 次真实 NEC2 求解 / 约 6 ms** 墙钟
时间，从 ±75 mm 的搜索区间收敛到 **L = 477.892 mm**（300 MHz 下 ≈ 0.478 λ），
得到 R = 71.85 Ω、X = +0.03 Ω、G = 2.13 dBi —— 即优化器在目标函数里没有植入任何
天线理论的前提下，亚毫米精度地复现了教科书的细线谐振长度。

### 旗舰案例 —— 9 参数 Yagi-Uda 逆向设计

平台的旗舰演示：300 MHz 的 5 单元 Yagi-Uda，**9 个连续设计变量**（5 个单元长度
+ 4 个单元间距），由 `scipy.optimize.differential_evolution` 驱动，并在**每一次
迭代里都用真实 NEC2** 评估 —— 5858 次求解器调用，笔记本上 12.7 秒墙钟时间。完整
讲解：[`docs/case_study_yagi.md`](docs/case_study_yagi.md)。

```bash
python3 scripts/case_yagi.py     # 基线 + 优化 → results/ 下的 JSON
python3 scripts/plot_yagi.py     # → docs/assets/yagi_design.png
```

![Yagi-Uda 逆向设计](docs/assets/yagi_design.png)

**干净的 5-vs-5 对比（相同单元数，相同 NEC2 后端）：**

| 指标 | Viezbicke 5 单元 (NBS TN 688) | YAF AI 5 单元 | Δ |
|---|---|---|---|
| 前向增益 G_fwd | +11.03 dBi | **+12.63 dBi** | **+1.60 dB** |
| 前后比 F/B | 13.79 dB | **15.00 dB** | **+1.21 dB** |
| 天线臂长 | 1.00 λ | 1.17 λ | +0.17 λ |
| 单元数 | 5 | 5 | 0（归因干净） |

**在相同单元数下，AI 设计在增益*与* F/B 两个维度上同时 Pareto 占优经典的
Viezbicke 5 单元基线。** 在更广的公开 5 单元设计空间里（Viezbicke、ARRL 手册、
DL6WU、Lawson/Cebik），AI 在两轴上严格占优其中 3 个（见 `docs/case_study_yagi.md`
§6）。

优化器*独立地*复现了 Viezbicke 式的引向器锥度（L：0.440 → 0.434 → 0.429 m，从后
向前单调递减）以及均衡的 0.243 λ 反射器间距 —— 二者都与公开的 5 单元 Yagi 配方
一致 —— 而它对天线设计一无所知。回路里的"AI"部分只是个微分进化；平台真正独特的
贡献在于*它所对照优化的对象*：真实的矩量法物理，而非代理模型或解析模型。

> *副产品：那 5858 条 DE 历史记录（`results/yagi_optimized.json`）是一个现成的
> FNO 代理模型训练集 —— 每一组输入几何与输出（R、X、G_fwd、G_back、F/B）都已
> 记录，谁想在后续阶段尝试主动学习的 DE 都可直接取用。*

## 架构概览

```
┌────────────────────────────────────────────────────┐
│  Web UI (React + Three.js + WebGPU)                │
│  3D 编辑器 │ 设计空间浏览器 │ 实验跟踪 │ 实时仿真监控│
└────────────────────────┬───────────────────────────┘
                         │ REST / WebSocket
┌────────────────────────▼───────────────────────────┐
│  API 网关 (FastAPI + Pydantic)                     │
└────────────────────────┬───────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────┐
│  编排核心 (Python asyncio + Celery)                │
└───┬─────────┬──────────┬───────────┬───────────┬───┘
    │         │          │           │           │
┌──────┐  ┌──────┐  ┌────────┐  ┌────────┐  ┌────────┐
│ 几何 │  │ AI   │  │ 求解器 │  │ 优化器 │  │ 后处理 │
│ 内核 │  │ 引擎 │  │ 适配器 │  │ 引擎   │  │ 分析器 │
└──────┘  └──────┘  └────────┘  └────────┘  └────────┘
```

## 一键启动

```bash
# 克隆项目
git clone https://github.com/1ove9/antenna-forge.git yaf && cd yaf

# 复制环境变量模板
cp .env.example .env

# 启动全部服务
docker compose up -d

# 健康检查
curl http://localhost:8000/health
# → {"status": "ok", "version": "0.1.0"}

# 访问前端
open http://localhost:5173
```

## 核心模块

| 模块 | 路径 | 说明 |
|------|------|------|
| 领域模型 | `yaf_core/domain/` | Design、Geometry、Simulation、Optimization |
| 端口协议 | `yaf_core/ports/` | SolverAdapter、AIBackend、CADBackend |
| 几何内核 | `yaf_core/geometry/` | OpenCASCADE、参数化生成器、SIREN、拓扑优化 |
| 物理模型 | `yaf_core/physics/` | 超表面、RIS、OAM、石墨烯、时空调制 |
| 求解器 | `yaf_solvers/` | openEMS、NEC2、MEEP、HFSS、CST、FEKO |
| AI 引擎 | `yaf_ai/` | Diffusion、VAE、GAN、FNO、PINN、可微 FDTD、贝叶斯优化 |
| API 服务 | `yaf_api/` | FastAPI + WebSocket |
| 任务队列 | `yaf_worker/` | Celery + Redis |
| 数据库 | `yaf_db/` | PostgreSQL + Qdrant |
| 前端 | `frontend/` | React 18 + Three.js + TypeScript |

## API 快速上手

```bash
# 创建设计
curl -X POST http://localhost:8000/api/v1/designs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test_dipole",
    "frequency_range": [2.4e9, 2.5e9],
    "size_constraint": {"x_min": -0.1, "x_max": 0.1, "y_min": -0.1, "y_max": 0.1, "z_min": -0.1, "z_max": 0.1},
    "polarization": "linear",
    "material_palette": ["copper"]
  }'

# 用 NEC2 仿真
curl -X POST http://localhost:8000/api/v1/simulations \
  -H "Content-Type: application/json" \
  -d '{"design_id": "<design-uuid>", "solver": "nec2", "frequency_min": 2400000000, "frequency_max": 2500000000}'
```

## AI 演示

```bash
# 可微 FDTD 梯度优化
python -m yaf_ai.differentiable.diff_fdtd_jax --demo

# VAE 天线几何生成（--epochs 2 即可触发权重保存；20 是默认收敛轮数）
python -m yaf_ai.generative.vae_designer --train --epochs 20

# 贝叶斯优化
python -m yaf_ai.optimization.bayesian --demo

# 端到端逆向设计流水线
python -m yaf_ai.inverse_design.pipeline --demo

# 半波偶极子端到端示例（NEC2 → S11 + 增益）
python scripts/demo_dipole.py
```

> 见 `docs/HONEST_STATUS.md`：**两个求解器都不伪造结果。** NEC2（`necpp`）
> 与 openEMS 都跑真实求解器，后端缺失时显式抛 `SolverUnavailable` —— 两条路径
> 都没有任何静默的解析降级。

## 验收命令

项目的验收命令：6 条核心命令（基础设施、测试、可微 FDTD、生成模型、demo），加上
围绕真实 NEC2 的真值校验与 Yagi 案例 demo。本仓库当前全部通过以下命令：

```bash
# 1. 基础设施启动
docker compose up -d                                          # postgres / redis / minio / qdrant / api

# 2. API 健康检查
curl -fsS http://localhost:8000/health                        # → 200 {"status":"ok","version":"0.1.0"}

# 3. 测试套件（含真实 NEC2 半波偶极子断言）
pytest tests/ -x -q                                           # → all passed

# 4. 可微 FDTD —— 梯度流验证
python -m yaf_ai.differentiable.diff_fdtd_jax --demo          # → "✓ Gradient flow verified" + 单调下降的 loss

# 5. VAE 训练 + 权重检查点
python -m yaf_ai.generative.vae_designer --train --epochs 2   # → 写出 models/vae_designer.pt

# 6. 偶极子 demo —— S11 + 增益（真实 NEC2）
python scripts/demo_dipole.py                                 # → 打印 S11/VSWR/峰值增益 (2.20 dBi)

# 7. NEC2 真值校验 vs 教科书
python3 scripts/verify_dipole.py                              # → PASS: R=68.30 Ω (误差 6.4%), G=2.12 dBi

# 8. openEMS 真值校验 —— 真实 full-wave FDTD vs 腔模解析公式
python3 scripts/verify_patch.py                               # → PASS: 谐振 2.435 GHz vs 2.513 GHz (误差 3.1%)

# 9. 三联图展示 PNG（Z 扫描 / 极坐标方向图 / S11+带宽）
python3 scripts/demo_wow.py                                   # → docs/assets/dipole_demo.png

# 10. 闭环逆向设计（真实 NEC2 在回路中）
python3 scripts/demo_inverse_design.py                        # → 477.89 mm + inverse_design_convergence.png

# 11. Yagi-Uda 案例 —— 9 参数 DE × 真实 NEC2
python3 scripts/case_yagi.py                                  # → 基线 + 优化 JSON，+1.60 dB Pareto 占优 Viezbicke
python3 scripts/plot_yagi.py                                  # → docs/assets/yagi_design.png

# 静态类型检查
mypy yaf_core yaf_ai yaf_solvers --strict                     # → Success: no issues in 64 source files
```

各条命令的逐条可信度标注见 `docs/HONEST_STATUS.md`（2026-05-25 修订）；"全绿之后
还差什么"见 `docs/next-steps.md`；Yagi 案例的完整讲解见
`docs/case_study_yagi.md`。

## 技术栈

| 层 | 技术 |
|----|------|
| 后端 | Python 3.11、FastAPI、Pydantic v2 |
| 可微分 | JAX、Flax、Optax |
| 深度学习 | PyTorch 2.x |
| 几何 | pythonocc-core、trimesh、gmsh |
| 任务队列 | Celery + Redis |
| 数据库 | PostgreSQL 16、Qdrant（向量） |
| 对象存储 | MinIO（S3 兼容） |
| 前端 | React 18、TypeScript、Vite、Three.js |
| 部署 | Docker Compose（dev）、Kubernetes（prod） |


## 许可证

YAF 源代码以 **MIT 许可证** 发布 —— 见 [`LICENSE`](LICENSE)。

**第三方依赖各自持有其许可证，部分为 copyleft。** 其中可选的 `necpp` 矩量法
后端，以及 `openEMS` / `CSXCAD` FDTD 后端，均为 GPL 许可。YAF 不捆绑或再分发
它们；用户自行安装，并自行承担由此可能产生的"组合作品"义务。完整的许可证边界
讨论以及对下游再分发者的缓解建议见 [`NOTICE`](NOTICE)。本说明出于善意提供，不
构成法律意见。


## 开源核心版与增强版

YAF 采用 **open-core（开源核心）** 模式。本仓库是**核心引擎**：免费、可自托管、
MIT 许可，覆盖**线天线**（NEC2 矩量法）与**平面 / 贴片天线**（openEMS full-wave
FDTD），由**经典优化**（差分进化 / 黄金分割搜索）驱动，在这个范围内是完整、可
独立使用的。

**当前开源核心版已具备**

- 基于 `necpp` 的真实 NEC2 矩量法线天线仿真 —— 无解析降级（求解器缺失时直接抛
  异常，绝不伪造结果）。
- 基于真实 openEMS（`openEMS` / `CSXCAD` Python 绑定）的 full-wave FDTD 仿真：
  自动建 CSX 结构、跑时域求解、从端口提取 S11 / 输入阻抗、用 NF2FF 提取增益方向
  图。同样的诚实原则 —— 绑定缺失时抛 `SolverUnavailable`，绝不伪造结果。
- 真实求解器进入每一次迭代的经典优化：半波偶极子谐振搜索、9 参数 Yagi-Uda 逆向
  设计。
- 已知答案真值校验与可复现基准：半波偶极子（`scripts/verify_dipole.py`，NEC2）
  与矩形微带贴片（`scripts/verify_patch.py`，openEMS —— 实测谐振频率与腔模解析
  公式相差 3.1%）。另见 `docs/case_study_yagi.md`。
- FastAPI 服务、Pydantic 领域模型，以及求解器 / AI 适配器接口。

源序科技（Source Sequence）另行维护一个面向专业与商业用户的**增强版**（hosted /
commercial edition，内嵌网页平台）。为不夸大，以下能力均属**规划中 / 路线图**，
**在开源核心版与增强版中都尚未交付：**

**规划中 / 路线图（当前尚不可用）**

- 更广的 full-wave 覆盖 —— 微带阵列、超表面、完整 3D 结构，以及商业求解器
  （HFSS / CST / FEKO / COMSOL）。*（openEMS FDTD 后端现已是真实可用的，并通过
  了一个贴片天线真值校验；更广的几何覆盖与商业求解器适配器仍在路线图上，见
  `docs/HONEST_STATUS.md`。）*
- 生成式 AI 几何设计（diffusion / VAE）接入真实物理 oracle。*（这些生成模型在
  仓库内目前仅为 **early / experimental** 代码：用合成几何训练、尚未接入仿真闭环，
  不是 production-ready。）*
- 多目标、多频带联合优化。
- RIS（可重构智能表面）逆向设计。
- 浏览器内可视化设计平台。
- 云端算力 —— 无需本地安装求解器即可运行设计。
- 团队协作与设计版本管理。

一句话区分：**开源核心版让你验证方法、复现基准；增强版面向把它用到真实工程项目
里。** 上述路线图中的任何一项都不代表当前已实现。

- 增强版正在开发中，敬请期待。
- 商业合作咨询：目前请先提交 GitHub issue。


## 致谢

本项目由一名工程师独立完成，实现过程中借助了 AI 编程助手。架构设计、物理验证
方法（真实 NEC2 真值校验与已知答案回归），以及基准测试设计（5-vs-5 Yagi 对比与
`docs/HONEST_STATUS.md` 中的诚实分级）均出自作者本人。凡某个模块的设计参考了某个
开源项目，该项目都会在文件顶部以及 `NOTICE` 中注明引用。
