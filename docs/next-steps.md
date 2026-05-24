# 从"全绿管线"到"能发明可制造天线"——下一步路线图

> 配套阅读：`docs/HONEST_STATUS.md`。读完那份能理解为什么下面这条路是不得不走的。
>
> 排序按**依赖关系**——每一步解锁下一步，跳步会导致"看起来更花哨但下游验证不动"。

---

## 阶段一：把"求解器"从演示级抬到工程级

### 1.1 WSL2 + 真实 openEMS + nec2c（第一步、阻塞一切下游）

为什么是第一步：当前所有"物理仿真"路径都走 §1 fallback，AI 管线接的不是物理 oracle 而是闭式近似。在没有真求解器之前，再多优化 VAE / FNO / 可微 FDTD 都是空转。

具体动作：

1. **WSL2 Ubuntu 24.04**：`wsl --install -d Ubuntu-24.04`，给 16 GB 内存 / 8 核 / 200 GB 虚拟盘。
2. **nec2c**：`apt install nec2c` 或者从 `https://www.qsl.net/5b4az/pages/nec2.html` 编译。验证：`nec2c -i samples/dipole.nec -o out.txt` 跑通。
3. **openEMS 全家桶**：
   - 编译 openEMS 主体（C++）。文档在 `_reference/openEMS/INSTALL`。
   - Python 绑定：`pip install openEMS CSXCAD` 或从源码 build `_reference/openEMS/python/`。
   - 验证：跑 `_reference/openEMS/python/Examples/rectangular_resonant_cavity.py`，能拿到 S 参数曲线就算成功。
4. **把 YAF API 容器换到 WSL**：现在的 docker-compose 跑在 Windows Docker Desktop 上，没法访问 WSL 里编译的 openEMS。两条路：(a) 在 WSL 里 `docker compose up`；(b) 容器里加一层 install openEMS 的 RUN 层（推荐 a，build 时间短得多）。
5. **集成验证**：删掉 `OpenEMSAdapter._run_analytical` 的 fallback 入口，让 `_run_with_openems_api` 成为唯一路径，写一个"半波偶极子 → S11 谐振点在 2.45 GHz ±5%"的回归测试。

### 1.2 用解析解锚定每个 adapter

对 NEC2 和 openEMS 各做"已知答案"测试（参考 Balanis 第 4 章）：

| 天线 | 频段 | 期望 S11 | 期望 \|Z_in\| | 期望增益 |
|---|---|---|---|---|
| 自由空间半波偶极子（L = λ/2 - δ）| 2.45 GHz | < −10 dB（带变压器）| 73 + j42 Ω（无变压器）| 2.15 dBi（±0.3）|
| 1/4 波单极 + 大地平面 | 2.45 GHz | < −10 dB | 36 + j21 Ω | 5.15 dBi |
| 矩形贴片（W = 0.6λ, L = 0.3λ, FR4 h=1.6mm）| 2.4 GHz | < −10 dB | 50 Ω 馈电 | 6–8 dBi |
| 3-元 Yagi（Balanis 例 11.7.1 参数）| 300 MHz | — | — | 7.5 dBi |

通不过任何一项 → 适配器有 bug，先修。这套验证比当前的"status == success"硬得多。

### 1.3 把 fallback 改成"显式 unavailable"而不是"伪造结果"

`_compute_analytical` 和 `_run_analytical` 当前会**返回一个看起来正常的 `SimulationResult`**，调用方分不清"真求解"vs"近似"。改造：

- `SolverAdapter.health_check()` 在 boot 时跑一次，结果落到 `SimulationResult.solver_metadata["solver_mode"]` 里（`"native" | "subprocess" | "fallback_analytical"`）。
- API 层在响应里显式带 `"warning": "..."` 字段，前端红色提示，**让用户知道这次结果不是 EM 真值**。
- 提供一个 `--no-fallback` 模式，没有真求解器就直接 `raise SolverUnavailable`，不要静默降级——这样 CI 里能挡住"求解器没装、CI 看起来还过"的灾难。

---

## 阶段二：把 AI 管线接到物理 oracle

需要阶段一先就位（否则训练数据是噪声）。

### 2.1 数据集：从真实物理仿真生成 ≥ 10⁴ 样本

VAE / FNO / DDPM 训练所需的几何 ↔ S 参数标签对，必须用阶段一的真求解器跑出来：

- 用 `yaf_core/geometry/parametric.py` 已有的参数化生成器（dipole / patch / spiral / horn / sierpinski）扫参数空间，对每个几何用 openEMS 仿一次。
- 数据集 schema：`(geometry_grid_64x64, S11_vs_freq[51], gain_dbi, vswr, bandwidth_pct)`。
- 单次 openEMS 仿真 ≈ 30-90 秒（2D 模型）/ 5-20 分钟（3D），10⁴ 样本意味着 10⁵ 秒 = ~28 小时的求解器机时——值得，但要规划。
- 落到 MinIO + Postgres metadata，**绑定 hash 防漂移**。

### 2.2 FNO 真正接到 `_screen_candidates`

当前 `pipeline.py:_screen_candidates` 走的是 `compactness/n_faces` 启发式打分（HONEST_STATUS §3.3）。2.1 的数据集就位后：

1. 在 B1 数据上训 FNO 拟合 `(geometry) → S11(f)`。MSE / 频域 L2 目标。
2. 用一个 hold-out 100 几何对 openEMS 真值验证 FNO，要求 |S11(f)| 误差 < 1.5 dB（90 分位）。
3. 把 `_screen_candidates` 改成"调用 FNO 预测 S11 → 计算 figure of merit → 选 top-k"。
4. 集成测试：generate 50 个候选，FNO 筛 top-5，openEMS 验证 → top-5 的真实 FoM 要显著高于随机 5 个。

### 2.3 VAE 改成有条件生成

当前 VAE 是无条件——`generate()` 只采样隐空间，没有"我要 2.4 GHz、5 dBi、左旋圆极化"输入。改造：

- Encoder 输入：`(geometry, spec_embedding)`，spec_embedding 来自 `DesignSpec.frequency_range + target_gain_dbi + polarization`。
- Decoder 输入：`(z, spec_embedding)`。
- Loss：BCE + β·KL + λ·spec_consistency（用 FNO 预测的 S11 和 spec 做匹配损失）。
- 这是论文 arxiv:2505.18188 真正的玩法（也是 `_reference/Inverse-design-of-metasurfaces` 里 youxch 的范式）。

### 2.4 可微 FDTD 从 2D TM 抬到 3D（或者保留 2D 但换 CPML）

当前 `diff_fdtd_jax.py` 是 2D TM + 解析衰减 PML，loss landscape 太平、PML 反射也太大。两条路：

- **保守**：保留 2D 但换成真正的 CPML（fdtdx `_reference/fdtdx/src/fdtdx/objects/boundaries/perfectly_matched_layer.py` 的算法），收敛速度会明显提升。
- **激进**：抄 fdtdx 的 3D 实现到 YAF，作为一个独立的 `yaf_ai/differentiable/diff_fdtd_jax_3d.py`，保留现在的 2D 当作 unit test 用。

无论哪条，都要写一个 "梯度数值正确性" 的回归测试：和有限差分梯度对比，相对误差 < 1e-4。

---

## 阶段三：从仿真到制造

### 3.1 几何 ↔ 制造约束

当前 `yaf_core/geometry/` 输出的是顶点/面表示，没有任何 DfM（Design for Manufacturing）约束。要加：

- **最小线宽 / 最小间距**：5 mil PCB 工艺要求 ≥ 0.127 mm，10 mil 要 ≥ 0.254 mm。
- **过孔限制**：直径范围、aspect ratio、blind/buried 配置。
- **介质叠层**：FR4 / Rogers RO4350B / RO4003C 的厚度梯度。
- **PCB 工艺 vs LTCC vs 3D 打印**：每种工艺对应一个 `ManufacturabilityProfile`，违反约束的几何在 SIMP 滤波阶段就被惩罚。

在 `yaf_core/geometry/parametric.py` 和 SIMP 里加 manufacturability penalty term。

### 3.2 输出格式：Gerber / IPC-2581 / STEP

- `yaf_solvers/.../to_native_format` 现在只输出仿真用格式。要补：
  - `yaf_core/manufacturing/gerber.py`：从 SIMP 密度场生成 Gerber RS-274X。
  - `yaf_core/manufacturing/step_export.py`：用 pythonocc-core 把 BREP 导出 STEP（已经有 `kernel.py` 的 OCC 包装做基础）。
  - `yaf_core/manufacturing/bom.py`：物料清单（基板、铜厚、表面处理）。

### 3.3 制造 → 测试闭环

- 把生成的设计发到打样厂（JLCPCB / PCBWay API）；
- 收到样品后用矢网仪（VNA）测 S11 / S21，结果上传到 `SimulationJob` 的 measurement 字段；
- 把 measurement 数据对照 openEMS 仿真 → 模型残差作为下一轮 GP 的训练点（active learning，论文 arxiv:2505.18188 的最后一块）。

---

## 阶段四：服务化、可观测性、协作

只有阶段一/二/三跑通了一遍真实闭环，下面的才有意义。

### 4.1 接通 docker-compose 里那 5 个服务的真实流量

- `yaf_api/main.py::lifespan` 接 PG/Redis/MinIO/Qdrant。
- `yaf_db/models.py` 写 Alembic migration。
- `yaf_worker/tasks/simulate.py` 真正调 openEMS，结果落 MinIO。
- WebSocket 推送 solver 进度，前端订阅。

### 4.2 前端：从骨架到可用工具

- `frontend/src/pages/DesignEditor.tsx` 现在只是占位。需要：
  - 真实的 react-three-fiber 3D 编辑器（参数化几何、SIMP 密度可视化）；
  - S 参数 / 远场图实时绘制；
  - 设计版本对比；
  - 物料/工艺选择器。
- 截止指标：UX 测试至少 3 个人能从"输入 spec"走到"导出制造文件"全程不卡。

### 4.3 多用户、权限、审计

- 现在 `yaf_api/routers/` 用内存 dict，所有用户共享。
- JWT 鉴权 + per-design ACL；
- 审计日志（哪个用户在哪个时刻改了哪个 design，绑定 git 风格的 design version hash）。

### 4.4 监控

- Prometheus 指标（`prometheus_client` 已在 deps 里，但没用）：`yaf_solver_duration_seconds{solver="openems"}`、`yaf_pipeline_loops_total`、`yaf_design_state{state="..."}` 等。
- structlog → loki / OpenTelemetry。

---

## 阶段五：把"发明"做成实事

到这里基础设施齐了，是时候真正瞄准"发明"。

### 5.1 新物理类的 benchmark suite

为每个 §3 物理目标（超表面 / RIS / OAM / 时空调制 / 等离子体 / 液态金属 / 石墨烯）准备一个"已发表论文的可复现实验"：

- 选 2-3 篇 2023-2025 顶刊（TAP / IEEE Trans. Antennas / Nature Comm.）的目标性能。
- 用 YAF 跑同样 spec，看能不能从 0 启动生成出**结构上不同、性能持平或更好**的设计。
- 失败案例比成功案例更重要——记录到 `docs/case-studies/*.md`，分析为什么 AI 没找到那个解（数据集不够？loss 不对？物理 oracle 漂了？）。

### 5.2 知识产权 & 论文

- 任何 YAF 生成的"新"设计，自动跑 prior-art 检索（USPTO / Google Patents API），打个 novelty score。
- 高 novelty + 高仿真性能的设计 → 工艺组小批量打样 → 形成专利材料。
- 这本来就是源序科技商业层面要的东西。

---

## 时间预算（粗估）

| 阶段 | 工程师·周（单人）| 解锁下游 |
|---|---|---|
| 1.1 WSL + 真实 solver | 1–2 | 全部 |
| 1.2 已知答案测试 | 1 | 2.1 |
| 1.3 fallback 收口 | 0.5 | — |
| 2.1 真实数据集生成 | 2–3（含 ~30 小时机时）| 2.2/2.3 |
| 2.2 FNO 接 oracle | 2 | 2.3 |
| 2.3 conditional VAE | 2 | 5.1 |
| 2.4 真 PML / 3D FDTD | 3–4 | 5.1 |
| 3.1 DfM 约束 | 2 | 3.2 |
| 3.2 Gerber/STEP 导出 | 2 | 3.3 |
| 3.3 制造 → 测试闭环 | 4（含寄样回程时间）| 5.1 |
| 4.1–4.4 服务/前端/监控 | 4–6 | 5.1 |
| 5.1 物理 benchmark suite | 6+（持续）| 论文 / 专利 |

**乐观估计**：单人全栈 ~30 工程师·周到达"能跑真实 closed-loop 设计"的状态。
**实际**：会更长——制造打样的物理寄送和测量时间不能压缩。

---

## 不要做的事（trap list）

1. **现在去优化 VAE 损失函数 / 调超参 / 上 diffusion**——在 §3.2 的"训练数据没有物理标签"修好前都是徒劳。
2. **现在去补 HFSS / CST / FEKO / COMSOL adapter 真实实现**——这些是商业 licensed solver，没真实测试机器和 license 之前每写一行都是猜的。先把开源的 openEMS 和 nec2c 做扎实。
3. **现在去做 Kubernetes / 高可用 / 多节点训练**——单机都没跑透，分布式只是把单点 bug 放大到多点。
4. **现在去重写前端**——后端真实业务还没就位，前端再漂亮也没东西可显示。
5. **现在去给 mypy --strict 加更多严格度**（比如 strict_concatenate、ban Any）——HONEST_STATUS §4 已经标好"待收紧"清单，按那个顺序来，先把核心物理跑对再去抠这个。

---

## 最近一周的"敲门砖"

如果我**只能做一件事**：跑通 1.1 + 1.2 的"半波偶极子 73+j42 Ω"测试。

这一件事满足后：

- 证明 openEMS adapter 真实可信；
- 给后续每一个 adapter 改进提供回归基准；
- 一旦真值数据开始流，B 阶段的数据集生成就可以自动化跑起来。

**这是从"骨架"到"产品"的真正分水岭。**
