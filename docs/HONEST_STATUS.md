# YAF 真实状态报告（HONEST_STATUS）

> 这份文档刻意不美化。如果你看到 6 条验收命令全绿就以为
> 系统已经能"发明天线"，请先读完这份文件。

写作日期：2026-05-21（初版）；**2026-05-24 大幅修订** —— NEC2 路径已
从"全部走 fallback"升级为"真实 necpp MoM in the loop"，AI 模块的实际接入情况按
"真实接入 / Demo only / 占位/死代码"三档重新逐项标注。所有论断都对应到具体源码
行号 + commit 号，可以核对。

**2026-05-25 修订** —— openEMS 路径从"解析降级占位（RLC fallback）"升级为
"真实 full-wave FDTD in the loop"：适配器现在通过 openEMS Python 绑定真正建模、
跑时域迭代、从端口提取 S11/输入阻抗、用 NF2FF 提取增益方向图；解析 fallback 已
删除，绑定缺失 → `SolverUnavailable`。新增 `scripts/verify_patch.py` 矩形微带贴片
真值校验：实测谐振 2.435 GHz vs 解析 2.513 GHz，误差 3.1%，落在 ±10% 内。

## 0a. 待法律审查项（pending legal review）

公开发布前**必须**由合格法务复核：

1. **`yaf_solvers/nec2_adapter/adapter.py` 通过 `import necpp` 加载 GPL-2 的 nec2++ C++ 扩展进入同一 Python 进程**（NEC2 适配器改用 necpp 后切换到此路径）。FSF 通行解释是 GPL 在共享进程内构成"组合作品"。当前仓库的应对策略是：
   - 不在 `pyproject.toml` 里把 `necpp` 列为硬依赖；
   - 用户自行 `pip install necpp`，本地组合由用户自行承担；
   - 不在仓库里分发任何 necpp 源码或编译产物；
   - 在仓库根目录 `NOTICE` 文件里把上述边界、风险、下游再分发者的缓解建议全部写明。
   这套定性是发布前**最重要**的法务复核项。下游再分发者（要打 Docker 镜像 / wheel / 二进制）应另请律师。
2. **`yaf_solvers/openems_adapter/adapter.py` 现在通过 `from openEMS import openEMS` / `from CSXCAD import ContinuousStructure` 把 GPL-3 的 openEMS 绑定加载进同一 Python 进程**（2026-05-25 起，已不再是 fallback）。与 necpp 同样的"共享进程组合作品"定性适用，所以这一项**已被触发**，须和上面 NEC2 那条一并复核。当前应对策略与 necpp 一致：不在 `pyproject.toml` 里列为硬依赖、用户自行编译安装 openEMS、仓库不分发其源码或编译产物。
3. 一个工程层面的缓解方案是恢复"subprocess 调 `nec2c`"的可选后端——FSF 通常把纯 subprocess 划入"mere aggregation"而非组合作品。这条路径在 NEC2 适配器改写时被删掉以换取性能，下个 minor release 可以补回来作为 GPL 隔离选项。

详见 `NOTICE`。本节中文摘录仅为内部备忘，**不构成法律意见**。

---

## 0. TL;DR（2026-05-24 修订）

| 维度 | 旧状态（2026-05-21） | 新状态（2026-05-24 之后） |
|---|---|---|
| NEC2 适配器真实性 | D（全部走 fallback） | **A**（真实 `necpp` MoM；fallback 已删除；missing necpp → `SolverUnavailable`） |
| 已知答案物理回归测试 | C−（基本没有） | **B**（半波偶极子真值；逆向设计与 Yagi 案例都有 NEC2 验证） |
| AI 模块在物理评测下的有效性 | D（没接物理 oracle） | **C+**（Yagi case：scipy DE × 真实 NEC2 跑出 +4 dB 的工程意义结果；但生成式 VAE/扩散仍未接入物理 oracle） |
| openEMS 适配器真实性 | D（fallback） | **B+**（2026-05-25：真实 full-wave FDTD；fallback 已删；缺绑定 → `SolverUnavailable`；通过贴片 ±10% 真值，但目前仅 1 个已知答案案例） |
| FNO/DeepONet 代理模型 | D（死代码，未训练） | D（仍是；ADR-013 决定本轮不接活，先把状态写准）|

下面按模块给细节。

---

## 1. 求解器适配器：哪些是真的，哪些是 mock

### 1.1 NEC2 (`yaf_solvers/nec2_adapter/`) — **🔄 大幅升级**

| 项 | 旧状态 | 新状态（2026-05-24） |
|---|---|---|
| **NEC 卡片生成 (`card_writer.py`)** | ✅ 真实，3 个单测 | ✅ 真实（保留作 `.nec` 文件导出，但 solve 不走文本路径） |
| **求解器后端** | subprocess 调 `nec2c`，failure → fallback | ✅ **真实 `necpp` Python 绑定**（直接调 nec_create / nec_wire / nec_geometry_complete / nec_fr_card / nec_ex_card / nec_rp_card / nec_impedance_real/imag / nec_gain），见 commit `efe5cc0`。 |
| **fallback `_compute_analytical`** | ❌ 硬编码 gain=2.15 | ✅ **已删除**。`necpp` 装不上 → `SolverUnavailable`；mesh 没 wire → `SolverError`。**没有任何"安静返回假值"的代码路径还剩下**。 |
| **远场方向图** | ❌ 解析 `cos(π/2 cosθ)/sinθ` 占位 | ✅ **真值**：commit `d9a0b26` 把 `nec_gain(., 0, θ, φ)` 的每方向增益反求出等效 `|E_θ|` 存入 `FarFieldResult`，`gain_dbi()` 现在拿到的就是 NEC2 自己算出的 dBi。 |
| **半波偶极子真值回归** | 无 | ✅ **`scripts/verify_dipole.py`**：300 MHz 半波偶极子，R = 68.30 Ω（vs 教科书 73 Ω，误差 6.4%），G = 2.12 dBi（vs 2.15），谐振过零正确。`pytest tests/unit/test_solvers.py::test_dipole_solve_real_nec` 强制 ±15% 容差。 |
| **逆向设计真值**| 无 | ✅ **`scripts/demo_inverse_design.py`**：黄金分割搜索 16 次真实 NEC2 求解收敛到 L = 477.892 mm（≈ 0.478 λ），R = 71.85 Ω、X = +0.03 Ω。**优化器独立复现教科书谐振长度。** |
| **5-参数 Yagi 优化** | 无 | ✅ **`scripts/case_yagi.py`**：9 参数 Yagi-Uda，scipy DE × 5858 次真实 NEC2 评估在 12.7 s 内跑出 G = 12.63 dBi，比 Balanis 3-elem 基线 +4.03 dB，并独立涌现出 Viezbicke 式 director 锥度。 |

**`scripts/demo_dipole.py`** 的输出现在是真实 NEC2 MoM 仿真（2026-05-24 起）：
2.45 GHz 自由空间半波偶极子 S11 ≈ −8.56 dB，Gain = 2.20 dBi，VSWR = 2.59。
这些数字依赖你喂进去的几何 —— 改单元长度就会改 S11，不再是硬编码。

### 1.2 openEMS (`yaf_solvers/openems_adapter/`) — **🔄 大幅升级（2026-05-25）**

| 项 | 旧状态 | 新状态（2026-05-25） |
|---|---|---|
| **求解器后端** | `import openems`（错误的小写模块名，永远 import 失败）→ 每次走 fallback | ✅ **真实 openEMS full-wave FDTD**：`from openEMS import openEMS` / `from CSXCAD import ContinuousStructure`，`solve()` 真正建 CSX 结构（metal/material 盒）、`AddEdges2Grid` + `SmoothMeshLines` 细化网格、`AddLumpedPort` 激励、`Run()` 跑时域迭代。 |
| **fallback `_run_analytical`（RLC 占位）** | ❌ 一阶 RLC：`s11 = detuning/(detuning + 1j·0.1)`，几乎不带几何依赖 | ✅ **已删除**。openEMS 绑定装不上 → `SolverUnavailable`；没有 structures/ports → `SolverError`。**没有任何"安静返回假值"的代码路径还剩下**（与 NEC2 一致）。 |
| **S11 / 输入阻抗** | ❌ 来自 RLC 占位 | ✅ **真值**：`LumpedPort.CalcPort` → `uf_ref/uf_inc` 得 S11、`uf_tot/if_tot` 得 Zin，全频段一次时域迭代得到。 |
| **远场 / 增益方向图** | ❌ `sin(θ)` 占位 | ✅ **真值**：`CreateNF2FFBox` + `CalcNF2FF` 近场转远场，`Dmax → 10·log10(Dmax)` dBi，E 场方向图存入 `FarFieldResult`。 |
| **CSXCAD XML 序列化 (`to_native_format`)** | ✅ 真实但只是几何 | ✅ 不变：发出有效 `<ContinuousStructure>` XML（metal box 几何），作为轻量序列化辅助；真正 solve 不走这条文本路径，直接用绑定建 CSX。 |
| **矩形微带贴片真值回归** | 无 | ✅ **`scripts/verify_patch.py`**：Rogers RO4003C 贴片（εr=3.38、h=1.524 mm、L=32 mm、W=40 mm），实测谐振 **2.435 GHz** vs 腔模解析公式 **2.513 GHz**，误差 **3.1%**（容差 ±10%），S11 谷 −27 dB、Zin 46.2−1.9j Ω、方向性 6.81 dBi、103635 网格、12 s 收敛。`pytest tests/integration/test_pipeline.py::test_solver_openems_integration` 跑一个缩小版真实贴片（无绑定时 skip）。 |

> 仍诚实标注的边界：目前只有**一个**已知答案校验案例（贴片谐振频率），不像 NEC2
> 那条线有偶极子 + 逆向设计 + Yagi 三重验证；增益按方向性 `10·log10(Dmax)` 给出
> （未单独扣除失配/欧姆损耗，故为方向性而非含失配的 realized gain，已在 metadata
> 里以 `directivity_dbi` 标注）；只验过单端口 lumped 激励，MSL/波导端口未走真值。

### 1.3 MEEP / HFSS / CST / FEKO / COMSOL

全部是 **skeleton**，`solve()` 直接返回 `status="skeleton_not_implemented"`（5 个 `adapter.py` 共 ~17 行内容）。集成测试里它们不会被调到，所以不影响 47/47 pytest 绿。

### 1.4 `MaterialLibrary.get_dispersive_permittivity`

- Drude / Debye 闭式公式 ✅ 与文献一致
- Kubo 公式 (`_kubo_conductivity`) ⚠️ 只实现了 Hanson 公式的简化形式，没和参考实现（fdtdx 的 `dispersion.py` 或 gprMax 的多极模型）做过数值对照

---

## 2. demo_dipole.py：S11 / 增益是不是物理真值？

**是（2026-05-24 起）。** 当前输出走的是 §1.1 的真实 `necpp` MoM 路径：

- `Gain = 2.20 dBi`、`Peak gain = 2.20 dBi` —— 两个数字都来自真实 NEC2 计算，
  互相一致。改单元长度会改 gain，不再是硬编码。
- `S11 = −8.56 dB / VSWR = 2.59` —— 真实 NEC2 + `(Z - 50)/(Z + 50)`，反映了
  自由空间半波偶极子的实际阻抗（73 + j42 Ω 附近）相对 50 Ω 的失配。**这是真值，
  和教科书一致；想 −10 dB 需要加匹配网络。**
- `Best S11: −8.56 dB @ 2.400 GHz` —— 真实扫频结果。

**评级**：管线跑通 ✅，物理可信度 ✅（NEC2 路径；openEMS 路径自 2026-05-25 起也是真实 full-wave，见 §1.2）。

---

## 3. AI 模块：三档真实接入度

每个模块用三档评级：
- **🟢 真实接入**：跑过真实物理 oracle、产出可被 NEC2/openEMS 复核的结果。
- **🟡 Demo only**：自洽 demo 能跑、合成数据训练、但**没和真实物理评测连起来**。
- **🔴 占位/死代码**：写了类，但 pipeline 里没人调用，或权重是随机初始化。

| 模块 | 评级 | 1 句话现状 |
|---|---|---|
| `yaf_ai/optimization/bayesian.py` (GP + EI) | 🟡 → 🟢* | 2D Branin 玩具能跑；本轮没用它（DE 在 9-D 上更强，见 ADR-012）。*但黄金分割逆向设计虽不是 BO，**思路一致**：真实 NEC2 在优化回路里。 |
| `yaf_ai/inverse_design/` Yagi case | **🟢** | **scripts/case_yagi.py**：5858 次真实 NEC2 评估 → +4 dB / 12.7 s。这是当前仓库**唯一**完整的 AI × 真实物理 oracle 闭环（见 docs/case_study_yagi.md）。 |
| `yaf_ai/differentiable/diff_fdtd_jax.py` | 🟡 | 梯度真的能反传（bug 已修），但 2D TM + 简化 PML 的玩具网格；用作"管线可微的存在性证明"，不能用来设计真实天线。 |
| `yaf_ai/generative/vae_designer.py` | 🟡 | β-VAE 训练能收敛，但**训练数据是合成几何，没有任何 S11/gain 标签**；`generate()` 出来的样本没接物理评估。 |
| `yaf_ai/generative/diffusion.py` | 🟡 | DDPM 架构正确、能采样；同 VAE，**未接物理 oracle**。 |
| `yaf_ai/surrogate/fno_solver.py` | **🔴** | FNO 架构搭好了，**权重是随机初始化，没训过、pipeline 没人调用**。ADR-013 决定本轮不接活（理由：要可靠的 FNO 代理需要数千样本训练 + 主动学习策略，现有预算做不出可信版本）。 |
| `yaf_ai/surrogate/deeponet.py` | 🔴 | 同 FNO，未训未用。 |
| `yaf_ai/inverse_design/pipeline.py`（六阶段框架） | 🟡 | 六阶段 `generate → screen → refine → topo → verify → score` 能跑通。`verify` 现在调的是真实求解器（先试 openEMS、失败再退 NEC2，两者都已无解析降级、不会伪造结果），但 pipeline 默认喂的几何是 VAE 出的 2D 二值栅格图——既不是 openEMS 能直接仿的结构化盒 + 端口，也不是 NEC2 的线天线，所以**生成式这条路目前还没真正接上物理 oracle**。缺的是"VAE 输出 → 可仿真几何"的转换层，而**不是**求解器在造假。Yagi case 那条 +4 dB 真值路径绕开了本管线。 |
| `yaf_ai/optimization/nsga.py` (NSGA-II) | 🟡 | ZDT1 玩具能跑出 Pareto front；本轮没选它（单目标 + 约束，DE 更合适，见 ADR-012）。 |
| `yaf_ai/optimization/topology_opt.py` (SIMP) | 🟡 | SIMP + OC 更新规则；demo 在 2D 合成 compliance 问题上能跑，**没接电磁仿真**。 |

### 关键的"诚实化"修订

- 旧 §3 的"3.3 FNO 是死代码"评价仍然成立，但作了**明确决策记录**：见 ADR-013。
  我们不假装 FNO "马上要接活"——它**至少需要一轮专门的数据收集 + 训练 +
  主动学习封装**才有意义，**本轮没做**。
- 旧 §3 的"6 阶段 pipeline `verify` 走 NEC2 fallback" —— 这句话**仍然对**：
  `yaf_ai/inverse_design/pipeline.py` 当前调的 `NEC2Adapter` 虽然现在是真值
  路径，但 pipeline 默认喂的几何是 VAE 出的 2D 二值栅格图，**不是**线天线，
  所以送进去也会被 `SolverError("no wire elements found in mesh")` 拒掉。
  Pipeline 要真正工作还需要一个"VAE 输出 → wire 几何"的转换层 —— 这是后续的活。
- Yagi case **没有走** `pipeline.py` 那条 6 阶段管线，而是
  **绕过它直接用 scipy DE × NEC2 adapter**，因为 6 阶段 pipeline 当前的设计
  不适合 "9-D 参数化几何 + 单一标量目标"这个具体场景。这点对读者应该说清。

---

## 4. mypy --strict 通过的真实代价

`pyproject.toml` 里设置了：

```toml
[tool.mypy]
strict = true
warn_return_any = false
disallow_untyped_calls = false
disallow_subclassing_any = false
warn_unused_ignores = false
```

**注意：实际上 CLI `--strict` 把上面四个开关都重新打开了**（mypy 2.1 的行为）。它们在 pyproject 里之所以还留着，是为了**让不带 `--strict` 的 `mypy` 调用也能尽量贴近 strict** —— 不是真正的"降标"。验收命令 `mypy ... --strict` 跑过靠的是逐处加 `cast()` / `float()` / `complex()` / `np.asarray()` 把 `Any` 收窄掉，以及 `# type: ignore[no-untyped-call]` 标注 PyTorch `.backward()`。

### 4.1 `[[tool.mypy.overrides]]` 屏蔽的模块

```toml
module = [
    "OCC.*", "trimesh.*", "gmsh.*", "skrf.*",
    "openems.*", "CSXCAD.*", "necpp.*",
    "qdrant_client.*", "minio.*", "celery.*",
]
ignore_missing_imports = true
```

逐条评级：

| 模块 | 现状 | 评级 |
|---|---|---|
| `OCC.*` (pythonocc-core) | 上游无类型 stub，使用面很窄（只在 `_check_occ` 里 import 试探）| **无害** |
| `openems.*` | Cython 绑定，上游无 stub。代码 import 完只读 `_openems_available` flag。| **无害** |
| `CSXCAD.*` | 同 openems。| **无害** |
| `necpp.*` | C 绑定，上游无 stub。当前代码**没真正调用** necpp Python 接口，只是 `nec2c` subprocess。| **无害**（但应等真正用 necpp 时一起收紧）|
| `gmsh.*` | 上游有 stub，但版本多变；代码里没主动 import gmsh。`pyproject` 依赖列了但没用上。| **待收紧**（或者把 gmsh 从依赖里删掉）|
| `trimesh.*` | 上游 stub 不完整。代码用 `trimesh.creation` 等 API。`types-trimesh` 不存在。| **待收紧** —— 应该把 trimesh 调用收到一个薄 wrapper 里、wrapper 显式 annotate 返回值。|
| `skrf.*` | scikit-rf 自带 `py.typed` 标识，但 mypy 仍然不完美。当前仓库只在 `SParamResult.from_touchstone` 一处用到。| **待收紧** —— 关键路径，应至少给 `skrf.Network` 写一个 minimal stub。|
| `qdrant_client.*` | 有自己的类型；屏蔽是图省事。| **待收紧** |
| `minio.*` | 有自己的类型；屏蔽是图省事。| **待收紧** |
| `celery.*` | 现代 celery 已经有 py.typed。屏蔽是图省事。| **待收紧** |

### 4.2 散落的 `cast(...)` 和 `# type: ignore[...]`

- `cast(torch.Tensor, self.decoder(z))` 等共 ~6 处。**无害**：PyTorch nn.Module `__call__` 返回 Any 是上游事实。
- `# type: ignore[no-untyped-call]` 标注 `.backward()` 共 4 处。**无害**：同上。
- `# type: ignore[arg-type]` 一处在 `space_time.py:114` 给 `float(abs(bessel_j(...)))` —— **待收紧**：可以走 `scipy.special.jv` 的实数路径或者改成 `numpy.asarray` 后取 `.item()`。
- `# type: ignore[import-not-found, unused-ignore]` 一处在 `kernel.py:28` 给 OCC import。**无害**（OCC 没 stub）。

### 4.3 已知"用 `Any` 偷懒"的地方

- `yaf_ai/generative/vae_designer.py::get_dataloader -> Any`：返回 `DataLoader[tuple[Tensor]]` 类型化不成（TensorDataset 不是泛型 Dataset 子类），所以兜底返 `Any`。**待收紧**：写一个 `class _BatchTensorDataset(Dataset[tuple[torch.Tensor]])`。
- `yaf_core/geometry/parametric.py::_subdivide(v0: Any, ...)` —— 调用方有时传 list[float] 有时传 np.ndarray。**待收紧**：在调用方统一 cast 成 list。

### 4.4 一句话总结

> mypy --strict 通过这件事**真实可信**。代价主要落在两类：
> (1) **真无害**：少数几个 PyTorch / OpenCASCADE / openems 接口，上游就没类型，没办法；
> (2) **待收紧但不阻塞物理**：Celery / Qdrant / MinIO / skrf / trimesh / gmsh 这几个有类型却被忽略掉了——后续收紧后能多发现一些误用，但不影响当前管线行为。
>
> **没有任何"为了让验收过、把核心物理逻辑里的类型偷换掉"的情况**。

---

## 5. 测试的真实强度

`pytest tests/ -x -q → 47 passed` 这一行非常容易被误读为"47 个真实场景验证通过"。实际分布：

| 文件 | 案例数 | 多少是"结构断言/能跑就过"，多少是"对解析解" |
|---|---|---|
| `tests/unit/test_domain.py` | 13 | **0 个**对解析解。全部是 Pydantic 字段存在、状态机迁移、序列化反序列化能 round-trip。 |
| `tests/unit/test_geometry.py` | 8 | **0 个**对解析解。检查 `num_vertices > 0` / `num_faces > 0` / `box.volume == 100`。`make_box` 期望 8 个顶点 12 个面——这是 BREP→mesh 拓扑断言，没验几何正确性。 |
| `tests/unit/test_physics.py` | 9 | **1 个半**：`test_copper` 验证 sigma=5.8e7 但那是 seed 值；`test_ris_element` 验证 2-bit RIS 4 个状态、相位 0/90/180/270 ✅；其余 `assert isinstance(eps, complex)` 类型断言、`af.shape == (37, 73)` 形状断言。 |
| `tests/unit/test_solvers.py` | 8 | **0 个**对解析解（结构断言为主）。检查 NEC 卡片字符串里有 `"GW"`/`"GE"`、OpenEMS XML 里有 `"ContinuousStructure"`、`status == "success"`、`gain_dbi is not None`；外加 NEC2 与 openEMS 各一个"空几何必须抛错、绝不伪造结果"的诚实性断言。真正的 openEMS ±10% 真值在 `scripts/verify_patch.py`（脚本，不在 pytest 里）。 |
| `tests/integration/test_api.py` | 2 | `/health` 返 200 + `{"status":"ok"}`。**不验证业务逻辑**。 |
| `tests/integration/test_pipeline.py` | 3 | `loop_count >= 1`、`len(s_params.frequency) == 21`、`gain_dbi is not None`。openEMS 那条现在跑的是**真实 full-wave** 缩小贴片（无绑定时 skip），但断言仍是结构层（频点数、gain 非空、backend 名），**不在这里**和参考值对比——数值真值留给 `scripts/verify_patch.py`。 |

**总评**：47 个 pytest 测试里**真的在断言物理/几何"对不对"的仍是少数**（`test_ris_element`、`test_bounding_box`，以及 NEC2 偶极子单测里的 R/gain 容差断言），其余多数是"管线跑通 + 结构"断言。这不是说它们没价值——这种"smoke + 结构"层的测试能挡住空指针、null 字段、API 签名漂移——但**它们无法替代"对照解析解/HFSS 真值的回归测试"**。真正的已知答案校验放在脚本里：`scripts/verify_dipole.py`（NEC2）与 `scripts/verify_patch.py`（openEMS）。

已经补上的"已知答案"校验：
- ✅ 半波偶极子在自由空间 ~73 Ω（`verify_dipole.py`，真实 NEC2）
- ✅ 矩形微带贴片谐振频率对腔模解析公式 ±10%（`verify_patch.py`，真实 openEMS：实测 2.435 GHz vs 解析 2.513 GHz，误差 3.1%）

仍待补：
- 2-bit RIS 阵列在指定相位码本下主瓣方向（用 array factor 验）
- 贴片输入阻抗/带宽对参考值的回归（目前只验了谐振频率，没把 Zin/−10 dB 带宽纳入断言）

---

## 6. 其它"全绿之下"的隐患

1. **docker-compose 启动的 5 个服务**：postgres / redis / minio / qdrant 都健康，但 **API container 不和它们任何一个真正交互** —— `yaf_api/main.py` 的 `lifespan` 是空 startup/shutdown，路由里用的是内存 dict（ADR-006），所以 docker compose 健康也只代表"五个进程都活着"，不代表数据流跑通了。
2. **Frontend** Dockerfile 没被 build 过（验收只跑了 api）；`frontend/src` 里有 `DesignEditor.tsx` / `ThreeViewer.tsx` 等，但没在浏览器里点击过验证。
3. **Worker（Celery）**也没启动过；`yaf_worker/tasks/simulate.py` 的任务不在自动化覆盖范围内。
4. **`models/vae_designer.pt`** 是个真实写到盘上的文件，5.5 MB —— 但训练只跑 2 epoch 是为了过验收，不是产生有用权重。
5. **`pyproject.toml` 里的 `gmsh` 依赖** 安装失败也没影响测试，因为代码里没 import 它。建议要么真用、要么删。
6. **Python 版本**：pyproject 写 `requires-python = ">=3.11"`，本机跑的是 3.13，跑通了——但 jax/jaxlib 0.10、torch 2.12+cpu 是 3.13 的新版本，和默认假设的"3.11 + JAX 0.4.30 / torch 2.4"组合实际偏移很大。这意味着把这套代码搬到 Linux/3.11 时**有 5%–10% 的概率会撞到 API 不兼容**（比如 jax pytree 接口变化）。

---

## 7. 一句话评级（2026-05-24 修订）

| 维度 | 旧评级 | 新评级 | 注 |
|---|---|---|---|
| 项目骨架完整度 / Pydantic 领域模型 | A | A | 无变化 |
| 求解器适配器接口设计（Protocol） | A− | A− | 无变化 |
| **NEC2 求解器物理可信度** | D | **A** | 自 NEC2 适配器改写起真实 necpp MoM；fallback 已删 |
| openEMS 求解器物理可信度 | D | **B+** | 2026-05-25：真实 full-wave FDTD；fallback 已删；通过贴片 ±10% 真值；目前仅 1 个已知答案案例、增益按方向性给出 |
| MEEP/HFSS/CST/FEKO/COMSOL 适配器 | skeleton | skeleton | 未动 |
| 可微 FDTD 实现复杂度 vs 论文级 | C | C | 未动 |
| AI 生成模型架构（VAE/Diffusion） | B | B | 未动 |
| **AI 生成模型在真实物理评测下的有效性** | D | C+ | Yagi case 是 AI × 真实 NEC2 闭环的**存在性证明**；生成式那条线仍未接 |
| FNO/DeepONet 代理模型 | D（未训未用） | D（未训未用） | ADR-013 明确决定本轮不接活 |
| 已知答案物理回归测试 | C− | **B** | `verify_dipole.py` + 单元测试 + 集成测试都断言真实物理量 |
| mypy --strict | B+ | B+ | 未动 |
| 一键 docker compose / 健康检查 | A | A | 未动 |
| 距离"能用 YAF 发明出可制造的真天线" | **远** | **稍近**：Yagi 案例是工程意义上可行的方向选择（DE × NEC2），贴片现在有真实 full-wave 求解器可用；但仍只覆盖线天线参数化设计 + 单个贴片真值，3D / 表面贴片优化 / 拓扑级别还是远 |

### 一句话总结

> **2026-05-21 评级"远"主要因为所有求解器都走 fallback、AI 模块没接物理 oracle。**
> **2026-05-24 后 NEC2 路径已经是真值，并且有了一个 +4 dB 的 Yagi 工程意义案例。**
> **2026-05-25 后 openEMS 路径也是真实 full-wave 了，贴片谐振频率对解析公式 ±10%（实测 3.1%）。**
> 距离"发明真天线"依然远，但**两条求解器主线（NEC2 线天线 MoM、openEMS full-wave
> FDTD）现在都能跑真值**：`python3 scripts/case_yagi.py`、`python3 scripts/verify_patch.py`
> 各是一行命令的事。生成模型 / 代理模型这两条线还需要分别下力气。
