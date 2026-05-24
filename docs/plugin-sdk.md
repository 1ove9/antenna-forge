# YAF 插件 SDK 文档

## 1. 概念

YAF 的所有外部能力（求解器、AI 模型、CAD 引擎）都是**插件**。每个插件是一个 Python 包，包含一个实现对应 Protocol 的类和一个 `plugin.toml` 声明文件。

## 2. 插件类型

### 2.1 求解器插件 (Solver)

实现 `yaf_core.ports.solver_port.SolverAdapter` Protocol。

**必须实现的方法:**
- `name`, `version`, `supports` (类属性)
- `capabilities()` → dict
- `mesh(geometry, spec)` → Mesh
- `solve(mesh, spec, progress_callback)` → SimulationResult
- `to_native_format(geometry)` → bytes
- `from_native_result(raw_output)` → SimulationResult
- `cancel(job_id)`
- `health_check()` → bool

### 2.2 AI 插件 (AI Backend)

实现 `yaf_core.ports.ai_port.AIBackend` Protocol。

**task_type**: `"generate"` | `"surrogate"` | `"optimize"`

### 2.3 CAD 插件 (CAD Backend)

实现 `yaf_core.ports.cad_port.CADBackend` Protocol。

**supported_formats**: `["step", "stl", "iges", "gdsii"]`

## 3. plugin.toml 格式

```toml
[plugin]
id = "my-solver"
name = "My Custom Solver"
version = "0.1.0"
type = "solver"
entrypoint = "my_solver.adapter:MySolver"

[capabilities]
methods = ["fdtd", "mom"]
frequency_range = [1e6, 100e9]
gpu = true
```

## 4. 快速开发一个新求解器

```python
# my_solver/adapter.py
from yaf_solvers.base import BaseSolverAdapter
from yaf_core.domain.geometry import Geometry, Mesh
from yaf_core.domain.simulation import SimulationResult, SimulationSpec

class MySolver(BaseSolverAdapter):
    name = "my_solver"
    version = "0.1.0"
    supports = {"fdtd"}

    async def mesh(self, geometry, spec):
        # 实现网格生成逻辑
        ...

    async def solve(self, mesh, spec, progress_callback=None):
        # 实现求解逻辑
        ...

    def to_native_format(self, geometry):
        # 转换为求解器原生格式
        ...

    async def from_native_result(self, raw_output):
        # 解析求解器输出
        ...
```

## 5. 插件发现

插件放在 `plugins/` 目录下，系统启动时自动扫描并注册。通过 `plugin.toml` 的 `entrypoint` 字段定位适配器类。
