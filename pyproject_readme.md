# pyproject.toml 配置解析

> RF-DETR 项目的核心配置文件，涵盖构建系统、依赖管理、代码质量工具（lint/format/type-check）、测试框架等所有开发与发布配置。

---

## 1. `[build-system]` — 构建系统声明

```toml
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"
```

| 字段 | 用途 |
|------|------|
| `requires` | 声明构建项目所需的 PEP 517 构建依赖（`setuptools` + `wheel`） |
| `build-backend` | 指定构建后端为 `setuptools.build_meta`，即使用 setuptools 来执行实际的构建过程 |

---

## 2. `[project]` — 项目元数据（PEP 621）

```toml
[project]
name = "rfdetr"
version = "1.8.0"
description = "RF-DETR"
readme = "README.md"
authors = [{name = "Roboflow, Inc", email = "develop@roboflow.com"}]
license = {text = "Apache License 2.0"}
requires-python = ">=3.12,<3.13"
```

| 字段 | 用途 |
|------|------|
| `name` | PyPI 包名，用户通过 `pip install rfdetr` 安装 |
| `version` | 当前版本号（1.8.0） |
| `description` | 包简短描述 |
| `readme` | 指向 `README.md` 作为 PyPI 项目描述页 |
| `authors` | 作者/维护者信息（Roboflow, Inc.） |
| `license` | 许可证类型：Apache License 2.0 |
| `requires-python` | Python 版本约束：仅支持 3.12.x |

### `dependencies` — 核心运行时依赖

```toml
dependencies = [
    "requests",                        # 从远程 URL 下载模型权重
    "numpy",                           # 推理、评估、导出、可视化中的数组转换
    "torch>=2.11.0",                   # 核心张量运算与模型前向传播
    "torchvision>=0.26.0",             # 图像变换与操作（与 torch>=2.11.0 对齐）
    "tqdm",                            # 权重下载进度条
    "transformers>=5.1.0,<6.0.0",     # DINOv2 backbone 加载
    "pydantic>=2.0,<3",               # ModelConfig / TrainConfig 配置校验
    "supervision>=0.29.0",             # 推理输出（Detections, Masks）
    "pyDeprecate>=0.9,<0.10",         # 旧 API 的弃用警告
]
```

这些是 `pip install rfdetr` 时**始终安装**的核心依赖。

---

## 3. `[project.optional-dependencies]` — 可选依赖组

用户可通过 `pip install rfdetr[组名]` 按需安装。

| 组名 | 用途 | 关键依赖 |
|------|------|----------|
| `lora` | LoRA 低秩微调 | `peft` |
| `train` | 模型训练 | `peft`, `pytorch_lightning`, `torchmetrics`, `pycocotools`, `albumentations`, `roboflow` |
| `onnx` | ONNX 模型导出 | `onnx`, `onnxsim`, `onnx_graphsurgeon`, `onnxruntime`, `polygraphy` |
| `trt` | TensorRT 推理加速 | `pycuda`, `onnxruntime-gpu`, `tensorrt`, `polygraphy` |
| `tflite` | TFLite 模型导出 | `onnx2tf`, `flatbuffers`, `tf-keras`, `tensorflow` |
| `kornia` | GPU 端数据增强 | `kornia` |
| `loggers` | 训练日志记录 | `tensorboard`, `wandb`, `mlflow`, `clearml` |
| `visual` | 可视化与分析 | `matplotlib`, `pandas`, `seaborn` |
| `cli` | 命令行接口 | `jsonargparse` |
| `plus` | Plus 模型支持（PML 许可证） | `rfdetr_plus` |

---

## 4. `[dependency-groups]` — 开发依赖组（uv 专属）

与可选的 extras 类似，但仅用于开发环境，通过 `uv sync --group 组名` 安装。

| 组名 | 用途 | 关键依赖 |
|------|------|----------|
| `build` | 构建与发布 | `twine`, `wheel`, `build` |
| `tests` | 测试框架 | `pytest`, `pytest-cov`, `pytest-xdist`, `pytest-rerunfailures`, `pytest-timeout` |
| `typing` | 静态类型检查 | `mypy`, `numpy<2.4`, `types-PyYAML`, `types-requests`, `types-tqdm` |
| `docs` | 文档构建 | `mkdocs-material`, `mkdocstrings`, `mkdocs-jupyter`, `mike` |

---

## 5. `[project.urls]` — 项目链接

```toml
[project.urls]
Homepage = "https://github.com/roboflow/rf-detr"
```

PyPI 页面上显示的项目主页链接。

---

## 6. `[project.scripts]` — CLI 入口点

```toml
[project.scripts]
rfdetr = "rfdetr.cli:main"
```

安装后，用户可在终端直接运行 `rfdetr` 命令，它会调用 `rfdetr.cli` 模块中的 `main()` 函数。

---

## 7. `[tool.uv]` — UV 包管理器配置

```toml
[tool.uv]
override-dependencies = [
    "ml-dtypes==0.5.1; python_version >= '3.12' and python_version < '3.13'",
]
```

| 配置 | 用途 |
|------|------|
| `override-dependencies` | 强制覆盖传递依赖的版本，解决 `onnx2tf` 的 `ml-dtypes` / `numpy` 版本冲突 |

### `[tool.uv.sources]` — 自定义包来源

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cu130", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }]
torchvision = [{ index = "pytorch-cu130", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }]
```

指定 `torch` 和 `torchvision` 从 PyTorch 官方 CUDA 13.0 索引安装，而非默认 PyPI（PyPI 上的 torch 是 CPU 版本）。

### `[[tool.uv.index]]` — 自定义包索引

```toml
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

| 字段 | 用途 |
|------|------|
| `name` | 索引名称，供 `tool.uv.sources` 引用 |
| `url` | PyTorch CUDA 13.0 的 wheel 托管地址 |
| `explicit = true` | 只有显式声明 `index = "pytorch-cu130"` 的包才从此索引获取 |

---

## 8. `[tool.setuptools]` — Setuptools 打包配置

### 包发现

```toml
[tool.setuptools.packages.find]
where = ["src"]
include = ["rfdetr*"]
```

在 `src/` 目录下查找所有以 `rfdetr` 开头的包（如 `rfdetr`, `rfdetr.models`, `rfdetr.training` 等）。

### 包数据

```toml
[tool.setuptools]
include-package-data = false

[tool.setuptools.package-data]
rfdetr = ["py.typed", "models/backbone/dinov2_configs/*.json"]
```

| 配置 | 用途 |
|------|------|
| `include-package-data = false` | 默认不包含非 Python 文件 |
| `[tool.setuptools.package-data]` | 显式声明要打包的非代码文件：`py.typed`（PEP 561 类型标记）和 DINOv2 配置文件 |

---

## 9. `[tool.ruff]` — Ruff Linter & Formatter 配置

```toml
[tool.ruff]
fix = true
line-length = 120
target-version = "py310"
```

| 配置 | 用途 |
|------|------|
| `fix = true` | 自动修复可修复的 lint 问题 |
| `line-length = 120` | 每行最大 120 字符 |
| `target-version = "py310"` | 以 Python 3.10 语法为标准 |

### `[tool.ruff.lint]` — Lint 规则选择

```toml
[tool.ruff.lint]
select = ["E", "W", "F", "I", "N"]
extend-select = ["RUF100"]
ignore = ["E722"]
```

| 规则代码 | 规则集 | 用途 |
|----------|--------|------|
| `E` | pycodestyle errors | 代码风格错误 |
| `W` | pycodestyle warnings | 代码风格警告 |
| `F` | pyflakes | 逻辑错误检测（未使用变量等） |
| `I` | isort | 导入排序 |
| `N` | pep8-naming | 命名规范（如小写变量 N806） |
| `RUF100` | yesqa | 检测无用的 `# noqa` 注释 |
| `E722` (忽略) | bare-except | 允许裸 `except`（暂不强制） |

### `[tool.ruff.lint.per-file-ignores]` — 文件级忽略

```toml
[tool.ruff.lint.per-file-ignores]
"notebooks/*.py" = ["E402"]
"docs/cookbooks/*.ipynb" = ["E402", "E501"]
```

- `E402`（模块级导入不在文件顶部）：Jupyter notebook 中 cell 内导入是有意为之
- `E501`（行过长）：notebook 中的 markdown 文本不需要自动换行

---

## 10. `[tool.docformatter]` — Docstring 格式化

```toml
[tool.docformatter]
wrap-summaries = 120
wrap-descriptions = 120
```

控制 docstring 的摘要和描述部分在 120 字符处自动换行。

---

## 11. `[tool.pytest.ini_options]` — Pytest 配置

### 默认选项

```toml
addopts = ["-v", "--color=yes", "--doctest-plus"]
```

| 选项 | 用途 |
|------|------|
| `-v` | 详细输出模式 |
| `--color=yes` | 强制彩色输出 |
| `--doctest-plus` | 运行所有模块中的 doctest |

### 搜索路径与标记

```toml
pythonpath = ["src"]
markers = [
    "gpu: tests that require GPU or are slow on CPU",
    "coco17: tests that require COCO 2017 images or annotations",
    "flaky: tests that may fail due to nondeterminism",
    "tflite: tests requiring onnx2tf and TFLite dependencies",
]
```

| 配置 | 用途 |
|------|------|
| `pythonpath` | 将 `src/` 加入 Python 搜索路径，使测试能直接 `import rfdetr` |
| `markers` | 自定义 pytest 标记，用于筛选/排除特定测试（如 `-m "not gpu"`） |

### 警告过滤

```toml
filterwarnings = [
    "ignore::pluggy.PluggyTeardownRaisedWarning",
    "ignore:.*was instantiated with pretrain_weights=None:UserWarning",
    "ignore:.*was instantiated with overrides that differ from the variant:UserWarning",
]
```

过滤掉测试中已知的、不影响结果的噪音警告（如 pytest-xdist 的 teardown 告警、预训练权重兼容性警告等）。

---

## 12. `[tool.codespell]` — 拼写检查

```toml
[tool.codespell]
skip = "*.pth"
```

跳过 `.pth` 文件（PyTorch 模型权重文件是二进制格式，拼写检查无意义）。

---

## 13. `[tool.mypy]` — Mypy 静态类型检查

```toml
[tool.mypy]
python_version = "3.10"
ignore_missing_imports = false
explicit_package_bases = true
strict = true
mypy_path = "src"
```

| 配置 | 用途 |
|------|------|
| `python_version = "3.10"` | 以 Python 3.10 类型系统为准 |
| `ignore_missing_imports = false` | 默认不忽略缺失的导入（迫使提供 stub 或显式声明忽略） |
| `explicit_package_bases = true` | `src/` 布局下正确推断包根目录 |
| `strict = true` | 开启 mypy 最严格模式 |
| `mypy_path = "src"` | 将 `src/` 加入 mypy 搜索路径 |

### 排除目录

```toml
exclude = ["^docs/(hooks|scripts)/"]
```

跳过文档辅助脚本的类型检查。

### 覆写规则（`overrides`）

分三类：

1. **测试/可视化库**（`tests.*`, `matplotlib`, `pandas`, `seaborn`）：完全忽略类型错误
2. **无 stub 的第三方库**（`torchvision`, `pycocotools`, `kornia`, `onnx*`, `tensorflow` 等）：仅忽略 `missing imports`
3. **项目内部待修复模块**（`rfdetr.*` 下大量模块）：暂完全忽略类型错误，待逐步修复

---

## 配置要点总结

| 类别 | 配置节 | 核心工具 |
|------|--------|----------|
| 构建与发布 | `[build-system]`, `[tool.setuptools]` | setuptools, wheel, twine |
| 依赖管理 | `[project]`, `[dependency-groups]`, `[tool.uv]` | uv |
| 代码格式化 | `[tool.ruff]`, `[tool.docformatter]` | ruff, docformatter |
| 代码检查 | `[tool.ruff.lint]`, `[tool.mypy]`, `[tool.codespell]` | ruff, mypy, codespell |
| 测试 | `[tool.pytest.ini_options]` | pytest |
