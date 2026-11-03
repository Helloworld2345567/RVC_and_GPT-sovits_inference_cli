# RVC 与 GPT-SoVITS 推理

一个仓库、一套 Python 隔离环境，提供 RVC 离线音色转换、人声分离，以及 GPT-SoVITS 文本转语音服务。两个项目保留各自目录、模型布局和原有命令。

| 目录 | 用途 | 入口与文档 |
| --- | --- | --- |
| `RVC/` | 音色转换、人声分离 | `rvc`、`python -m rvc`；[使用说明](RVC/README.md) |
| `GPT-SoVITS/` | 文本转语音 HTTP 服务 | `python api_v2.py`；[使用说明](GPT-SoVITS/README.md) |

## 安装共享环境

使用 **Python 3.12 x64**、FFmpeg 和 ffprobe。Windows 安装还需 [uv](https://docs.astral.sh/uv/getting-started/installation/)；首次安装需 Visual Studio C++ Build Tools 和 Windows SDK，用于编译 PyOpenJTalk。以下安装命令在本仓库根目录执行。

本机基线是 CUDA Toolkit **12.9.86**；CUDA 安装使用官方 `cu129` 源中的 **Torch/TorchAudio 2.8.0** 配套构建。其他运行库优先采用最新稳定版，必要兼容约束记录在 [`constraints.txt`](constraints.txt)，已验证环境的具体版本记录在 [`requirements-lock.txt`](requirements-lock.txt)。Torch wheel 自带所需 CUDA 运行库，运行时还需要兼容的 NVIDIA 驱动。

### Windows PowerShell

在 Visual Studio Installer 中安装“使用 C++ 的桌面开发”及 Windows SDK。安装器会自动发现 Visual Studio（包括 Preview）并启用 x64 编译环境；本机验证使用 MSVC 14.44。PyOpenJTalk 从源码编译以适配 NumPy 2，CMake、Cython 等构建依赖由安装工具自动准备。

```powershell
.\install.ps1 -Device CU129
. .\.codex\venv\Scripts\Activate.ps1
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

安装器统一管理根目录 `.codex/venv`，安装两套推理依赖和 RVC 命令，并准备 GPT-SoVITS 的语言资源。仅使用 CPU 时，将安装选项改为 `-Device CPU`。新终端只需重新激活这一个环境。

### Linux CPU

准备 Python 3.12、FFmpeg、libsndfile 与 C/C++ 构建工具，然后在根目录执行：

```bash
python3.12 -m venv .codex/venv
source .codex/venv/bin/activate
export PIP_CACHE_DIR="$PWD/.codex/pip-cache"
export NLTK_DATA="$PWD/GPT-SoVITS/.codex/runtime-cache/nltk_data"
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -c constraints.txt
python -m pip install --no-deps -e RVC
python scripts/prepare_language_resources.py
python -m pip check
```

## 使用

激活共享环境后，在 RVC 目录执行原有命令：

```powershell
Set-Location RVC
rvc --help
rvc separate list
rvc convert --model assets/weights/voice.pth --input input.wav --output output.wav --index-rate 0 --device cuda
Set-Location ..
```

准备 GPT-SoVITS 模型和配置后，在其目录启动服务：

```powershell
Set-Location GPT-SoVITS
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_local.yaml
```

`tts_local.yaml` 的准备方法见 [GPT-SoVITS 快速开始](GPT-SoVITS/README.md)。服务默认只监听本机，生成音频使用原有 `POST /tts` 接口。两个程序分别启动进程，共用同一 Python 解释器及依赖。

## 角色模型目录

角色模型统一放在 `models/<服务>/<角色 ID>/<模型版本>/`，中文角色名保存在
`model.json`。共享基座、HuBERT、RMVPE 和声码器继续使用各项目的资源目录。

```text
models/
  rvc/klee/v2/
    model.pth
    model.index
    model.json
  gpt-sovits/klee/v4-zh/
    gpt.ckpt
    sovits.pth
    references/default.wav
    model.json
```

在共享环境中执行独立目录工具：

```powershell
# 检查模型，返回 JSON；--import 为新模型生成或补全 model.json。
python -B scripts/model_catalog.py scan --root models --import

# 将已识别且完整的模型整理为上述路径，内容相同的重复模型只保留一份。
python -B scripts/model_catalog.py normalize --root models
```

新增模型可先放入独立目录，再运行扫描；连接本目录的机器人也可通过“刷新模型”
完成同样的登记。扫描不会移动文件或启动推理。RVC 根据 checkpoint 的配置和权重
结构识别，GPT 与 SoVITS 分别检查；`.pth` 扩展名本身不决定服务类型。未知格式、
缺少配套权重、参考音频或转写，以及无法确定的多文件配对都会显示为 `blocked`。

每组模型的 `model.json` 保存稳定 ID、显示名称、服务类型、模型版本、相对资源
路径和 GPT-SoVITS 的参考文本/语言。一个目录有多个候选时，拆分目录或在该文件中
明确 `files` 配对。参考 WAV 文件名为 `【默认】完整参考文本.wav` 时，首次导入可
提取文本；其他名称需填写 `reference.text`。参考音频须完整且为 3～10 秒。
修改显示名称或移动模型目录不需要更换 ID。

```json
{
  "schemaVersion": 1,
  "id": "klee-v4-zh",
  "serviceId": "gpt-sovits",
  "displayName": "可莉",
  "modelVersion": "v4",
  "files": {
    "gpt": "gpt.ckpt",
    "sovits": "sovits.pth",
    "reference": "references/default.wav"
  },
  "reference": {
    "text": "填写参考音频实际说出的文字",
    "promptLang": "all_zh",
    "textLang": "all_zh"
  }
}
```

服务地址、Python 环境、设备和运行项目路径属于部署配置，不写入模型说明。
扫描输出协议为 `version: 1`，每项包含根目录相对路径、配置状态及 `resourceHash`；
后者仅覆盖本模型的执行资源和参考设置，训练日志不影响它。资源摘要和识别结果缓存在
`.codex/model-catalog-cache.json`，只有大小或修改时间变化的文件才重新读取。
调用方可用 `--cache <文件>` 指定自己的缓存位置。

## 测试

在根目录、已激活共享环境的终端中安装测试依赖：

```powershell
python -m pip install -r requirements-dev.txt -c constraints.txt
python -m pip check
python -B -m unittest discover -s tests -p test_model_catalog.py
Set-Location RVC
python -B -m pytest -q
Set-Location ../GPT-SoVITS
python -B -m unittest discover -s tests -t . -p "test_*.py"
```

两个套件分别执行，避免同名 `tests` 包干扰。RVC 数值回归使用随仓库保存的小型合成样本，无需旧 Git 历史。根目录 [CI 工作流](.github/workflows/inference.yml) 在 Windows/Linux CPU 环境安装共同依赖，运行两套测试、RVC wheel 检查及 GPT-SoVITS 镜像构建检查。

2026-10-31 在 Windows、RTX 5060 Ti、Torch 2.8.0+cu129 共享环境中的验收结果：

| 验证 | 结果 |
| --- | --- |
| RVC 回归 | 120 项通过、6 项实际 DirectML 测试跳过；14 个子测试通过 |
| RVC GPU 组件 | 四类 RVC、两类 RoFormer、HuBERT v1/v2 推理通过 |
| GPT-SoVITS 回归 | 89 项测试完成，1 个依赖本地 G2PW/tokenizer 的子场景跳过；多语言、LoRA、HuBERT 与 API 回归通过 |

测试使用合成模型和固定输入，实际测试数量以当前命令输出为准。

## 来源与许可证

本仓库从以下代码快照开始管理，不合入原提交历史：

| 项目 | 导入来源 |
| --- | --- |
| RVC | [Helloworld2345567/Retrieval-based-Voice-Conversion-WebUI](https://github.com/Helloworld2345567/Retrieval-based-Voice-Conversion-WebUI)，提交 `deabe1d` |
| GPT-SoVITS | [Helloworld2345567/GPT-SoVITS](https://github.com/Helloworld2345567/GPT-SoVITS)，提交 `5ed064c` |

原项目和第三方许可证保留在各自目录，见 [RVC LICENSE](RVC/LICENSE)、[RVC 引用说明](RVC/MIT协议暨相关引用库协议) 与 [GPT-SoVITS LICENSE](GPT-SoVITS/LICENSE)。模型权重遵循其各自来源的许可。

## Limitations / Known Issues

- 仓库不包含真实模型权重。接口及合成模型测试不能证明实际音质、完整模型显存需求或生成速度。
- 共享环境支持 NVIDIA CUDA 和 CPU。RVC 原 DirectML 依赖绑定 Torch 2.4.1，与共享环境不同；保留的 DirectML 命令参数并不表示此环境已安装该后端。
- GPT-SoVITS 的旧 FlashAttention wheel 绑定 Torch 2.7/CUDA 12.8，不能安装到当前共享环境。基础推理和可用的 PyTorch CUDA Graph 后端不要求这个 wheel。
- CPU CI 不验证实际 GPU 推理；Docker 构建及无模型测试也不代表带真实权重的服务已验收。
