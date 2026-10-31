# 推理服务部署

服务入口为 `api_v2.py`。本地部署使用合并仓库根目录的共享 Python 环境，服务命令和模型相对路径仍以 `GPT-SoVITS/` 为工作目录。Docker Compose 文件也位于该目录。

## 环境和设备选择

共享环境使用 **Python 3.12 x64**、配套的 **Torch/TorchAudio 2.8.0**，GPU 安装匹配本机 CUDA Toolkit **12.9**。完整依赖和必要的兼容限制由根目录清单维护，安装方法见[根 README](../../README.md#安装共享环境)。

| 设备 | Windows 根安装器选项 | PyTorch 下载源 | `custom.device` |
| --- | --- | --- | --- |
| NVIDIA / CUDA 12.9 | `-Device CU129`，默认值 | `https://download.pytorch.org/whl/cu129` | `cuda` 或 `cuda:0` |
| CPU | `-Device CPU` | `https://download.pytorch.org/whl/cpu` | `cpu` |

CPU 配置使用 `is_half: false`。安装后检查实际使用的解释器、Torch 构建与设备，再设置推理配置：

```bash
python -c "import sys, torch, torchaudio; print(sys.executable); print('torch:', torch.__version__, 'audio:', torchaudio.__version__); print('CUDA:', torch.version.cuda, 'GPU available:', torch.cuda.is_available())"
```

## Windows：使用根目录共享环境

准备 [uv](https://docs.astral.sh/uv/getting-started/installation/)，并将包含 `ffmpeg.exe`、`ffprobe.exe` 的 [FFmpeg](https://ffmpeg.org/download.html) 工具包加入 `PATH`。首次安装还需 Visual Studio C++ Build Tools 的“使用 C++ 的桌面开发”组件和 Windows SDK；根安装器自动发现 Visual Studio（包括 Preview）并启用 x64 编译环境。在合并仓库根目录执行：

```powershell
ffmpeg -version
ffprobe -version
.\install.ps1 -Device CU129
. .\.codex\venv\Scripts\Activate.ps1
Set-Location GPT-SoVITS
```

安装器创建共享环境、安装两套运行依赖并准备语言资源。之后每个新终端只需从根目录激活 `.codex/venv`，再进入本目录；RVC 也使用相同解释器。PyOpenJTalk 0.4.1 从源码编译以适配 NumPy 2，CMake、Cython 等构建依赖由安装工具自动准备；`jieba_fast` 使用现有 Windows Python 3.12 预编译 wheel。

## Linux：使用共享 CPU 环境

按[根 README 的 Linux CPU 步骤](../../README.md#linux-cpu)安装 Python 3.12、FFmpeg、libsndfile、构建工具及共享依赖，然后进入 `GPT-SoVITS/` 启动服务。语言资源由 NLTK 和 OpenJTalk 准备；模型权重另行提供。安装依赖与语言资源需要联网，离线部署见下文。

## 模型和启动配置

推理模型可从 [GPT-SoVITS 模型库](https://huggingface.co/lj1995/GPT-SoVITS) 和 [预训练资源库](https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained) 获取。保留模型资源原有文件名与目录结构：

| 资源 | 默认位置或内容 |
| --- | --- |
| BERT | `GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/`：模型配置、权重及 tokenizer |
| CNHuBERT | `GPT_SoVITS/pretrained_models/chinese-hubert-base/`：模型配置、权重及音频配置 |
| GPT / SoVITS | 配置中 `t2s_weights_path` / `vits_weights_path` 指向的 `.ckpt` / `.pth` |
| G2PW | `GPT_SoVITS/text/G2PWModel/`：`g2pW.onnx`、配置及配套字表、JSON 文件 |
| v2Pro / v2ProPlus 声纹模型 | `GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt` |
| v3 声码器 | `GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x/` 下的 `config.json`、`bigvgan_generator.pt` |
| v4 声码器 | `GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth` |
| v5dev / v5turbo 声码器 | `GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth` |
| 可选超采样 | 规范位置 `GPT_SoVITS/vendor/ap_bwe/24kto48k/g_24kto48k.zip` 及同目录 `config.json`；旧 `tools/AP_BWE_main/24kto48k/` 仍作为回退 |
| 参考音频 | 例如 `references/reference.wav`，请求指定其本地路径 |

LoRA 权重还需要相应版本的基础 SoVITS 权重。准备实际使用版本所需的资源即可。

复制 [v4 CPU 样例](../examples/tts_v4_cpu.yaml) 到 `GPT_SoVITS/configs/tts_local.yaml`，再编辑其中的 **`custom`**。`v1`、`v2`、`v2Pro`、`v2ProPlus`、`v3`、`v4`、`v5dev`、`v5turbo` 块提供各版本路径示例；启动时读取 `custom`，编辑其他块不会自动切换当前模型。以下为 CPU / v2 示例：

```yaml
custom:
  version: v2
  device: cpu
  is_half: false
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
  t2s_weights_path: GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
  vits_weights_path: GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth
```

本机可使用绝对路径；相对模型路径和参考音频路径按 `GPT-SoVITS/` 目录解析。参考音频通过路径读取，请求正文不上传音频。更换模型版本时同时检查 `custom.version`、GPT、SoVITS、基础权重及声码器是否配套。

## 启动、日志和请求

在已激活的隔离环境中启动：

```bash
python api_v2.py -c GPT_SoVITS/configs/tts_local.yaml -a 127.0.0.1 -p 9880
```

Windows 也使用同样参数：

```powershell
python .\api_v2.py -c GPT_SoVITS/configs/tts_local.yaml -a 127.0.0.1 -p 9880
```

`-c/--tts_config` 指定配置文件，`-a/--bind_addr` 指定监听地址，`-p/--port` 指定端口。默认地址为 `127.0.0.1:9880`；需要外部客户端访问时使用 `-a 0.0.0.0`。

模型在服务启动阶段加载，日志出现 `Application startup complete` 后再发送请求。权重缺失会导致启动失败；单独运行 `python -c "import api_v2"` 不加载 TTS 权重，不能替代完整启动检查。`api_v2.py` 是稳定兼容入口，实际应用和 CLI 位于 `GPT_SoVITS/service/`，推理实现位于 `GPT_SoVITS/inference/`。服务固定使用一个 worker；请求、参考音频设置和模型切换共享该进程的模型状态。

日志默认输出到控制台。Linux 保存日志：

```bash
mkdir -p .codex/logs
python -u api_v2.py -c GPT_SoVITS/configs/tts_local.yaml -a 0.0.0.0 -p 9880 2>&1 | tee .codex/logs/api.log
```

PowerShell 保存日志：

```powershell
New-Item -ItemType Directory -Force .codex/logs | Out-Null
python -u .\api_v2.py -c GPT_SoVITS/configs/tts_local.yaml -a 0.0.0.0 -p 9880 *>&1 | Tee-Object -FilePath .codex/logs/api.log
```

打开 [交互式 API 文档](http://127.0.0.1:9880/docs) 查看 `/tts` 的 GET/POST 参数。普通请求可使用 `media_type: "wav"`、`streaming_mode: false`；模型和参考音频准备好后，以实际生成音频的请求确认部署可用。流式参数、示例及支持组合见 [API 使用说明](api.md)。

保留的模型操作接口为 `GET /set_refer_audio`、`GET /set_gpt_weights` 和 `GET /set_sovits_weights`。`/set_*` 与推理串行执行；切换仅影响当前进程，重启后仍按 `custom` 加载。

## Docker Compose

默认镜像使用 Python 3.12 与 PyTorch 2.8.0 / CUDA 12.9，安装根目录联合依赖，包含 FFmpeg 及预下载的 NLTK/OpenJTalk 系统资源。构建上下文为合并仓库根目录，Dockerfile 位于 `GPT-SoVITS/`。准备 Docker Engine、Compose，并为 NVIDIA 部署配置 NVIDIA Container Toolkit。Compose 中默认有 `gpus: all`。

将配置、权重和参考音频放到上述宿主机目录，创建 `references/` 与 `.codex/runtime-cache/` 后运行：

```bash
docker compose config
docker compose up -d --build
docker compose logs -f gpt-sovits-inference
```

`docker compose config` 检查配置展开是否正确；模型加载和进程启动结果查看容器日志。镜像入口直接执行 `python api_v2.py`，内部监听 `0.0.0.0:9880`，只声明 API 端口 `9880`。默认宿主机映射也是 `9880:9880`；可通过环境变量 `API_PORT` 修改宿主机端口，例如：

```bash
API_PORT=19880 docker compose up -d
```

PowerShell 设置 `$env:API_PORT = '19880'` 后执行 `docker compose up -d`。容器内仍使用 `9880`。

### CPU 容器

编辑 `docker-compose.yaml`，删除 `gpus: all`；将挂载配置中的 `custom.device` 设为 `cpu`、`custom.is_half` 设为 `false`，再启动 Compose。当前默认 CUDA 基础镜像也可以执行 CPU 推理，这种运行方式仍使用同一镜像。

### 挂载自定义权重

Compose 将默认预训练资源、G2PW、推理配置、参考音频和 AP-BWE 目录挂载到 `/app/GPT-SoVITS/...`，均为只读；`.codex/runtime-cache/` 为可写持久化缓存。AP-BWE 同时挂载规范目录 `/app/GPT-SoVITS/GPT_SoVITS/vendor/ap_bwe/24kto48k` 和旧目录 `/app/GPT-SoVITS/tools/AP_BWE_main/24kto48k`：运行时优先规范目录，保留旧挂载是为了已有部署在迁移权重前仍可运行。自定义权重若放在 `models/`，在服务的 `volumes` 中追加：

```yaml
services:
  gpt-sovits-inference:
    volumes:
      - ./models:/app/GPT-SoVITS/models:ro
```

这是对现有服务配置的补充，保留原来的模型、配置、参考音频和缓存挂载。也可以把宿主机上的其他模型目录挂载到 `/app/GPT-SoVITS/models`。根目录 `.dockerignore` 已排除隔离环境和默认模型目录；使用其他自定义权重目录时，也将其加入根 `.dockerignore`，使权重通过挂载提供并减少镜像构建上下文。

在容器部署的配置和 API 请求中使用**容器路径**，例如：

```yaml
custom:
  t2s_weights_path: /app/GPT-SoVITS/models/voice/model.ckpt
  vits_weights_path: /app/GPT-SoVITS/models/voice/model.pth
```

同一 `custom` 中保留版本、设备和其他基础模型路径。参考音频使用 `/app/GPT-SoVITS/references/reference.wav`，热切换接口的 `weights_path` 使用 `/app/GPT-SoVITS/models/voice/model.ckpt` 或 `/app/GPT-SoVITS/models/voice/model.pth`。宿主机的 `D:\models\voice.pth` 等路径不能直接由容器读取。

检查服务状态和停止服务：

```bash
docker compose ps
docker compose down
```

## 缓存和离线准备

离线机器需要先具备运行依赖、FFmpeg 和所用版本的全部模型。可在相同平台、Python 版本及设备后端的环境完成安装和资源准备后迁移部署材料；安装脚本自身会访问 PyPI/PyTorch 和语言资源下载站点。规范运行资源由 `GPT_SoVITS/runtime/paths.py` 定位，顶层 `runtime/` 仅用于可选便携 FFmpeg、DLL、site-packages 和 NLTK 数据。

安装器、CI 和镜像构建统一调用[语言资源准备脚本](../../scripts/prepare_language_resources.py)。需要单独准备或检查资源时，在合并仓库根目录、已激活的共享环境中执行：

```bash
python scripts/prepare_language_resources.py
```

该脚本从官方固定地址准备五项 NLTK 资源，并检查英语分词、词性标注与 OpenJTalk；默认保存到本目录 `.codex/runtime-cache/nltk_data/`，也可通过 `NLTK_DATA` 指定位置。

| 资源 | 准备方法与运行位置 |
| --- | --- |
| NLTK | 上述脚本准备 `cmudict`、`punkt`、`punkt_tab`、`averaged_perceptron_tagger`、`averaged_perceptron_tagger_eng` |
| OpenJTalk 系统词典 | 由同一语言资源准备脚本下载并检查 |
| OpenJTalk 用户词典 | 保留 `GPT_SoVITS/text/ja_userdic/userdict.csv`；首次日文处理编译到 `.codex/runtime-cache/openjtalk/`，后续按校验信息复用 |
| 语言检测 | 提前放置 `.codex/runtime-cache/fast_langdetect/lid.176.bin`，避免首次文本处理时下载 |
| BERT / CNHuBERT / G2PW / 声码器 | 按模型目录完整复制，配置指向本地路径；仅有 GPT 和 SoVITS 权重不足以完成推理 |

语言检测缓存可用 `GPT_SOVITS_LANGDETECT_CACHE` 指定其他可写目录。已有 `GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin` 也会被直接只读使用；默认缓存目录中的同名模型优先。源码和模型目录只读部署时，仍需给 `.codex/runtime-cache/` 写权限。

设置 `NLTK_DATA` 可指向预先准备的 NLTK 目录；运行时也会依次查找项目便携包的 `runtime/nltk_data/` 和 `.codex/runtime-cache/nltk_data/`。便携包中存在 `runtime/ffmpeg.exe` / `runtime/ffprobe.exe` 时优先使用它们，否则从系统 `PATH` 查找。

本地 Hugging Face 模型及 tokenizer 准备完整后，可设置 `HF_HUB_OFFLINE=1` 验证离线启动和一次真实请求。该变量只控制 Hugging Face 的访问，G2PW、语言检测和 NLTK 等资源仍须分别准备。Docker 离线运行需要提前构建或导入完整镜像，并准备宿主机挂载目录；`.dockerignore` 排除了默认预训练权重、G2PW、AP-BWE ZIP 和 `.codex` 缓存目录，镜像构建不会自动包含这些资源。容器内 NLTK 默认使用镜像预置的 `/opt/nltk_data`。

## 检查和 CI

先检查入口可导入，再运行回归：

```bash
python -c "import api_v2; from GPT_SoVITS.service.app import APP; from GPT_SoVITS.inference import TTS, TTS_Config; assert api_v2.APP is APP and api_v2.tts_pipeline is None"
python -m unittest discover -s tests -t . -p "test_*.py"
```

这些检查覆盖 API 参数和错误响应、流式清理与取消、模型切换、权重兼容、多语言处理及推理组件数值行为。语言测试需要安装器准备的资源；没有本地 G2PW 模型时，依赖该模型的中文子项会显式跳过。真实 `/tts` 生成还需要使用自己的权重与参考音频单独验证。

[CI 工作流](../../.github/workflows/inference.yml) 在 Windows/Linux 上安装 Python 3.12 共享 CPU 环境，分别运行 RVC 与 GPT-SoVITS 回归；Docker 作业使用根目录联合依赖构建镜像，执行容器内导入、HTTP 合约与 RVC 命令检查。该作业不使用真实 TTS 权重。测试分类和模块边界见 [目录与架构](architecture.md)。

2026-10-31 的 Windows 共享环境回归完成 89 项测试，覆盖中英日韩粤语言快照、LoRA 合并、HuBERT 与 API；1 个依赖本地 G2PW/tokenizer 的子场景因资源缺失跳过。联合环境及 RVC GPU 组件结果见[根 README](../../README.md#测试)。

## Limitations / Known Issues

- 仓库不包含真实模型权重。组件数值回归、无模型服务导入和 HTTP 合约测试不代表真实音频效果、完整模型显存或吞吐已验收。
- Windows 的旧 FlashAttention wheel 绑定 Torch 2.7/CUDA 12.8，不能用于当前 Torch 2.8/CUDA 12.9 共享环境；基础推理与可用的 PyTorch CUDA Graph 后端不依赖该 wheel。
- 普通 CI 使用 CPU。Docker 构建和无模型测试不替代真实 GPU、权重及参考音频的部署验收。
