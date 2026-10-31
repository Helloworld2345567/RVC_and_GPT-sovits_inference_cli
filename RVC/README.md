# RVC CLI 推理

离线语音音色转换和人声分离。支持 RVC v1/v2、有 F0/无 F0、单说话人/多说话人及 FAISS 检索；人声分离提供五个固定模型。共享环境使用 CPU 或 NVIDIA CUDA。

## 安装

RVC 与 GPT-SoVITS 使用仓库根目录同一个 **Python 3.12 x64** 环境。先按[根目录安装说明](../README.md#安装共享环境)安装；Windows 从根目录执行：

```powershell
.\install.ps1 -Device CU129
. .\.codex\venv\Scripts\Activate.ps1
Set-Location RVC
rvc --help
```

共享环境使用 Torch/TorchAudio 2.8.0，NVIDIA 构建对应 CUDA 12.9；CPU 安装使用根安装器的 `-Device CPU`。FFmpeg 与 ffprobe 需可从 `PATH` 访问。`pyproject.toml` 从 `requirements-inference.txt` 读取 RVC 依赖，完整联合依赖与必要版本约束在仓库根目录维护。

根安装器已执行 RVC editable install。构建和安装独立 wheel 时，在本目录执行：

```bash
python -m pip install build
python -m build --wheel
python -m pip install --no-deps dist/rvc_cli_inference-*.whl
```

wheel 不包含模型、索引、测试或缓存。wheel 安装后的命令可在任意工作目录运行；外部模型目录需按下节和 [CLI 文档](docs/cli.md) 配置。

## 模型资源

```text
assets/
  hubert_base/        # config.json 与配套模型权重，包含 final_proj
  rmvpe/             # 按设备准备 rmvpe.pt 或 rmvpe.onnx
  weights/           # 用户的 RVC 推理 .pth
  indices/           # 用户的 added .index
  pymss_weights/     # 五个分离模型的权重
```

这些目录是本地运行数据，不会进入 Git 或 wheel，详见 [`assets/README.md`](assets/README.md)。源码检出时默认使用项目内目录；wheel 安装自动使用平台用户数据目录（Windows `%LOCALAPPDATA%/RVC`，Linux `XDG_DATA_HOME/rvc`），也可用 `RVC_DATA_DIR`、各资源环境变量、`--model-dir` 或显式路径覆盖。临时下载与 worker 文件使用对应用户缓存目录或 `RVC_CACHE_DIR`，不写入 `site-packages`。分离 YAML 和第三方许可文本随包安装。

RVC 变声需要本地 `.pth` 推理模型和适配 RVC 的 Transformers 格式 HuBERT/ContentVec。后者包含 `final_proj`，可从 [VoiceConversionWebUI 模型仓库](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main) 的 `hubert_base/` 获取；模型随附的 `preprocessor_config.json` 可以保留，但当前加载路径不要求该文件。

`.index` 由用户提供，`--index-rate 0` 时无需索引。有 F0 的模型选择 `--f0-method rmvpe` 时，CPU/CUDA 使用 `rmvpe.pt`，DirectML 使用同目录的 `rmvpe.onnx`；这两种权重也可从上述模型仓库获取。PM 无需额外音高模型，FCPE 使用 `torchfcpe` 附带的权重。无 F0 模型不加载音高模型。

分离配置随 `rvc/separation/resources/` 提供。`separate infer` 缺权重时下载选中的权重，已有权重直接复用；也可先执行 `separate download`。源码检出默认下载到 `assets/pymss_weights/`；wheel 安装默认下载到用户数据目录，也可通过 `--model-dir` 或 `PYMSS_MODEL_DIR` 指定。

## 使用

`python -m rvc` 与安装后的 `rvc` console script 等价：

```bash
rvc --help
rvc convert --help
rvc separate infer --help

# 不使用检索索引
rvc convert --model assets/weights/voice.pth --input input.wav --output output.wav --index-rate 0 --device cpu

# 选择多说话人模型的 speaker
rvc convert --model assets/weights/voice.pth --list-speakers
rvc convert --model assets/weights/voice.pth --speaker-id 1 --input input.wav --output output.wav --index assets/indices/voice_spkid1.index

# 目录递归转换，保留相对目录
rvc convert --model assets/weights/voice.pth --input inputs --output outputs --recursive --format flac --index-rate 0

# 查询五个分离模型
rvc separate list
rvc separate infer model_bs_roformer_ep_368_sdr_12.9628.ckpt --input input.wav --output separated --device cpu
```

`--device` 可选 `auto`、`cpu`、`cuda`、`directml`。自动模式按现有显卡资格规则选择 CUDA，随后尝试 DirectML，最后 CPU。显式选择不可用设备会报错。使用 `--device-id` 选择设备。

输出格式支持 WAV、FLAC、MP3 和 M4A。详细参数、分离模型表、输出命名和旧入口约定见 [CLI 文档](docs/cli.md)。

## 代码、架构与验证

规范实现位于 `rvc/core/`、`rvc/vc/` 和 `rvc/separation/`；`infer/cli.py`、`tools/pymss/cli.py` 及少量 `rvc` 根模块是迁移兼容 adapter。依赖方向、公共 interface、资源合同和兼容矩阵见 [架构说明](docs/architecture.md)。

激活已安装推理依赖的环境后运行：

```bash
python -m pip install -r ../requirements-dev.txt -c ../constraints.txt
python -m pytest -q
```

pytest 末行报告的是当前环境的 collected/passed/skipped 数量，不是固定产品版本号；硬件、平台 marker 和新增合同测试都会改变计数。结构回归使用随仓库保存的小型合成模型样本，不需要旧 Git 历史或真实权重。覆盖内容与重跑方式见 [验证说明](docs/validation.md)。

## 许可证

项目使用 [MIT 许可证](LICENSE)。保留的第三方代码、模型来源和外部工具见 [许可与引用说明](MIT协议暨相关引用库协议)，分离模块附有 [PyMSS](rvc/separation/LICENSE.pymss) 和 [PyMSS Core](rvc/separation/LICENSE.pymss-core) 的许可原文。

## Limitations / Known Issues

- 代码和 CI 验收使用 CLI、资源合同及合成模型；不代表真实音质或五个完整分离权重的效果验收。
- 普通 CI 是 Windows/Linux CPU 基线，不验证 NVIDIA CUDA 或实际 DirectML 硬件。
- 原 DirectML 后端依赖 Torch 2.4.1，与本仓库共享环境的 Torch 2.8.0 不兼容；默认跳过实际 DirectML 测试。
