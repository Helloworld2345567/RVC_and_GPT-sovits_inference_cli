# 推理验证

RVC 使用合并仓库的共享 Python 3.12 环境。常规回归由 CPU 上的小型模型、合成音频与固定数值样本构成；实际测试数量与结果以当前 pytest 输出为准。

## 测试覆盖

- RVC v1/v2、有 F0/无 F0 以及两类 RoFormer 的权重加载和数值输出，与随仓库保存的合成基线比较。
- 微型 12 层 HuBERT 的本地保存、加载与前向计算；验证 `final_proj` 权重、v1 第九层特征及 v2 最后隐藏层。
- checkpoint 配置、权重键名、说话人选择、索引匹配，以及损坏、空、错维度索引的失败行为。
- 小型 RVC checkpoint 经 CLI/Pipeline 输出有效音频并释放模型；五个分离预设的固定配置、分块合成和输出映射。
- 中文与空格路径、单文件与批量处理、递归转换、输出冲突、部分失败和返回码。
- WAV、FLAC、MP3、M4A 的实际合成音频编码与解码。
- 模拟下载的复用及失败清理、模块依赖边界、兼容入口和 DirectML worker 的协议。
- wheel 中的 CLI、YAML 与许可证资源，以及从仓库外调用已安装命令。

## 数值基线

`tests/fixtures/model_regression/` 保存六类小模型的合成权重、输入、随机种子及参考输出。样本来自原 RVC 提交 `81eed5e8f68b6bed1789f682fe78cdd324495afc`；测试读取随仓数据，不需要该提交对象或 Git 历史，也不下载预训练权重。

样本中的来源信息用于追踪生成版本。修改模型实现时，先解释数值变化，再决定是否需要重新生成；不能通过更新参考输出掩盖未解释的回归。

## 本地重跑

先按[根 README](../../README.md)安装并激活共享环境。以下命令从 `RVC/` 目录执行：

```bash
python -m pip install -r ../requirements-dev.txt -c ../constraints.txt
python -m pip check
python -B -m pytest -q
```

FFmpeg 和 ffprobe 必须位于 `PATH`。测试临时文件集中在项目 `.codex/`。实际 DirectML 用例默认跳过，因为共享环境使用 Torch 2.8.0，而原 DirectML 运行库绑定 Torch 2.4.1。

## Wheel 与 CI

在 `RVC/` 中构建：

```bash
python -m build --wheel --outdir ../.codex/dist
```

安装生成的 wheel 后，从仓库外运行 `rvc --help`、`rvc separate list`、`python -I -m infer.cli --help` 和 `python -I -m tools.pymss.cli list`。还需确认分离 YAML 与两个 PyMSS 许可文件可读，wheel 不包含测试、模型、缓存或字节码。

根目录 [CI 工作流](../../.github/workflows/inference.yml) 在 Windows Server 2022 与 Ubuntu 24.04 的同一个 Python 3.12 CPU 环境中分别运行 RVC 和 GPT-SoVITS 的套件，再构建和检查 wheel。浅克隆与代码压缩包都能执行数值回归。

## Limitations / Known Issues

- 合成模型测试不代表真实 RVC、HuBERT、RMVPE 或五个完整分离模型的音质、显存和速度已验收。
- CPU CI 不覆盖 NVIDIA CUDA 硬件或实际 DirectML 适配器。
