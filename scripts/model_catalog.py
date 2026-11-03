"""Discover and organize local RVC/GPT-SoVITS voice models without inference.

The JSON protocol is independent of any bot host. Model manifests contain relative
resource paths; deployment addresses, Python paths and credentials never belong here.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import sys
import uuid
import wave


PROJECT = Path(__file__).resolve().parent.parent
CHARACTERS = {"可莉": "klee", "胡桃": "hutao", "派蒙": "paimon", "诺艾尔": "noelle", "香菱": "xiangling", "宵宫": "yoimiya"}
DISPLAY_NAMES = {value: key for key, value in CHARACTERS.items()}
HEADERS = {b"00": ("v1", False), b"01": ("v2", False), b"02": ("v3", False), b"03": ("v3", True), b"04": ("v4", True), b"05": ("v2Pro", False), b"06": ("v2ProPlus", False), b"07": ("v5dev", True), b"08": ("v5turbo", True)}
MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")
LANGUAGES = {"auto", "en", "zh", "ja", "all_zh", "all_ja", "auto_yue", "yue", "ko", "all_yue", "all_ko"}
DUPLICATE_ID = "模型 ID 重复，请为不同模型设置独立 ID"


def digest_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def relative_file(root: Path, value: str, base: Path | None = None):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("资源路径必须是模型目录中的相对路径")
    path = (base or root) / value
    if path.is_symlink() or path.is_junction():
        raise ValueError("模型资源不能使用目录链接")
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("资源路径超出模型根目录")
    return resolved


class ResourceCache:
    def __init__(self, root: Path, file: Path | None):
        self.root, self.file = root.resolve(), file
        self.data = {}
        self.hash_reads = 0
        self.probe_reads = 0
        if file and file.is_file():
            try:
                saved = json.loads(file.read_text(encoding="utf-8"))
                if saved.get("version") == 2 and saved.get("root") == str(self.root):
                    self.data = saved.get("files", {})
            except (OSError, ValueError, TypeError):
                pass

    def entry(self, path: Path):
        path = path.resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("模型资源不存在或不属于模型目录")
        key = path.relative_to(self.root).as_posix()
        stat = path.stat()
        stamp = [stat.st_size, stat.st_mtime_ns]
        item = self.data.get(key)
        if not isinstance(item, dict) or item.get("stamp") != stamp:
            item = {"stamp": stamp}
            self.data[key] = item
        return item

    def digest(self, path: Path):
        item = self.entry(path)
        if "sha256" not in item:
            before = path.stat()
            value = hashlib.sha256()
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    value.update(block)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("模型资源在扫描期间变化，请重新刷新")
            item["sha256"] = value.hexdigest()
            self.hash_reads += 1
        return item["sha256"]

    def probe(self, path: Path):
        item = self.entry(path)
        if "probe" not in item:
            self.probe_reads += 1
            item["probe"] = inspect_checkpoint(path)
        return item["probe"]

    def moved(self, source: Path, target: Path):
        key = source.relative_to(self.root).as_posix()
        value = self.data.pop(key, None)
        if value and target.is_relative_to(self.root):
            self.data[target.relative_to(self.root).as_posix()] = value

    def save(self):
        if self.file:
            write_json(self.file, {"version": 2, "root": str(self.root), "files": self.data})


def safe_load(path):
    import torch

    return torch.load(str(path), map_location="cpu", weights_only=True)


def inspect_checkpoint(path: Path):
    """Classify actual structure; never execute general pickle or guess by size."""
    try:
        with path.open("rb") as stream:
            header = stream.read(2)
        if path.suffix.lower() == ".pth" and header in HEADERS:
            # Tagged SoVITS archives use the upstream two-byte version header.
            from io import BytesIO
            import torch

            with path.open("rb") as stream:
                stream.read(2)
                archive = BytesIO(b"PK" + stream.read())
            spec = importlib.util.spec_from_file_location("catalog_hparams", PROJECT / "GPT-SoVITS/GPT_SoVITS/utils.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with torch.serialization.safe_globals([(module.HParams, "utils.HParams")]):
                checkpoint = torch.load(archive, map_location="cpu", weights_only=True)
            version, lora = HEADERS[header]
            if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("weight"), dict) or not checkpoint["weight"] or not isinstance(checkpoint.get("config"), (dict, module.HParams)):
                raise ValueError("SoVITS 权重不完整")
            if lora and (not isinstance(checkpoint.get("lora_rank"), int) or checkpoint["lora_rank"] <= 0):
                raise ValueError("SoVITS LoRA 缺少有效 rank")
            return {"kind": "sovits", "modelVersion": version, "modelKind": "lora" if lora else "full"}
        checkpoint = safe_load(path)
        if not isinstance(checkpoint, dict):
            raise ValueError("权重不是模型字典")
        weights = checkpoint.get("weight")
        config = checkpoint.get("config")
        if isinstance(config, (list, tuple)) and isinstance(weights, dict) and "emb_g.weight" in weights:
            module_path = PROJECT / "RVC/rvc/vc/checkpoints.py"
            spec = importlib.util.spec_from_file_location("model_catalog_rvc_checkpoints", module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            metadata = module.metadata_from_checkpoint(checkpoint)
            return {"kind": "rvc", "modelVersion": metadata.version, "usesF0": metadata.uses_f0, "sampleRate": metadata.sample_rate, "speakerCount": metadata.speaker_count}
        if isinstance(config, dict) and isinstance(weights, dict):
            if "model.ar_text_embedding.word_embeddings.weight" in weights and "model.ar_audio_embedding.word_embeddings.weight" in weights:
                return {"kind": "gpt"}
            # Untagged SoVITS requires structural evidence and a declared version.
            version = config.get("model", {}).get("version") if isinstance(config.get("model"), dict) else None
            if version in {item[0] for item in HEADERS.values()} and any(key.startswith("enc_p.") for key in weights) and any(key.startswith("dec.") for key in weights):
                return {"kind": "sovits", "modelVersion": version, "modelKind": "full"}
        raise ValueError("无法识别为 RVC、GPT 或 SoVITS 权重")
    except Exception as error:
        # Do not echo pickle payloads or paths supplied by an archive into the API.
        return {"kind": "unknown", "error": "无法安全识别权重，请检查文件格式和模型完整性", "errorType": type(error).__name__}


def identity(directory: Path, files):
    value = " ".join([directory.as_posix(), *(file.name for file in files)])
    for display, slug in CHARACTERS.items():
        if display in value or slug in directory.parts:
            return slug, display
    name = directory.name
    if name in {"rvc", "gpt-sovits", "v1", "v2", "v4", "v4-zh"}:
        name = directory.parent.name
    slug = re.sub(r"[^a-z0-9_-]+", "-", name.lower()).strip("-")
    return slug[:45] or "voice", name


def _files_under(root: Path):
    files = []
    for path in root.rglob("*"):
        if len(path.relative_to(root).parts) > 12:
            raise ValueError("模型目录超过最大层级")
        if path.is_symlink() or path.is_junction():
            raise ValueError("模型目录包含目录链接")
        if path.is_file():
            files.append(path)
            if len(files) > 10000:
                raise ValueError("模型目录文件数超过限制")
    return sorted(files, key=lambda path: path.as_posix().lower())


def discover_manifest(directory: Path, service: str, pths, cache, existing=None):
    manifest = dict(existing or {})
    files = dict(manifest.get("files", {})) if isinstance(manifest.get("files", {}), dict) else {}
    diagnostics = []
    slug, display = identity(directory, pths)
    manifest.setdefault("schemaVersion", 1)
    manifest.setdefault("serviceId", service)
    manifest.setdefault("displayName", display)
    declared = manifest.get("modelVersion")
    primary = "weights" if service == "rvc" else "sovits"
    wanted = "rvc" if service == "rvc" else "sovits"
    candidates = [file for file in pths if cache.probe(file).get("kind") == wanted]
    if not files.get(primary):
        if len(candidates) == 1:
            files[primary] = candidates[0].relative_to(directory).as_posix()
        elif len(candidates) > 1:
            diagnostics.append("存在多个角色权重，请拆分目录或在 model.json 指定配对")
        else:
            diagnostics.append("缺少可识别的角色权重")
    if files.get(primary):
        try:
            meta = cache.probe(relative_file(cache.root, files[primary], directory))
            if meta.get("kind") != wanted:
                diagnostics.append("角色权重类型与服务类型不一致")
            elif declared and declared != meta.get("modelVersion"):
                diagnostics.append("声明的模型版本与权重不一致")
            else:
                manifest["modelVersion"] = meta["modelVersion"]
        except (OSError, ValueError):
            diagnostics.append("角色权重不存在或路径无效")
    version = manifest.get("modelVersion", "unknown")
    manifest.setdefault("id", f"{slug}-{version}" + ("-zh" if service == "gpt-sovits" else ""))
    if service == "rvc":
        indexes = sorted(directory.glob("*.index"))
        if not files.get("index"):
            if len(indexes) == 1:
                files["index"] = indexes[0].name
            elif len(indexes) > 1:
                diagnostics.append("存在多个索引，请在 model.json 指定匹配索引")
    else:
        gpts = [file for file in directory.glob("*.ckpt") if cache.probe(file).get("kind") == "gpt"]
        if not files.get("gpt"):
            if len(gpts) == 1:
                files["gpt"] = gpts[0].name
            else:
                diagnostics.append("缺少 GPT 权重" if not gpts else "存在多个 GPT 权重，请指定配对")
        references = sorted(directory.rglob("*.wav"))
        if not files.get("reference"):
            if len(references) == 1:
                files["reference"] = references[0].relative_to(directory).as_posix()
            else:
                diagnostics.append("缺少参考音频" if not references else "存在多个参考音频，请指定默认参考")
        reference = dict(manifest.get("reference", {})) if isinstance(manifest.get("reference", {}), dict) else {}
        if files.get("reference") and not reference.get("text"):
            stem = Path(files["reference"]).stem
            # Human speech is recoverable only from the tagged original filename.
            if stem.startswith("【") and "】" in stem:
                reference["text"] = stem.split("】", 1)[1]
        reference.setdefault("promptLang", "all_zh")
        reference.setdefault("textLang", "all_zh")
        manifest["reference"] = reference
    manifest["files"] = files
    return manifest, diagnostics


def validate_model(root, directory, manifest, cache, diagnostics):
    diagnostics = list(diagnostics)
    service = manifest.get("serviceId")
    if service not in ("rvc", "gpt-sovits"):
        service = "gpt-sovits" if "gpt-sovits" in directory.parts else "rvc"
        diagnostics.append("model.json 的服务类型无效")
    model_id = manifest.get("id", "")
    if not isinstance(model_id, str) or not MODEL_ID.fullmatch(model_id):
        model_id = "invalid-" + hashlib.sha256(directory.relative_to(root).as_posix().encode()).hexdigest()[:20]
        diagnostics.append("model.json 的模型 ID 无效")
    if manifest.get("schemaVersion") != 1:
        diagnostics.append("不支持的 model.json 版本")
    hashes, files, metadata = {}, {}, {}
    required = {"weights"} if service == "rvc" else {"gpt", "sovits", "reference"}
    paths = manifest.get("files", {})
    for key in sorted(required | ({"index"} if paths.get("index") else set())):
        if not paths.get(key):
            diagnostics.append(f"缺少模型资源：{key}")
            continue
        try:
            file = relative_file(root, paths[key], directory)
            hashes[key] = cache.digest(file)
            files[key] = file.relative_to(root).as_posix()
            if key in {"weights", "gpt", "sovits"}:
                metadata[key] = cache.probe(file)
                expected = {"weights": "rvc", "gpt": "gpt", "sovits": "sovits"}[key]
                if metadata[key].get("kind") != expected:
                    diagnostics.append(f"资源类型错误或无法安全读取：{key}")
            if key == "reference":
                with wave.open(str(file), "rb") as audio:
                    duration = audio.getnframes() / audio.getframerate()
                    if not 3 <= duration <= 10:
                        diagnostics.append("参考音频必须为 3～10 秒完整 WAV")
            if key == "index":
                with file.open("rb") as stream:
                    header = stream.read(8)
                if len(header) != 8 or header[:4] not in {b"IwFl", b"IxF2", b"IxFI"}:
                    diagnostics.append("无法识别 RVC 索引格式")
                elif struct.unpack("<i", header[4:])[0] != (256 if manifest.get("modelVersion") == "v1" else 768):
                    diagnostics.append("RVC 索引维度与模型版本不一致")
        except (OSError, ValueError, wave.Error, EOFError):
            diagnostics.append(f"模型资源不存在、损坏或路径无效：{key}")
    primary = metadata.get("weights" if service == "rvc" else "sovits", {})
    version = manifest.get("modelVersion", "unknown")
    if primary.get("modelVersion") and primary["modelVersion"] != version:
        diagnostics.append("声明的模型版本与权重不一致")
    if service == "rvc" and primary.get("usesF0") is False:
        diagnostics.append("当前 RVC 服务需要 F0 模型")
    reference = manifest.get("reference") if service == "gpt-sovits" else None
    if service == "gpt-sovits":
        if not isinstance(reference, dict):
            reference = {}
        if not isinstance(reference.get("text"), str) or not reference["text"].strip():
            diagnostics.append("缺少参考音频的准确转写文本")
        if reference.get("promptLang") not in LANGUAGES or reference.get("textLang") not in LANGUAGES:
            diagnostics.append("参考或生成语言无效")
    display = manifest.get("displayName")
    if not isinstance(display, str) or not display.strip():
        display = directory.name
        diagnostics.append("缺少显示名称")
    result = {"id": model_id, "serviceId": service, "displayName": display, "modelVersion": version, "modelLabel": f"{display} · {version}" + (" LoRA" if primary.get("modelKind") == "lora" else ""), "state": "blocked" if diagnostics else "ready", "files": files, "resourceHash": digest_json({"files": hashes, "reference": reference, "modelVersion": version}), "diagnostics": list(dict.fromkeys(diagnostics))}
    if reference is not None:
        result["reference"] = reference
    if primary.get("modelKind"):
        result["modelKind"] = primary["modelKind"]
    for key in ("sampleRate", "speakerCount", "usesF0"):
        if key in primary:
            result[key] = primary[key]
    return result


def scan(root: Path, *, import_models=False, cache_file=None):
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("模型根目录不存在")
    cache = ResourceCache(root, cache_file)
    all_files = _files_under(root)
    manifests = {file.parent: file for file in all_files if file.name == "model.json"}
    pth_groups, resource_directories = {}, set()
    for file in all_files:
        if file.suffix.lower() == ".pth":
            pth_groups.setdefault(file.parent, []).append(file)
        if file.suffix.lower() in {".pth", ".ckpt", ".index"}:
            resource_directories.add(file.parent)
    models, details, diagnostics, ids = [], [], [], set()
    for directory in sorted(set(manifests) | resource_directories, key=lambda p: p.as_posix().lower()):
        existing, errors = None, []
        if directory in manifests:
            try:
                existing = json.loads(manifests[directory].read_text(encoding="utf-8"))
                if not isinstance(existing, dict) or not isinstance(existing.get("files", {}), dict):
                    raise ValueError("invalid manifest")
            except (ValueError, OSError):
                existing = {"id": "invalid-" + hashlib.sha256(directory.relative_to(root).as_posix().encode()).hexdigest()[:20], "schemaVersion": 1}
                errors.append("model.json 无法读取，请修正 JSON 内容")
        pths = pth_groups.get(directory, [])
        kinds = {cache.probe(file).get("kind") for file in pths}
        service = existing.get("serviceId") if existing else None
        if service not in ("rvc", "gpt-sovits"):
            has_gpt = any(cache.probe(file).get("kind") == "gpt" for file in directory.glob("*.ckpt"))
            service = "gpt-sovits" if "sovits" in kinds or "gpt-sovits" in directory.parts or has_gpt else "rvc"
        if "rvc" in kinds and "sovits" in kinds:
            errors.append("同目录混合多个服务模型，请拆分目录")
        manifest, inferred = discover_manifest(directory, service, pths, cache, existing)
        if existing is None and (service, manifest["id"]) in ids:
            manifest["id"] = manifest["id"][:60] + "-" + uuid.uuid4().hex[:8]
        model = validate_model(root, directory, manifest, cache, errors + inferred)
        identity_key = (model["serviceId"], model["id"])
        collision = identity_key in ids
        if collision:
            model["state"] = "blocked"
            model["diagnostics"].append(DUPLICATE_ID)
            for previous in models:
                if (previous["serviceId"], previous["id"]) == identity_key:
                    previous["state"] = "blocked"
                    message = DUPLICATE_ID
                    if message not in previous["diagnostics"]:
                        previous["diagnostics"].append(message)
        ids.add(identity_key)
        if import_models and not errors and manifest != existing:
            write_json(directory / "model.json", manifest)
        if not collision:
            models.append(model)
        details.append((directory, manifest, model))
    cache.save()
    return {"version": 1, "models": models, "diagnostics": list(dict.fromkeys(diagnostics))}, details, cache


def normalize(root: Path, *, cache_file=None):
    root = root.resolve()
    _, details, cache = scan(root, import_models=True, cache_file=cache_file)
    # Explicit normalization may resolve duplicate IDs by comparing complete
    # resource hashes. All other failures, including those hidden by a duplicate
    # ID's public aggregation, must stop the move.
    if any(issue != DUPLICATE_ID for _, _, model in details for issue in model["diagnostics"]):
        raise ValueError("存在缺项或歧义，请先按 scan 诊断修正，尚未移动资源")
    moves, duplicates = [], []
    targets, target_manifests, used_ids = {}, {}, {}
    plans = []

    def canonical_target(directory, model):
        slug, _ = identity(directory, [root / file for file in model["files"].values()])
        variant = model["modelVersion"] + ("-zh" if model["serviceId"] == "gpt-sovits" else "")
        return root / model["serviceId"] / slug / variant

    # An existing canonical package owns its ID and manifest even when a newly
    # imported directory sorts before it alphabetically.
    ordered = sorted(details, key=lambda item: (item[0] != canonical_target(item[0], item[2]), item[0].as_posix().lower()))
    for directory, manifest, model in ordered:
        target = canonical_target(directory, model)
        previous = targets.get(target)
        if previous and previous != model["resourceHash"]:
            target = target.with_name(target.name + "-" + model["resourceHash"][:8])
        if target in targets and targets[target] != model["resourceHash"]:
            raise ValueError("模型变体目标冲突，尚未移动资源")
        targets[target] = model["resourceHash"]
        names = {"weights": "model.pth", "index": "model.index", "gpt": "gpt.ckpt", "sovits": "sovits.pth", "reference": "references/default.wav"}
        selected_names = {key: names[key] for key in model["files"]}
        if target not in target_manifests:
            selected_id = manifest["id"]
            identity_key = (model["serviceId"], selected_id)
            if identity_key in used_ids and used_ids[identity_key] != target:
                selected_id = selected_id[:60] + "-" + model["resourceHash"][:12]
                identity_key = (model["serviceId"], selected_id)
                if identity_key in used_ids and used_ids[identity_key] != target:
                    raise ValueError("模型变体 ID 冲突，尚未移动资源")
            used_ids[identity_key] = target
            target_manifests[target] = {**manifest, "id": selected_id, "files": selected_names}
        plans.append((directory, manifest, model, target, selected_names))
    # Validate every destination before moving anything.
    for directory, manifest, model, target, names in plans:
        if not target.resolve().is_relative_to(root):
            raise ValueError("整理目标超出模型目录")
        for key, filename in names.items():
            source, destination = root / model["files"][key], target / filename
            if destination.exists() and source.resolve() != destination.resolve() and cache.digest(source) != cache.digest(destination):
                raise ValueError("规范目标已有不同内容，请先处理命名冲突")
    for directory, manifest, model, target, names in plans:
        for key, filename in names.items():
            source, destination = root / model["files"][key], target / filename
            if source.resolve() == destination.resolve():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if cache.digest(source) != cache.digest(destination):
                    raise ValueError("目标内容在整理期间发生变化")
                source.unlink()
                duplicates.append(source.relative_to(root).as_posix())
            else:
                source.rename(destination)
                cache.moved(source, destination)
                moves.append({"from": source.relative_to(root).as_posix(), "to": destination.relative_to(root).as_posix()})
        if (directory / "train.log").is_file():
            source = directory / "train.log"
            destination = target / "metadata/train.log"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if source.read_bytes() == destination.read_bytes():
                    source.unlink()
                else:
                    source.rename(destination.with_name("train-" + hashlib.sha256(source.read_bytes()).hexdigest()[:8] + ".log"))
            else:
                source.rename(destination)
        if directory != target and (directory / "model.json").is_file():
            (directory / "model.json").unlink()
    # Each target is written once: a duplicate cannot replace its owner's
    # display name, reference settings, ID or other manifest information.
    for target, manifest in target_manifests.items():
        write_json(target / "model.json", manifest)
    sample = root / "胡桃/泡沫.mp3"
    if sample.is_file():
        target = root.parent / "outputs/samples/hutao/泡沫.mp3"
        if not target.resolve().is_relative_to(root.parent / "outputs"):
            raise ValueError("样例目标超出 outputs")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sample.read_bytes() != target.read_bytes():
                raise ValueError("样例目标已存在不同歌曲")
            sample.unlink()
        else:
            sample.rename(target)
        moves.append({"from": "胡桃/泡沫.mp3", "to": "../outputs/samples/hutao/泡沫.mp3"})
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()), key=lambda path: len(path.parts), reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()
    cache.save()
    result, _, _ = scan(root, import_models=True, cache_file=cache_file)
    result["normalized"] = {"movedFiles": len(moves), "deduplicatedFiles": len(duplicates), "moves": moves, "removedDuplicates": duplicates}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("scan", "normalize"))
    parser.add_argument("--root", type=Path, default=PROJECT / "models")
    parser.add_argument("--import", dest="import_models", action="store_true", help="生成或补全 model.json，不移动模型")
    parser.add_argument("--cache", type=Path, default=PROJECT / ".codex/model-catalog-cache.json")
    args = parser.parse_args()
    try:
        with redirect_stdout(sys.stderr):
            result = normalize(args.root, cache_file=args.cache) if args.command == "normalize" else scan(args.root, import_models=args.import_models, cache_file=args.cache)[0]
        print(json.dumps(result, ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"version": 1, "models": [], "diagnostics": [str(error)], "error": type(error).__name__}, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
