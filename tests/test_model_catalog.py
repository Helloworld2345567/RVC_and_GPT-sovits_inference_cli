import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("model_catalog", ROOT / "scripts/model_catalog.py")
catalog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog)


def rvc(directory, name="voice.pth", version="v2"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    torch.save({"weight": {"emb_g.weight": torch.zeros(2, 3)}, "config": [0] * 17 + [48000], "version": version, "f0": 1}, path)
    return path


def gpt(directory, name="voice.ckpt"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    torch.save({"config": {"model": {}}, "weight": {"model.ar_text_embedding.word_embeddings.weight": torch.zeros(2, 3), "model.ar_audio_embedding.word_embeddings.weight": torch.zeros(2, 3)}}, path)
    return path


def sovits(directory, name="voice.pth"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    torch.save({"config": {"model": {}}, "weight": {"ref_enc.weight": torch.zeros(2, 3)}, "lora_rank": 32}, path)
    with path.open("r+b") as stream:
        stream.write(b"04")
    return path


def reference(directory, name="【默认】你好.wav", seconds=4):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(b"\x01\x00" * (24000 * seconds))
    return path


class CatalogTests(unittest.TestCase):
    def setUp(self):
        temporary = ROOT / ".codex/model-catalog-task"
        temporary.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="test-", dir=temporary)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "models"
        self.root.mkdir()
        self.cache = Path(self.temporary.name) / "cache.json"

    def scan(self):
        return catalog.scan(self.root, import_models=True, cache_file=self.cache)

    def test_real_structures_classify_pth_and_gpt_without_extension_guess(self):
        first = rvc(self.root / "胡桃")
        second = sovits(self.root / "可莉")
        third = gpt(self.root / "可莉")
        separator = self.root / "separation.ckpt"
        torch.save({"state_dict": {"weights": torch.zeros(1)}}, separator)
        self.assertEqual(catalog.inspect_checkpoint(first)["kind"], "rvc")
        self.assertEqual(catalog.inspect_checkpoint(second)["kind"], "sovits")
        self.assertEqual(catalog.inspect_checkpoint(third)["kind"], "gpt")
        self.assertEqual(catalog.inspect_checkpoint(separator)["kind"], "unknown")

    def test_import_persists_identity_and_repeat_scan_uses_cache(self):
        rvc(self.root / "可莉")
        report, _, first = self.scan()
        selected = report["models"][0]
        self.assertEqual(selected["state"], "ready")
        self.assertEqual(selected["id"], "klee-v2")
        self.assertGreater(first.hash_reads, 0)
        with patch.object(catalog, "inspect_checkpoint", side_effect=AssertionError("should use cached metadata")):
            repeated, _, cached = self.scan()
        self.assertEqual(repeated, report)
        self.assertEqual(cached.hash_reads, 0)
        self.assertEqual(cached.probe_reads, 0)
        (self.root / "可莉").rename(self.root / "renamed")
        moved, _, _ = self.scan()
        self.assertEqual(moved["models"][0]["id"], selected["id"])
        self.assertEqual(moved["models"][0]["resourceHash"], selected["resourceHash"])

    def test_missing_pair_or_ambiguous_reference_is_blocked(self):
        folder = self.root / "可莉"
        sovits(folder)
        first, _, _ = self.scan()
        self.assertEqual(first["models"][0]["state"], "blocked")
        gpt(folder)
        reference(folder)
        reference(folder, "【开心】再见.wav")
        second, _, _ = self.scan()
        self.assertEqual(second["models"][0]["id"], first["models"][0]["id"])
        self.assertTrue(any("多个参考" in issue for issue in second["models"][0]["diagnostics"]))
        manifest = json.loads((folder / "model.json").read_text(encoding="utf-8"))
        manifest["files"]["reference"] = "【默认】你好.wav"
        catalog.write_json(folder / "model.json", manifest)
        third, _, _ = self.scan()
        self.assertEqual(third["models"][0]["state"], "ready")
        self.assertEqual(third["models"][0]["reference"]["text"], "你好")

    def test_only_changed_resource_updates_its_identity_logs_do_not(self):
        first = rvc(self.root / "可莉")
        rvc(self.root / "胡桃")
        before, _, _ = self.scan()
        (first.parent / "train.log").write_text("noise")
        unchanged, _, _ = self.scan()
        self.assertEqual(before, unchanged)
        with first.open("ab") as output:
            output.write(b"new-resource")
        changed, _, _ = self.scan()
        hashes = lambda report: {item["id"]: item["resourceHash"] for item in report["models"]}
        self.assertNotEqual(hashes(changed)["klee-v2"], hashes(before)["klee-v2"])
        self.assertEqual(hashes(changed)["hutao-v2"], hashes(before)["hutao-v2"])

    def test_index_dimension_and_escaping_manifest_are_rejected(self):
        folder = self.root / "胡桃"
        rvc(folder)
        (folder / "voice.index").write_bytes(b"IwFl" + struct.pack("<i", 256))
        report, _, _ = self.scan()
        self.assertEqual(report["models"][0]["state"], "blocked")
        self.assertTrue(any("维度" in error for error in report["models"][0]["diagnostics"]))
        manifest = json.loads((folder / "model.json").read_text(encoding="utf-8"))
        manifest["files"]["weights"] = "../../outside.pth"
        catalog.write_json(folder / "model.json", manifest)
        blocked, _, _ = self.scan()
        self.assertEqual(blocked["models"][0]["state"], "blocked")

    def test_normalize_preserves_reference_text_and_deduplicates_equal_models(self):
        folder, duplicate = self.root / "可莉/gpt-sovits", self.root / "v4/gpt-sovits"
        sovits(folder, "可莉.pth")
        gpt(folder, "可莉.ckpt")
        reference(folder)
        duplicate.mkdir(parents=True)
        for file in folder.iterdir():
            (duplicate / file.name).write_bytes(file.read_bytes())
        (folder / "train.log").write_text("training")
        self.scan()
        manifest = json.loads((folder / "model.json").read_text(encoding="utf-8"))
        catalog.write_json(duplicate / "model.json", manifest)
        blocked, _, _ = self.scan()
        self.assertEqual(blocked["models"][0]["state"], "blocked")
        result = catalog.normalize(self.root, cache_file=self.cache)
        self.assertEqual(len(result["models"]), 1)
        self.assertEqual(result["normalized"]["deduplicatedFiles"], 3)
        selected = result["models"][0]
        self.assertEqual(selected["files"]["sovits"], "gpt-sovits/klee/v4-zh/sovits.pth")
        self.assertEqual(selected["reference"]["text"], "你好")
        self.assertTrue((self.root / "gpt-sovits/klee/v4-zh/metadata/train.log").exists())
        again = catalog.normalize(self.root, cache_file=self.cache)
        self.assertEqual(again["models"], result["models"])
        self.assertEqual(again["normalized"]["movedFiles"], 0)

    def test_normalize_different_contents_with_duplicate_ids_keeps_unique_variants(self):
        first, second = self.root / "可莉/rvc", self.root / "v4/rvc"
        source = rvc(first, "可莉.pth")
        self.scan()
        second.mkdir(parents=True)
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        checkpoint["weight"]["emb_g.weight"] = torch.ones(2, 3)
        torch.save(checkpoint, second / "可莉.pth")
        manifest = json.loads((first / "model.json").read_text(encoding="utf-8"))
        catalog.write_json(second / "model.json", manifest)
        blocked, details, _ = self.scan()
        self.assertEqual(blocked["models"][0]["state"], "blocked")
        expected_resources = {model["resourceHash"] for _, _, model in details}
        result = catalog.normalize(self.root, cache_file=self.cache)
        self.assertEqual(len(result["models"]), 2)
        self.assertEqual({model["resourceHash"] for model in result["models"]}, expected_resources)
        self.assertEqual(len({model["id"] for model in result["models"]}), 2)
        self.assertTrue(all(model["state"] == "ready" for model in result["models"]))
        locations = [model["files"]["weights"] for model in result["models"]]
        self.assertIn("rvc/klee/v2/model.pth", locations)
        self.assertTrue(any(location.startswith("rvc/klee/v2-") for location in locations))
        for model in result["models"]:
            package = self.root / model["files"]["weights"]
            self.assertEqual(json.loads((package.parent / "model.json").read_text(encoding="utf-8"))["id"], model["id"])
        repeated = catalog.normalize(self.root, cache_file=self.cache)
        self.assertEqual(repeated["models"], result["models"])
        self.assertEqual(repeated["normalized"]["movedFiles"], 0)

    def test_duplicate_id_does_not_hide_invalid_resource_during_normalize(self):
        source = rvc(self.root / "可莉")
        self.scan()
        duplicate = self.root / "copied-klee"
        duplicate.mkdir()
        (duplicate / source.name).write_bytes(source.read_bytes())
        manifest = json.loads((source.parent / "model.json").read_text(encoding="utf-8"))
        manifest["files"]["index"] = "missing.index"
        catalog.write_json(duplicate / "model.json", manifest)
        with self.assertRaisesRegex(ValueError, "缺项或歧义"):
            catalog.normalize(self.root, cache_file=self.cache)
        self.assertTrue(source.is_file())
        self.assertTrue((duplicate / source.name).is_file())

    def test_incomplete_gpt_only_directory_remains_visible(self):
        gpt(self.root / "派蒙")
        result, _, _ = self.scan()
        self.assertEqual(len(result["models"]), 1)
        self.assertEqual(result["models"][0]["serviceId"], "gpt-sovits")
        self.assertEqual(result["models"][0]["state"], "blocked")

    def test_duplicate_manifest_id_blocks_only_that_model_and_returns_unique_ids(self):
        rvc(self.root / "可莉")
        rvc(self.root / "胡桃")
        self.scan()
        duplicate = self.root / "copied-klee"
        duplicate.mkdir()
        for source in (self.root / "可莉").iterdir():
            (duplicate / source.name).write_bytes(source.read_bytes())
        report, _, _ = self.scan()
        models = {model["id"]: model for model in report["models"]}
        self.assertEqual(len(report["models"]), 2)
        self.assertEqual(models["klee-v2"]["state"], "blocked")
        self.assertEqual(models["hutao-v2"]["state"], "ready")


if __name__ == "__main__":
    unittest.main()
