"""Real-library coverage for the shared Transformers and PEFT runtime."""

from io import BytesIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn


TEMP_ROOT = Path(__file__).resolve().parents[2] / ".codex" / "tmp"


class TinyCFM(nn.Module):
    """Use the same projection names targeted by SoVITS LoRA checkpoints."""

    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(8, 8)
        self.to_k = nn.Linear(8, 8)
        self.to_v = nn.Linear(8, 8)
        self.to_out = nn.Sequential(nn.Linear(8, 8))

    def forward(self, value):
        return self.to_out(self.to_q(value) + self.to_k(value) + self.to_v(value))


class SharedDependencyTests(unittest.TestCase):
    def test_bert_runtime_uses_full_precision_for_half_checkpoint_on_cpu(self):
        from transformers import BertConfig, BertModel, BertTokenizer

        from GPT_SoVITS.inference.model_runtime import ModelRuntimeMixin

        config = BertConfig(
            vocab_size=7,
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=16,
        )
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with torch.random.fork_rng(), tempfile.TemporaryDirectory(
            prefix="bert-precision-", dir=TEMP_ROOT
        ) as directory:
            torch.manual_seed(43)
            original = BertModel(config, add_pooling_layer=False).half().eval()
            original.save_pretrained(directory)
            tokenizer = BertTokenizer(vocab={
                "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
                "[MASK]": 4, "你": 5, "好": 6,
            })
            tokenizer.save_pretrained(directory)
            runtime = ModelRuntimeMixin()
            runtime.configs = SimpleNamespace(device="cpu", is_half=False)
            runtime.init_bert_weights(directory)
            self.assertEqual(next(runtime.bert_model.parameters()).dtype, torch.float32)
            original.float()
            inputs = runtime.bert_tokenizer("你好", return_tensors="pt")
            with torch.inference_mode():
                expected = original(**inputs).last_hidden_state
                actual = runtime.bert_model(**inputs).last_hidden_state
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cnhubert_loads_local_checkpoint_and_preserves_features(self):
        from transformers import HubertConfig, HubertModel, Wav2Vec2FeatureExtractor

        from GPT_SoVITS.feature_extractor.cnhubert import CNHubert

        config = HubertConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            conv_dim=(8, 8, 8),
            conv_stride=(5, 2, 2),
            conv_kernel=(10, 3, 3),
            feat_extract_norm="layer",
            num_conv_pos_embeddings=8,
            num_conv_pos_embedding_groups=2,
            mask_time_prob=0,
        )
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with torch.random.fork_rng(), tempfile.TemporaryDirectory(
            prefix="cnhubert-", dir=TEMP_ROOT
        ) as directory:
            torch.manual_seed(42)
            # Official CNHuBERT checkpoints store half-precision tensors. The
            # pipeline's is_half option, rather than that storage dtype, owns
            # the runtime precision.
            original = HubertModel(config).half().eval()
            extractor = Wav2Vec2FeatureExtractor(sampling_rate=16000)
            original.save_pretrained(directory)
            extractor.save_pretrained(directory)
            restored = CNHubert(directory).eval()
            original.float()
            self.assertEqual(next(restored.parameters()).dtype, torch.float32)
            waveform = torch.linspace(-0.75, 0.75, 1600)
            inputs = extractor(waveform, sampling_rate=16000, return_tensors="pt").input_values
            with torch.inference_mode():
                expected = original(inputs).last_hidden_state
                actual = restored(waveform)
            self.assertEqual(actual.shape[0], 1)
            self.assertGreater(actual.shape[1], 0)
            self.assertEqual(actual.shape[2], config.hidden_size)
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_sovits_lora_checkpoint_load_and_merge_preserve_output(self):
        from peft import LoraConfig, get_peft_model

        def inject(module):
            return get_peft_model(
                module,
                LoraConfig(
                    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                    r=2,
                    lora_alpha=2,
                    init_lora_weights=True,
                ),
            )

        with torch.random.fork_rng():
            torch.manual_seed(73)
            trained = nn.ModuleDict({"cfm": inject(TinyCFM())}).eval()
            with torch.no_grad():
                for name, parameter in trained.named_parameters():
                    if "lora_" in name:
                        parameter.uniform_(-0.2, 0.2)
            checkpoint = BytesIO()
            torch.save(trained.state_dict(), checkpoint)
            checkpoint.seek(0)
            restored = nn.ModuleDict({"cfm": inject(TinyCFM())}).eval()
            restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
            value = torch.randn(2, 3, 8)
            with torch.inference_mode():
                expected = trained["cfm"](value)
                before_merge = restored["cfm"](value)
                restored["cfm"] = restored["cfm"].merge_and_unload()
                after_merge = restored["cfm"](value)
            torch.testing.assert_close(before_merge, expected, rtol=0, atol=0)
            torch.testing.assert_close(after_merge, expected, rtol=1e-5, atol=1e-6)
            self.assertFalse(any("lora_" in name for name in restored.state_dict()))
            self.assertIsInstance(restored["cfm"].to_out[0], nn.Linear)


if __name__ == "__main__":
    unittest.main()
