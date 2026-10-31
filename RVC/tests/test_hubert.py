"""Exercise the real local Transformers checkpoint and RVC feature paths."""

import pytest
import torch
from transformers import HubertConfig

from rvc.vc.hubert import (
    HubertModelWithFinalProj,
    extract_hubert_features,
    load_hubert_model,
)


@pytest.fixture(scope="module")
def hubert_checkpoint(tmp_path_factory):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = HubertConfig(
            hidden_size=16,
            num_hidden_layers=12,
            num_attention_heads=2,
            intermediate_size=32,
            conv_dim=(8, 8),
            conv_stride=(5, 2),
            conv_kernel=(10, 3),
            num_conv_pos_embeddings=16,
            num_conv_pos_embedding_groups=2,
            classifier_proj_size=8,
            hidden_dropout=0,
            activation_dropout=0,
            attention_dropout=0,
            feat_proj_dropout=0,
            final_dropout=0,
            layerdrop=0,
            mask_time_prob=0,
            mask_feature_prob=0,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(319)
            original = HubertModelWithFinalProj(config).eval()
            source = torch.randn(1, 512)
        with torch.no_grad():
            original.final_proj.bias.copy_(torch.arange(8, dtype=torch.float32) / 10)
        directory = tmp_path_factory.mktemp("tiny-hubert")
        original.save_pretrained(directory)
        loaded = load_hubert_model(directory, torch.device("cpu"))
        yield original, loaded, source
    finally:
        torch.set_num_threads(previous_threads)


def test_local_hubert_loading_preserves_projection_and_weights(hubert_checkpoint):
    original, loaded, _ = hubert_checkpoint
    assert not loaded.training
    assert loaded.config.num_hidden_layers == 12
    assert loaded.final_proj.out_features == 8
    assert next(loaded.parameters()).dtype == torch.float32
    assert loaded.state_dict().keys() == original.state_dict().keys()
    for name, expected in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], expected, rtol=0, atol=0)


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("with_padding", [False, True])
def test_hubert_features_match_the_loaded_checkpoint(hubert_checkpoint, version, with_padding):
    original, loaded, source = hubert_checkpoint
    padding_mask = None
    attention_mask = None
    if with_padding:
        padding_mask = torch.zeros_like(source, dtype=torch.bool)
        padding_mask[:, -64:] = True
        attention_mask = (~padding_mask).long()
    with torch.inference_mode():
        outputs = original(
            input_values=source,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        expected = (
            original.final_proj(outputs.hidden_states[9])
            if version == "v1" else outputs.last_hidden_state
        )
        actual = extract_hubert_features(loaded, source, version, padding_mask)
    assert actual.shape[-1] == (8 if version == "v1" else 16)
    assert actual.shape[1] > 0
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
