import types
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM


class DummyQwen35Config:
    model_type = "qwen3_5_text"

    def __init__(self):
        self.num_hidden_layers = 4
        self.layers_block_type = [
            "linear_attention",
            "attention",
            "linear_attention",
            "attention",
        ]


class DummyAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 1, bias=False)
        self.attn = types.SimpleNamespace(
            k_scale=nn.Parameter(torch.tensor(-1.0), requires_grad=False),
            v_scale=nn.Parameter(torch.tensor(-1.0), requires_grad=False),
            k_scale_float=None,
            v_scale_float=None,
        )


class FakeQwen35Model:
    def __init__(self, params):
        self._params = params

    def named_parameters(self, remove_duplicate=False):
        del remove_duplicate
        return self._params.items()


class TestQwen35KVCacheScaleLoading(unittest.TestCase):
    def test_load_weights_remaps_kv_scale_names(self):
        param = MagicMock()
        param.weight_loader = MagicMock()
        fake_model = FakeQwen35Model({"model.layers.1.attn.k_scale": param})

        loaded_params = Qwen3_5ForCausalLM.load_weights(
            fake_model,
            [
                (
                    "model.language_model.layers.1.self_attn.k_scale",
                    torch.tensor(1.25),
                )
            ],
        )

        param.weight_loader.assert_called_once()
        call_args = param.weight_loader.call_args[0]
        self.assertIs(call_args[0], param)
        self.assertEqual(call_args[1].item(), 1.25)
        self.assertIn("model.layers.1.attn.k_scale", loaded_params)

    @patch("sglang.srt.models.qwen3_5.get_tensor_model_parallel_rank", return_value=0)
    @patch("sglang.srt.models.qwen3_5.get_tensor_model_parallel_world_size", return_value=1)
    @patch(
        "sglang.srt.models.qwen3_5.kv_cache_scales_loader",
        return_value=[
            (0, 0.1, 0.2),
            (1, 0.3, 0.4),
            (2, 0.5, 0.6),
            (3, 0.7, 0.8),
        ],
    )
    def test_load_kv_cache_scales_skips_linear_attention_layers(
        self, mock_loader, mock_tp_size, mock_tp_rank
    ):
        layer_one = DummyAttentionLayer()
        layer_three = DummyAttentionLayer()
        fake_model = types.SimpleNamespace(
            config=DummyQwen35Config(),
            layers=[nn.Identity(), layer_one, nn.Identity(), layer_three],
        )

        Qwen3_5ForCausalLM.load_kv_cache_scales(fake_model, "scales.json")

        mock_loader.assert_called_once_with(
            "scales.json",
            0,
            1,
            4,
            "qwen3_5_text",
        )
        self.assertAlmostEqual(layer_one.attn.k_scale.item(), 0.3)
        self.assertAlmostEqual(layer_one.attn.v_scale.item(), 0.4)
        self.assertAlmostEqual(layer_three.attn.k_scale.item(), 0.7)
        self.assertAlmostEqual(layer_three.attn.v_scale.item(), 0.8)
        self.assertEqual(layer_one.attn.k_scale_float, 0.3)
        self.assertEqual(layer_one.attn.v_scale_float, 0.4)
        self.assertEqual(layer_three.attn.k_scale_float, 0.7)
        self.assertEqual(layer_three.attn.v_scale_float, 0.8)
        self.assertEqual(mock_tp_size.call_count, 1)
        self.assertEqual(mock_tp_rank.call_count, 1)
