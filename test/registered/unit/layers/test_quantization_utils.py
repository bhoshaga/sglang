import unittest

from sglang.srt.layers.quantization.utils import is_layer_skipped


class TestQuantizationUtils(unittest.TestCase):
    def test_glob_patterns_match_exact_layer_prefixes(self):
        self.assertTrue(
            is_layer_skipped(
                "model.layers.12.self_attn.o_proj",
                ["model.layers.*.self_attn.o_proj"],
            )
        )

    def test_plain_strings_preserve_substring_matching(self):
        self.assertTrue(
            is_layer_skipped(
                "model.layers.3.self_attn.q_proj",
                ["self_attn.q_proj"],
            )
        )

    def test_fused_mapping_honors_glob_patterns(self):
        self.assertTrue(
            is_layer_skipped(
                "model.layers.7.self_attn.qkv_proj",
                [
                    "model.layers.*.self_attn.q_proj",
                    "model.layers.*.self_attn.k_proj",
                    "model.layers.*.self_attn.v_proj",
                ],
                fused_mapping={"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
            )
        )

    def test_fused_mapping_rejects_partial_glob_match(self):
        with self.assertRaises(ValueError):
            is_layer_skipped(
                "model.layers.7.self_attn.qkv_proj",
                ["model.layers.*.self_attn.q_proj"],
                fused_mapping={"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
            )

    def test_non_matching_glob_is_not_skipped(self):
        self.assertFalse(
            is_layer_skipped(
                "model.layers.12.self_attn.k_proj",
                ["model.layers.*.self_attn.o_proj"],
            )
        )

    def test_gate_up_proj_respects_globbed_gate_and_up_entries(self):
        self.assertTrue(
            is_layer_skipped(
                "model.layers.5.mlp.gate_up_proj",
                [
                    "model.layers.*.mlp.gate_proj",
                    "model.layers.*.mlp.up_proj",
                ],
            )
        )


if __name__ == "__main__":
    unittest.main()
