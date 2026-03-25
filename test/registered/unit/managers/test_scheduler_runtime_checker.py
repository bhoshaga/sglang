import types
import unittest

from sglang.srt.managers.scheduler_runtime_checker_mixin import (
    SchedulerRuntimeCheckerMixin,
)


class TestSchedulerRuntimeChecker(unittest.TestCase):
    def test_waiting_queue_mamba_slot_is_not_reported_as_leak(self):
        fake_scheduler = types.SimpleNamespace(
            waiting_queue=[
                types.SimpleNamespace(req_pool_idx=None, mamba_pool_idx=10),
            ],
            tree_cache=types.SimpleNamespace(
                full_protected_size=lambda: 0,
                mamba_protected_size=lambda: 0,
                all_values_flatten=lambda: types.SimpleNamespace(tolist=lambda: []),
                all_mamba_values_flatten=lambda: types.SimpleNamespace(tolist=lambda: []),
            ),
            token_to_kv_pool_allocator=types.SimpleNamespace(
                size=125792,
                free_pages=types.SimpleNamespace(tolist=lambda: list(range(1, 705))),
                release_pages=types.SimpleNamespace(
                    tolist=lambda: list(range(705, 125793))
                ),
            ),
            req_to_token_pool=types.SimpleNamespace(
                mamba_pool=types.SimpleNamespace(
                    size=24,
                    free_slots=types.SimpleNamespace(
                        tolist=lambda: [
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            11,
                            12,
                            13,
                            14,
                            15,
                            16,
                            17,
                            18,
                            19,
                            20,
                            21,
                            22,
                            23,
                            24,
                        ]
                    ),
                )
            ),
            _get_mamba_token_info=lambda: (0, 1, 0.0, 0.0, 704, 125088, 23, 0),
            _session_held_tokens=lambda: 0,
        )

        memory_leak, token_msg = SchedulerRuntimeCheckerMixin._check_mamba_memory(
            fake_scheduler
        )

        self.assertFalse(memory_leak, token_msg)


if __name__ == "__main__":
    unittest.main()
