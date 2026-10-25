"""The shared profile must reuse the verified legacy COCO cache."""
import copy
import unittest

from benchmark.common import configuration, digest
from benchmark.data import prompt_records, sampling_protocol
from benchmark.sampling_profile import PROFILE, initial_noise, make_scheduler


class SamplingProfileTests(unittest.TestCase):
    def test_exact_legacy_coco_cache_identity(self):
        config = configuration(task="mscoco")
        records, _ = prompt_records(config)
        key = digest({"records": records, "model": config["base_model"],
                      "sampling": sampling_protocol(config)})
        self.assertEqual(key, "a971b6e6c75f9e45043ce7d80f141f862403476e289ae6bf23441849dad86351")
        config["sampling"].update(model_revision=None, allow_tf32=False,
                                   float32_matmul_precision="highest")
        self.assertEqual(sampling_protocol(config), PROFILE)

    def test_substantive_overrides_do_not_reuse_default(self):
        base = configuration(task="mscoco")
        for key, value in [("steps", 21), ("dtype", "float32"), ("scheduler", "ddim"),
                           ("noise", "direct float16 noise"), ("model_revision", "other")]:
            config = copy.deepcopy(base)
            config["sampling"][key] = value
            self.assertNotEqual(sampling_protocol(config), sampling_protocol(base))

    def test_initial_noise_matches_float32_then_cast(self):
        import torch
        for seed in [0, 1, 999]:
            expected = torch.randn((4, 8, 8), generator=torch.Generator().manual_seed(seed),
                                   dtype=torch.float32).half()
            actual = initial_noise((4, 8, 8), seed, dtype=torch.float16, device="cpu")
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_pinned_scheduler_and_twenty_steps(self):
        from diffusers import DDIMScheduler
        scheduler = make_scheduler(DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                                                beta_schedule="scaled_linear").config)
        self.assertEqual(type(scheduler).__name__, "DPMSolverMultistepScheduler")
        for key in ["algorithm_type", "solver_order", "solver_type", "steps_offset"]:
            self.assertEqual(scheduler.config[key], PROFILE["scheduler_config"][key])
        scheduler.set_timesteps(PROFILE["steps"])
        self.assertEqual(len(scheduler.timesteps), 20)

    def test_sampler_preserves_legacy_attention_processor(self):
        from unittest.mock import MagicMock, patch
        from diffusers import DDIMScheduler
        from benchmark.adapters import DiffusersAdapter
        fake = MagicMock()
        fake.scheduler.config = DDIMScheduler().config
        config = configuration(task="mscoco")
        config["sampling"].update(device="cpu", dtype="float32")
        with patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=fake):
            adapter = DiffusersAdapter(config)
        fake.enable_attention_slicing.assert_called_once_with()
