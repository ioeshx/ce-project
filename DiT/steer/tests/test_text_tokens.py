"""Check text attention sites against native SD3 joint attention and projection."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from diffusers import SD3Transformer2DModel

from benchmark.common import read_json
from DiT.steer import method, run


TARGETS = ("text_token", "text_token_output")


def tiny_model():
    return SD3Transformer2DModel(sample_size=4, patch_size=1, in_channels=4, out_channels=4,
        num_layers=3, attention_head_dim=4, num_attention_heads=2, joint_attention_dim=8,
        caption_projection_dim=8, pooled_projection_dim=8, pos_embed_max_size=4,
        dual_attention_layers=(0,), qk_norm="rms_norm").eval()


def text_attention(attn, image, text):
    """Full text-query attention, including image and text values."""
    def heads(x):
        return x.reshape(len(x), -1, attn.heads, 4).transpose(1, 2)

    qi = attn.norm_q(heads(attn.to_q(image)))
    qt = attn.norm_added_q(heads(attn.add_q_proj(text)))
    ki = attn.norm_k(heads(attn.to_k(image)))
    kt = attn.norm_added_k(heads(attn.add_k_proj(text)))
    values = torch.cat((heads(attn.to_v(image)), heads(attn.add_v_proj(text))), dim=2)
    result = F.scaled_dot_product_attention(torch.cat((qi, qt), dim=2), torch.cat((ki, kt), dim=2), values)
    result = result.transpose(1, 2).reshape(len(text), image.shape[1] + text.shape[1], -1)
    return result[:, image.shape[1]:]


class TextTokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    @torch.inference_mode()
    def test_sites_match_attention_and_projection_and_only_edit_conditional_text(self):
        for target in TARGETS:
            for average in (True, False):
                for cfg in (True, False):
                    for dtype in (torch.float32, torch.bfloat16):
                        with self.subTest(target=target, average=average, cfg=cfg, dtype=dtype):
                            torch.manual_seed(6)
                            model = tiny_model().to(dtype)
                            attn = model.transformer_blocks[0].attn
                            # A nonzero bias distinguishes the post-projection site.
                            attn.to_add_out.bias.fill_(0.25)
                            image, text = torch.randn(4, 7, 8).to(dtype), torch.randn(4, 5, 8).to(dtype)
                            baseline_image, baseline_text = attn(image, text)
                            raw = text_attention(attn, image, text)
                            torch.testing.assert_close(attn.to_add_out(raw), baseline_text)
                            feature = raw if target == "text_token" else baseline_text
                            conditional = slice(2, None) if cfg else slice(None)
                            hooks = method.SD3Steering(model, 1, cfg=cfg, steer_target=target, average_patches=average)
                            hooks.step = 0
                            try:
                                actual_image, actual_text = attn(image, text)
                                torch.testing.assert_close(actual_image, baseline_image, rtol=0, atol=0)
                                torch.testing.assert_close(actual_text, baseline_text, rtol=0, atol=0)
                                expected_mean = feature[conditional].float().mean(0)
                                if average:
                                    expected_mean = expected_mean.mean(0)
                                torch.testing.assert_close(hooks.sums[0, 0], expected_mean)
                                self.assertEqual(hooks.token_counts[0], (0, 5))
                                self.assertEqual(hooks.token_counts[-1], (0, 0))
                                self.assertEqual(hooks.sums[0, -1].count_nonzero().item(), 0)
                            finally:
                                hooks.close()

                            shape = (2, 1, 3, 8) if average else (2, 1, 3, 5, 8)
                            vectors = F.normalize(torch.randn(shape), dim=-1)
                            hooks = method.SD3Steering(model, 1, cfg=cfg, vectors=vectors, beta=-0.7,
                                steer_target=target, average_patches=average, trace_score=True)
                            hooks.step = 0
                            try:
                                expected = feature.clone()
                                expected[conditional] = method.steer_output(feature[conditional], vectors[:, 0, 0], -0.7)
                                if target == "text_token":
                                    expected = F.linear(expected, attn.to_add_out.weight, attn.to_add_out.bias)
                                actual_image, actual_text = attn(image, text)
                                torch.testing.assert_close(actual_image, baseline_image, rtol=0, atol=0)
                                torch.testing.assert_close(actual_text, expected)
                                if cfg:
                                    torch.testing.assert_close(actual_text[:2], baseline_text[:2],
                                                               atol=0.002 if dtype == torch.bfloat16 else 1e-7,
                                                               rtol=0.016 if dtype == torch.bfloat16 else 1e-6)
                                self.assertEqual(hooks.score_maps.shape, (2, 1, 3, 2 if cfg else 4, 5))
                            finally:
                                hooks.close()

    @torch.inference_mode()
    def test_full_transformer_zero_beta_and_hook_cleanup(self):
        torch.manual_seed(8)
        model = tiny_model()
        inputs = dict(hidden_states=torch.randn(4, 4, 4, 4), encoder_hidden_states=torch.randn(4, 5, 8),
                      pooled_projections=torch.randn(4, 8), timestep=torch.ones(4), return_dict=False)
        original = model(**inputs)[0]
        processors = list(model.attn_processors.values())
        for target in TARGETS:
            hooks = method.SD3Steering(model, 1, cfg=True, vectors=F.normalize(torch.randn(2, 1, 3, 8), dim=-1),
                beta=0, steer_target=target, steer_pool=True, pool_vectors=torch.randn(2, 8))
            try:
                torch.testing.assert_close(model(**inputs)[0], original, rtol=0, atol=0)
            finally:
                hooks.close()
            self.assertEqual(list(model.attn_processors.values()), processors)
            torch.testing.assert_close(model(**inputs)[0], original, rtol=0, atol=0)
            self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()))

    def test_fit_save_load_and_text_score_layer_ids(self):
        class Adapter:
            def __init__(self, config):
                self.config, self.dtype, self.device = config, torch.float32, torch.device("cpu")
                self.pipe = self
                self.transformer, self.vae_scale_factor = tiny_model(), 8
                self.scheduler = SimpleNamespace(timesteps=torch.tensor([2, 1]))

            def __call__(self, *, prompt, latents, num_inference_steps, **unused):
                generator = torch.Generator().manual_seed(sum(map(ord, prompt[0])))
                context = torch.randn(1, 5, 8, generator=generator)
                pooled = torch.randn(1, 8, generator=generator)
                for step in range(num_inference_steps):
                    noise = self.transformer(hidden_states=torch.cat((latents, latents)),
                        encoder_hidden_states=torch.cat((torch.zeros_like(context), context)),
                        pooled_projections=torch.cat((torch.zeros_like(pooled), pooled)),
                        timestep=torch.full((2,), 2 - step), return_dict=False)[0]
                    a, b = noise.chunk(2)
                    latents = latents - 0.1 * (a + 4.5 * (b - a))
                return SimpleNamespace(images=latents)

            def generate(self, records, sampling):
                return self(prompt=[records[0]["prompt"]], latents=torch.ones(1, 4, 4, 4),
                            num_inference_steps=sampling["steps"]).images

            def close(self):
                pass

        sampling = dict(steps=2, height=32, width=32, guidance_scale=4.5, max_sequence_length=16)
        with tempfile.TemporaryDirectory() as temp:
            for target in TARGETS:
                for average in (True, False):
                    checkpoint = Path(temp) / f"{target}-{average}"
                    settings = dict(group="ip", targets=["Snoopy", "Pikachu"], anchors=["", ""], seed=0,
                        calibration_prompts=2, steer_target=target, average_patches=average,
                        construction=dict(method="mean", normalize_differences=True))
                    config = dict(sampling=sampling, adapter={})
                    with patch.object(method, "DiTAdapter", Adapter), contextlib.redirect_stdout(io.StringIO()):
                        method.fit(checkpoint, settings, config)
                    metadata = read_json(checkpoint / "metadata.json")
                    self.assertEqual(metadata["position_axis"], "text token")
                    self.assertEqual(metadata["token_counts"], [[0, 5], [0, 5], [0, 0]])
                    state = torch.load(checkpoint / "steering.pt", weights_only=True)
                    self.assertEqual(state["vectors"].shape, (2, 2, 3, 8) if average else (2, 2, 3, 5, 8))
                    self.assertEqual(state["vectors"][:, :, -1].count_nonzero().item(), 0)
                    config.update(task="instance", case_id="ip_snoopy_pikachu", targets=settings["targets"])
                    config["adapter"] = dict(checkpoint=str(checkpoint), steer_target=target,
                        average_patches=average, beta=-2, clipping=True, trace_score=True)
                    records = [dict(prompt="Snoopy and Pikachu", concept="Pikachu", sample_id="one", seed=0,
                                    image=str(checkpoint / "images/one.png"))]
                    with patch.object(method, "DiTAdapter", Adapter), \
                         patch("DiT.steer.diagnostics.accumulate_scores") as accumulate, \
                         patch("DiT.steer.diagnostics.write_score_report"):
                        adapter = method.SteerAdapter(copy.deepcopy(config))
                        adapter.generate(records, sampling)
                        adapter.close()
                    call = accumulate.call_args
                    self.assertEqual(call.kwargs["position_kind"], "text")
                    self.assertEqual(call.kwargs["layer_ids"], [0, 1])
                    self.assertEqual(call.args[2].shape, (2, 2, 2, 1, 5))

    def test_cli_targets_have_separate_identity_and_keep_benchmark_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            plans = []
            for target in ("text_to_image", *TARGETS):
                with patch.object(run, "run") as execute:
                    run.main(["--cases", "ip_snoopy_pikachu", "--steer-target", target,
                              "--device", "cpu", "--dry-run", "--run-root", temp])
                args, experiments = execute.call_args.args[:2]
                _, settings, configs = experiments[0]
                plans.append((args, settings, configs[0]))
            self.assertEqual(len({p[0].method for p in plans}), 3)
            self.assertEqual(len({p[2]["adapter"]["checkpoint"] for p in plans}), 3)
            for _, _, config in plans[1:]:
                for field in ("dataset", "sampling", "evaluation", "targets"):
                    self.assertEqual(config[field], plans[0][2][field])


if __name__ == "__main__":
    unittest.main()
