"""Check queryGraft edit sites with small real SD3/FLUX Transformers on CPU."""
import copy
from functools import partial
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from diffusers import FluxTransformer2DModel, SD3Transformer2DModel

from benchmark.dit_adapter import DiTAdapter
from DiT.queryGraft.method import QueryGraft, resolve_settings


class Tokenizer:
    """Small deterministic tokenizer; actual CLIP/T5 are checked separately."""
    def __call__(self, texts, *, add_special_tokens=True, padding=None, max_length=None,
                 truncation=False, return_tensors=None, **kwargs):
        rows, masks, specials = [], [], []
        for text in texts:
            content = [3 + sum(map(ord, token)) for token in re.findall(r"\w+|[^\w\s]", text)]
            if truncation:
                content = content[:max_length - (2 if add_special_tokens else 0)]
            ids = [1, *content, 2] if add_special_tokens else content
            special = [1, *([0] * len(content)), 1] if add_special_tokens else [0] * len(content)
            size = len(ids)
            if padding == "max_length":
                ids += [0] * (max_length - size)
                special += [1] * (max_length - size)
            rows.append(ids)
            masks.append([1] * size + [0] * (len(ids) - size))
            specials.append(special)
        result = dict(input_ids=rows, attention_mask=masks, special_tokens_mask=specials)
        return {key: torch.tensor(value) for key, value in result.items()} if return_tensors else result


class QueryGraftTests(unittest.TestCase):
    def adapter(self, flux, representation, edit_layers="last", trace=False, eta=0, **source_settings):
        torch.manual_seed(7)
        if flux:
            transformer = FluxTransformer2DModel(
                in_channels=4, num_layers=1, num_single_layers=1,
                attention_head_dim=8, num_attention_heads=2,
                joint_attention_dim=16, pooled_projection_dim=8,
                axes_dims_rope=(2, 2, 4))
        else:
            transformer = SD3Transformer2DModel(
                sample_size=4, patch_size=2, in_channels=4, out_channels=4,
                num_layers=2, attention_head_dim=8, num_attention_heads=2,
                joint_attention_dim=16, caption_projection_dim=16,
                pooled_projection_dim=8, pos_embed_max_size=4,
                dual_attention_layers=(0, 1))
        transformer.eval()
        blocks = list(transformer.transformer_blocks)
        if flux:
            blocks += list(transformer.single_transformer_blocks)

        def embeddings(prompt, **kwargs):
            length = kwargs.get("max_sequence_length", 8)
            return torch.stack([torch.randn(length, 16, generator=torch.Generator().manual_seed(
                sum(map(ord, text)))) for text in prompt])

        def encode(prompt, **kwargs):
            sequence = embeddings(prompt)
            if not flux:
                sequence = torch.cat((embeddings(prompt, max_sequence_length=5), sequence), dim=1)
            pooled = torch.stack([torch.randn(8, generator=torch.Generator().manual_seed(
                sum(map(ord, text)) + 1)) for text in prompt])
            return (sequence, pooled, torch.zeros(sequence.shape[1], 3)) if flux else (sequence, None, pooled, None)

        pipe = SimpleNamespace(transformer=transformer, vae_scale_factor=1,
                               _get_t5_prompt_embeds=embeddings, encode_prompt=encode,
                               tokenizer=Tokenizer(), tokenizer_2=Tokenizer(), tokenizer_3=Tokenizer(),
                               tokenizer_max_length=5,
                               maybe_free_model_hooks=lambda: None)
        config = {"model_name": "flux1_dev" if flux else "sd35_medium", "targets": ["nudity"],
                  "sampling": {"height": 4, "width": 4, "max_sequence_length": 8}, "adapter": {}}
        config["adapter"]["query_graft"] = resolve_settings(config, {
            "representation": representation, "edit_layers": edit_layers, "trace": trace, "eta": eta,
            **source_settings})

        def initialize(adapter, native):
            adapter.pipe, adapter.flux = pipe, flux
            adapter.device, adapter.dtype = torch.device("cpu"), torch.float32
            adapter.config, adapter.runtime_info = copy.deepcopy(native), None

        # Capture native block outputs before the queryGraft output hook edits them.
        frames, norm_inputs, handles = {}, {}, []

        def capture(layer, module, args, output):
            context, image = output
            frames[graft.mode, layer] = (None if context is None else context.detach().clone(),
                                        image.detach().clone())

        def capture_norm(layer, module, args):
            norm_inputs[graft.mode, layer] = args[0].detach().clone()

        for layer, block in enumerate(blocks):
            handles.append(block.register_forward_hook(partial(capture, layer)))
            if hasattr(block, "norm2"):
                handles.append(block.norm2.register_forward_pre_hook(partial(capture_norm, layer)))
        with patch.object(DiTAdapter, "__init__", initialize):
            graft = QueryGraft(config)
        self.addCleanup(graft.close)
        for handle in handles:
            self.addCleanup(handle.remove)

        graft.batch, graft.forward_calls, graft.step = 2, 0, 0
        graft.step_traces = [[], []]
        batch = 2 if flux else 4  # SD3 native CFG order: negative, positive.
        call = dict(hidden_states=torch.randn(batch, 4, 4) if flux else torch.randn(batch, 4, 4, 4),
                    encoder_hidden_states=torch.randn(batch, 3, 16),
                    pooled_projections=torch.randn(batch, 8), timestep=torch.ones(batch), return_dict=False)
        if flux:
            call.update(txt_ids=torch.zeros(3, 3),
                        img_ids=torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]))
        return graft, blocks, call, frames, norm_inputs

    def test_block_output_replaces_final_image_states_only(self):
        for flux in (False, True):
            for edit_layers in ("last", "all"):
                with self.subTest(flux=flux, edit_layers=edit_layers):
                    graft, blocks, call, frames, norm_inputs = self.adapter(flux, "block_output", edit_layers)
                    outputs, handles = {}, []

                    def after_edit(layer, module, args, output):
                        context, image = output
                        outputs[graft.mode, layer] = (
                            None if context is None else context.detach().clone(), image.detach().clone())

                    for layer, block in enumerate(blocks):
                        handles.append(block.register_forward_hook(partial(after_edit, layer)))
                    for handle in handles:
                        self.addCleanup(handle.remove)
                    finish_mask = graft.finish_mask

                    def mixed_mask():
                        # Fixed spatial votes exercise both edited and untouched tokens.
                        graft.votes = torch.tensor([[[1, 0, 0, 1]], [[0, 1, 1, 0]]]) * len(graft.localizers)
                        finish_mask()

                    graft.active = True
                    with patch.object(graft, "finish_mask", side_effect=mixed_mask), torch.inference_mode():
                        graft.forward(**call)
                    mode = "user" if edit_layers == "last" else "edit"
                    mask = (graft.winner >= 0)[..., None]
                    for layer in range(len(blocks)):
                        context, before = frames[mode, layer]
                        actual_context, actual = outputs[mode, layer]
                        expected = before.clone()
                        if layer in graft.editors:
                            anchor = frames["anchor", layer][1]
                            expected[-graft.batch:] = torch.where(mask, anchor, before[-graft.batch:])
                            if ("anchor", layer) in norm_inputs:
                                # Native final states include the MLP residual update.
                                self.assertFalse(torch.allclose(anchor, norm_inputs["anchor", layer]))
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        if context is None:
                            self.assertIsNone(actual_context)
                        else:
                            torch.testing.assert_close(actual_context, context, rtol=0, atol=0)
                    self.assertEqual(graft.forward_calls, 2 if edit_layers == "last" else 3)

    def test_all_representations_match_native_when_mask_is_empty(self):
        for flux in (False, True):
            for representation in ("av", "residual", "block_output"):
                for edit_layers in ("last", "all"):
                    with self.subTest(flux=flux, representation=representation, edit_layers=edit_layers):
                        graft, _, call, _, _ = self.adapter(flux, representation, edit_layers, eta=1)
                        with torch.inference_mode():
                            native = graft.pipe.transformer(**call)[0]
                            graft.active = True
                            actual = graft.forward(**call)[0]
                        torch.testing.assert_close(actual, native, rtol=1e-5, atol=1e-6)
                        self.assertFalse(graft.triggered.any())
                        self.assertEqual(graft.forward_calls, 1)

    def test_block_output_diagnostics_use_final_output_changes(self):
        from safetensors.torch import load_file
        graft, _, call, frames, _ = self.adapter(False, "block_output", trace=True)
        with tempfile.TemporaryDirectory() as directory:
            graft.trace_dirs = [Path(directory) / str(i) for i in range(graft.batch)]
            finish_mask = graft.finish_mask

            def full_mask():
                graft.votes = torch.ones(2, 1, 4, dtype=torch.int32) * len(graft.localizers)
                finish_mask()

            graft.active = True
            with patch.object(graft, "finish_mask", side_effect=full_mask), torch.inference_mode():
                graft.forward(**call)
            for i, destination in enumerate(graft.trace_dirs):
                self.assertTrue((destination / "step_0000.png").is_file())
                tensors = load_file(destination / "step_0000.safetensors")
                self.assertTrue(tensors["applied_mask"].all())
                anchor = frames["anchor", graft.last_layer][1][i]
                before = frames["user", graft.last_layer][1][-graft.batch:][i]
                expected_rms = (anchor - before).square().mean().sqrt()
                self.assertGreater(expected_rms.item(), 0)
                torch.testing.assert_close(tensors["activation_delta_rms"][0, 0], expected_rms)
                self.assertEqual(graft.step_traces[i][0]["representation"], "block_output")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
