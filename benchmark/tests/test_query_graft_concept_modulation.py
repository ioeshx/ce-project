"""Compare concept modulation/residuals with independent native block forwards."""
from functools import partial
import unittest
from unittest.mock import patch

import torch

from benchmark.tests import test_query_graft as fixtures


class ConceptModulationTests(unittest.TestCase):
    adapter = fixtures.QueryGraftTests.adapter

    def test_zero_pooled_condition_and_native_concept_residuals(self):
        for flux, guided in ((False, False), (True, False), (True, True)):
            with self.subTest(flux=flux, guided=guided), torch.inference_mode():
                graft, blocks, call, frames, _ = self.adapter(
                    flux, "av", eta=1, guidance_embeds=guided)
                model = graft.pipe.transformer
                reference = type(model).from_config(model.config)
                reference.load_state_dict(model.state_dict())
                reference.eval()
                references = list(reference.transformer_blocks)
                if flux:
                    references += list(reference.single_transformer_blocks)
                call["timestep"] = torch.linspace(.2, .8, call["timestep"].shape[0])
                if guided:
                    call["guidance"] = torch.tensor([2.5, 4.0])
                native_output = model(**call)[0]
                inputs, normalized = {}, {}

                def capture_input(layer, module, args, kwargs):
                    inputs[layer] = dict(kwargs)

                def capture_normalized(layer, module, args, kwargs):
                    value = kwargs.get("encoder_hidden_states", kwargs["hidden_states"])
                    normalized[layer] = value.detach().clone()

                for layer, block in enumerate(blocks):
                    self.addCleanup(block.register_forward_pre_hook(
                        partial(capture_input, layer), with_kwargs=True).remove)
                    self.addCleanup(block.attn.register_forward_pre_hook(
                        partial(capture_normalized, layer), with_kwargs=True).remove)
                embed = model.time_text_embed.forward
                graft.active = True
                with patch.object(model.time_text_embed, "forward", wraps=embed) as embedding:
                    actual_output = graft.forward(**call)[0]
                torch.testing.assert_close(actual_output, native_output, atol=1e-6, rtol=1e-5)
                self.assertEqual(embedding.call_count, 2)
                user_args, concept_args = (entry.args for entry in embedding.call_args_list)
                self.assertTrue(torch.equal(user_args[-1], call["pooled_projections"]))
                self.assertTrue(torch.count_nonzero(concept_args[-1]) == 0)
                for actual, expected in zip(concept_args[:-1], user_args[:-1]):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.testing.assert_close(user_args[0], call["timestep"] * (1000 if flux else 1))
                if guided:
                    torch.testing.assert_close(user_args[1], call["guidance"] * 1000)
                concept_temb = reference.time_text_embed(*concept_args)
                torch.testing.assert_close(graft.concept_temb, concept_temb, atol=0, rtol=0)
                self.assertFalse(torch.allclose(embed(*user_args), concept_temb))

                for layer, block in enumerate(references):
                    single = flux and layer >= len(reference.transformer_blocks)
                    original = inputs[layer]
                    concepts = original["encoder_hidden_states"][:, graft.text_count:]
                    image = original["hidden_states"]
                    user_temb = original["temb"]
                    normalizer = block.norm if single else block.norm1_context
                    if not single and block.attn.context_pre_only:
                        concept_norm = normalizer(concepts, concept_temb)
                    else:
                        concept_norm = normalizer(concepts, emb=concept_temb)[0]
                    actual_norm = normalized[layer][:, graft.text_count:graft.text_count + graft.concept_count]
                    torch.testing.assert_close(actual_norm, concept_norm, atol=1e-6, rtol=1e-5)

                    # A native concept+image forward is an independent reference:
                    # image normalization retains the user condition, while all
                    # concept gates, modulation and residuals use concept_temb.
                    def image_condition(module, args, output):
                        user_output = module.forward(args[0], emb=user_temb)
                        if not single:
                            return user_output
                        mixed = torch.cat((output[0][:, :graft.concept_count],
                                           user_output[0][:, graft.concept_count:]), dim=1)
                        return (mixed, *output[1:])

                    image_normalizer = block.norm if single else block.norm1
                    handle = image_normalizer.register_forward_hook(image_condition)
                    try:
                        options = dict(hidden_states=image, encoder_hidden_states=concepts, temb=concept_temb)
                        if flux:
                            options["image_rotary_emb"] = tuple(
                                component[graft.text_count:] for component in original["image_rotary_emb"])
                        expected_context, _ = block(**options)
                    finally:
                        handle.remove()
                    actual_context = frames["user", layer][0]
                    if expected_context is None:
                        self.assertIsNone(actual_context)
                    else:
                        torch.testing.assert_close(actual_context[:, graft.text_count:], expected_context,
                                                   atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
