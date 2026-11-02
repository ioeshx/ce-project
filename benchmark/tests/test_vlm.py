"""Constrained VLM decoding regressions; all fixtures stay in temporary paths."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from benchmark.vlm import LABELS, PROMPT_VERSION, VLMClassifier, label_constraint, label_responses, parse_label


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class VLMTests(unittest.TestCase):
    def test_constraints_allow_every_label_and_reject_minnie_and_malformed_json(self):
        import torch
        tokenizer = CharacterTokenizer()
        for task in LABELS:
            allowed = label_constraint(tokenizer, task, 2, [0], 32)
            for response in label_responses(task):
                prefix = [1, 2]
                for token in tokenizer.encode(response) + [0]:
                    self.assertIn(token, allowed(0, torch.tensor(prefix)))
                    prefix.append(token)
                self.assertEqual(allowed(0, torch.tensor(prefix)), [0])
                self.assertEqual(allowed(0, torch.tensor(prefix + [1, 1])), [0])
                self.assertIn(parse_label(response, task), LABELS[task])
        allowed = label_constraint(tokenizer, "instance", 2, [0], 32)
        prefix = torch.tensor([1, 2] + tokenizer.encode('{"label":"Mi'))
        self.assertIn(ord("c"), allowed(0, prefix))
        self.assertNotIn(ord("n"), allowed(0, prefix))
        with self.assertRaises(ValueError):
            parse_label('{"label":"Minnie"}', "instance")
        with self.assertRaises(ValueError):
            parse_label('{"label":["Mickey","NONE"]}', "instance")
        with self.assertRaisesRegex(ValueError, "at least"):
            label_constraint(tokenizer, "style", 2, [0], 2)

    def test_predict_uses_real_transformers_constraints_for_padded_batches(self):
        import torch
        from transformers import BatchEncoding, GPT2Config, GPT2LMHeadModel, LogitsProcessor
        torch.set_num_threads(2)
        tokenizer = CharacterTokenizer()
        # A tiny text decoder exercises the same Transformers generate API as
        # Qwen, without loading/downloading production VLM weights.
        model = GPT2LMHeadModel(GPT2Config(vocab_size=128, n_positions=64, n_embd=16,
            n_layer=1, n_head=2, bos_token_id=1, eos_token_id=0, pad_token_id=1)).eval()
        forbidden = tokenizer.encode('{"label":"Minnie"}') + [0]
        shorter = tokenizer.encode('{"label":"NONE"}') + [0]

        class PreferInvalidName(LogitsProcessor):
            def __call__(self, input_ids, scores):
                for row, preferred in enumerate((forbidden, shorter)):
                    scores[row, preferred[min(input_ids.shape[1] - 3, len(preferred) - 1)]] += 1000
                return scores

        class Processor:
            supports_processor_kwargs = True
            def __init__(self):
                self.tokenizer = tokenizer
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return BatchEncoding({"input_ids": torch.tensor([[0, 1, 2], [1, 2, 3]]),
                                      "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]])})
            def batch_decode(self, generated, **kwargs):
                return ["".join(chr(int(t)) for t in row if int(t) not in (0, 1)) for row in generated]

        class Backend:
            def generate(self, **kwargs):
                self.arguments = kwargs
                return model.generate(**kwargs, logits_processor=[PreferInvalidName()])

        with tempfile.TemporaryDirectory(prefix="vlm-constraint-check-") as temp:
            paths = [Path(temp) / f"image-{i}.png" for i in range(2)]
            for path in paths:
                Image.new("RGB", (16, 16)).save(path)
            for task in LABELS:
                classifier = object.__new__(VLMClassifier)
                classifier.processor, classifier.model = Processor(), Backend()
                classifier.task, classifier.device = task, "cpu"
                classifier.max_new_tokens, classifier.eos_token_ids = 32, [0]
                classifier.supports_processor_kwargs = True
                predictions = classifier.predict(paths)
                self.assertEqual(len(predictions), 2)
                for prediction in predictions:
                    self.assertIn(prediction["label"], LABELS[task])
                    self.assertEqual(json.loads(prediction["raw_response"]), {"label": prediction["label"]})
                self.assertIn("prefix_allowed_tokens_fn", classifier.model.arguments)
                self.assertFalse(classifier.model.arguments["do_sample"])
                self.assertEqual(classifier.model.arguments["num_beams"], 1)

    def test_new_protocol_preserves_and_does_not_reuse_unconstrained_cache(self):
        from benchmark.classification import predict_field
        from benchmark.common import read_json
        class Backend:
            provenance = {"test_fixture": True}
            calls = 0
            def predict(self, paths):
                self.calls += 1
                return [{"raw_response": '{"label":"NONE"}', "label": "NONE"} for _ in paths]
        with tempfile.TemporaryDirectory(prefix="vlm-cache-protocol-") as temp:
            root = Path(temp)
            image = root / "image.png"
            Image.new("RGB", (16, 16)).save(image)
            resource = root / "vlm-fixture"
            resource.mkdir()
            (resource / "config.json").write_text('{}')
            opts = {"vlm_model": str(resource), "device": "cpu", "vlm_batch_size": 1}
            manifest = {"task": "instance", "model_name": "sd35_medium", "records": [{"image": str(image)}]}
            backend = Backend()
            with patch("benchmark.vlm.PROMPT_VERSION", "dit-single-label-v1"):
                predict_field(manifest, opts, "vlm", "image", root / "cache", lambda: backend)
            old = next((root / "cache").rglob("predictions.json"))
            previous = old.read_bytes()
            predict_field(manifest, opts, "vlm", "image", root / "cache", lambda: backend)
            self.assertEqual(backend.calls, 2)
            self.assertEqual(old.read_bytes(), previous)
            plans = list((root / "cache").rglob("run.json"))
            versions = {read_json(p)["identity"]["settings"]["instruction_version"] for p in plans}
            self.assertEqual(versions, {"dit-single-label-v1", PROMPT_VERSION})
            predict_field(manifest, opts, "vlm", "image", root / "cache",
                          lambda: self.fail("New constrained cache must be reused"))


if __name__ == "__main__":
    unittest.main()
