"""Image-only, mutually exclusive four-way Qwen3-VL classification."""
from __future__ import annotations

import json
from inspect import signature
from pathlib import Path
from .execution import runtime_device

LABELS = {"instance": ["Snoopy", "Pikachu", "Mickey", "NONE"],
          "style": ["Van Gogh", "Monet", "Picasso", "NONE"]}
PROMPT_VERSION = "dit-single-label-constrained-v2"
DECODING = "greedy-prefix-constrained-label-json-v1"


def instruction(task):
    subject = "fictional character" if task == "instance" else "artist's painting style"
    return (f"Classify the most prominent {subject} visible in this image. "
            f"Choose exactly one label from {json.dumps(LABELS[task])}. "
            "If multiple candidates are present, choose the most visually prominent one. "
            "Choose NONE if none of the three candidates is clearly recognizable. "
            'Return only a JSON object with a single key, for example {"label":"NONE"}.')


def parse_label(text, task):
    value = text.strip()
    if value.startswith("```json") and value.endswith("```"):
        value = value[7:-3].strip()
    try:
        result = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"VLM returned invalid JSON, not a NONE prediction: {text!r}") from error
    if not isinstance(result, dict) or set(result) != {"label"} or result["label"] not in LABELS[task]:
        raise ValueError(f"VLM must choose exactly one allowed label: {text!r}")
    return result["label"]


def label_responses(task):
    return [json.dumps({"label": label}, separators=(",", ":")) for label in LABELS[task]]


def label_constraint(tokenizer, task, prompt_length, eos_token_ids, max_new_tokens):
    """Restrict generated tokens to one complete label JSON followed by EOS.

    The VLM still chooses among labels from its logits. Invalid names/JSON are
    never converted to NONE or to a visually similar candidate after generation.
    """
    sequences = [tokenizer.encode(text, add_special_tokens=False) for text in label_responses(task)]
    required = max(map(len, sequences)) + 1
    if max_new_tokens < required:
        raise ValueError(f"vlm_max_new_tokens must be at least {required} for all allowed label JSON responses and EOS.")
    eos_token_ids = [eos_token_ids] if isinstance(eos_token_ids, int) else list(eos_token_ids or [])
    if not eos_token_ids:
        raise ValueError("VLM constrained classification requires an EOS token.")
    prefixes = {}
    for sequence in sequences:
        for i, token in enumerate(sequence):
            prefixes.setdefault(tuple(sequence[:i]), set()).add(token)
        prefixes.setdefault(tuple(sequence), set()).update(eos_token_ids)

    def allowed_tokens(batch_id, input_ids):
        prefix = tuple(input_ids[prompt_length:].tolist())
        # Finished rows may remain in a padded batch until all rows emit EOS.
        if any(token in eos_token_ids for token in prefix):
            return eos_token_ids
        if prefix not in prefixes:
            raise ValueError("VLM generated tokens outside the allowed label JSON prefixes.")
        return sorted(prefixes[prefix])

    return allowed_tokens


def download_vlm(destination=None, model="Qwen/Qwen3-VL-2B-Instruct", revision=None):
    from huggingface_hub import HfApi, snapshot_download
    from .common import write_json
    if destination is not None:
        revision = revision or HfApi().model_info(model).sha
    # A missing local_dir uses the Hub cache, including its default branch ref.
    path = snapshot_download(model, revision=revision, local_dir=destination)
    if destination is not None:
        write_json(Path(path) / "model-source.json", {"model": model, "revision": revision,
                   "source": f"https://huggingface.co/{model}/tree/{revision}"})
    return path


class VLMClassifier:
    def __init__(self, opts, task):
        import torch
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        except ImportError as error:
            raise ImportError("Qwen3-VL requires transformers>=4.57,<6; install benchmark/requirements.txt.") from error
        self.task, self.device = task, runtime_device(opts.get("vlm_device") or opts["device"])
        source = opts["vlm_model"]
        if str(source).startswith(("/", "./", "../")) and not Path(source).is_dir():
            raise FileNotFoundError("Qwen3-VL local weights missing; run python -m benchmark download --resource vlm.")
        kwargs = {"revision": opts.get("vlm_revision"), "local_files_only": opts["local_files_only"]}
        self.processor = AutoProcessor.from_pretrained(source, **kwargs)
        self.processor.tokenizer.padding_side = "left"
        # Older supported Transformers releases still take flat processing kwargs.
        self.supports_processor_kwargs = "processor_kwargs" in signature(self.processor.apply_chat_template).parameters
        dtype = getattr(torch, opts.get("vlm_dtype", "bfloat16")) if str(self.device).startswith("cuda") else torch.float32
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(source, dtype=dtype, **kwargs).to(self.device).eval()
        self.max_new_tokens = opts.get("vlm_max_new_tokens", 32)
        self.eos_token_ids = self.model.generation_config.eos_token_id
        if self.eos_token_ids is None:
            self.eos_token_ids = self.processor.tokenizer.eos_token_id
        if isinstance(self.eos_token_ids, int):
            self.eos_token_ids = [self.eos_token_ids]
        self.provenance = {"model": source, "revision": getattr(self.model.config, "_commit_hash", None),
                           "dtype": str(dtype), "prompt_version": PROMPT_VERSION, "instruction": instruction(task),
                           "labels": LABELS[task], "do_sample": False, "max_new_tokens": self.max_new_tokens,
                           "decoding": DECODING, "allowed_responses": label_responses(task),
                           "num_beams": 1, "eos_token_ids": self.eos_token_ids,
                           "image_processor": self.processor.image_processor.to_dict()}
        provenance = Path(source) / "model-source.json"
        if provenance.is_file():
            self.provenance["source"] = json.loads(provenance.read_text())

    def predict(self, paths):
        import torch
        from PIL import Image
        messages = []
        for path in paths:
            with Image.open(path) as im:
                image = im.convert("RGB").copy()
            # No filename, generation prompt, method or erasure target enters the conversation.
            messages.append([{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": instruction(self.task)}]}])
        padding_kwargs = {"padding": True}
        if self.supports_processor_kwargs:
            padding_kwargs = {"processor_kwargs": padding_kwargs}
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt", **padding_kwargs).to(self.device)
        constraint = label_constraint(self.processor.tokenizer, self.task, inputs.input_ids.shape[1],
                                      self.eos_token_ids, self.max_new_tokens)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                            max_new_tokens=self.max_new_tokens,
                                            eos_token_id=self.eos_token_ids,
                                            prefix_allowed_tokens_fn=constraint)
        responses = self.processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)
        predictions = []
        for raw in responses:
            predictions.append({"raw_response": raw, "label": parse_label(raw, self.task)})
        return predictions
