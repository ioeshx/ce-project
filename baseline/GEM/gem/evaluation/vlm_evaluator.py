import gc
from abc import ABC, abstractmethod
from collections import Counter
from typing import List

import torch
from PIL import Image
from tqdm import tqdm

from gem.evaluation.base_evaluator import BaseEvaluator


# ---------------------------------------------------------------------------
# Logic Utilities
# ---------------------------------------------------------------------------

def _parse_yes_no(text: str) -> str:
    """Parses model output to return a clean YES or NO."""
    text_lower = text.strip().lower()
    if "answer:" in text_lower:
        text_lower = text_lower.split("answer:")[-1].strip()

    tokens = text_lower.split()
    token = tokens[0].rstrip(".,!?") if tokens else ""
    return "YES" if token == "yes" else "NO"


MAX_NEW_TOKENS_TO_GENERATE = 32


# ---------------------------------------------------------------------------
# Base VLM Model Wrapper
# ---------------------------------------------------------------------------

class BaseVLM(ABC):
    MODEL_ID = None

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self._model = None
        self._processor = None

    @abstractmethod
    def load(self):
        pass

    def unload(self):
        if self._model is not None:
            del self._model
            del self._processor
            self._model, self._processor = None, None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @abstractmethod
    def ask(self, image: Image.Image, question: str) -> str:
        pass


# ---------------------------------------------------------------------------
# Model Implementations
# ---------------------------------------------------------------------------

class BLIP2VQA(BaseVLM):
    MODEL_ID = "Salesforce/blip2-opt-2.7b"

    def load(self):
        if self._model is not None: return
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        print(f"\n[BLIP-2] Loading {self.MODEL_ID} ...")
        self._processor = Blip2Processor.from_pretrained(self.MODEL_ID)
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
        ).to(self.device)
        self._model.eval()

    def ask(self, image: Image.Image, question: str) -> str:
        prompt = f"Question: {question.strip()} Answer:"
        inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS_TO_GENERATE)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        answer = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return _parse_yes_no(answer)


class Qwen25VLVQA(BaseVLM):
    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

    def load(self):
        if self._model is not None: return
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        print(f"\n[Qwen2.5-VL] Loading {self.MODEL_ID} ...")
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            device_map="auto",
        )
        self._model.eval()

    def ask(self, image: Image.Image, question: str) -> str:
        from qwen_vl_utils import process_vision_info
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(text=[text], images=image_inputs, videos=video_inputs, padding=True,
                                 return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS_TO_GENERATE)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        answer = self._processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
            0].strip()
        return _parse_yes_no(answer)


class Qwen3VLVQA(Qwen25VLVQA):
    """Integrated Qwen3-VL Support"""
    MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

    def load(self):
        if self._model is not None: return
        from transformers import AutoModelForVision2Seq, AutoProcessor
        print(f"\n[Qwen3-VL] Loading {self.MODEL_ID} ...")
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            device_map="auto",
        )
        self._model.eval()


class LLaVAVQA(BaseVLM):
    MODEL_ID = "llava-hf/llava-1.5-7b-hf"

    def load(self):
        if self._model is not None: return
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        print(f"\n[LLaVA] Loading {self.MODEL_ID} ...")
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self._model = LlavaForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            device_map="auto",
        )
        self._model.eval()

    def ask(self, image: Image.Image, question: str) -> str:
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = self._processor(text=prompt, images=image, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS_TO_GENERATE)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        answer = self._processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
            0].strip()
        return _parse_yes_no(answer)


class Phi35VisionVQA(BaseVLM):
    """SOTA Small Vision Model (Microsoft)"""
    MODEL_ID = "microsoft/Phi-3.5-vision-instruct"

    def load(self):
        if self._model is not None: return
        from transformers import AutoModelForCausalLM, AutoProcessor
        import torch

        print(f"\n[Phi-3.5-Vision] Loading {self.MODEL_ID} ...")

        # 1. Loading the Processor natively
        # We set trust_remote_code=True once here just to fetch the config,
        # but the actual logic will run through the local library.
        self._processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True
        )

        # 2. Loading the Model natively
        # By providing 'phi3_v' as the model type (or letting it auto-detect),
        # we bypass the broken remote scripts.
        self._model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            device_map="auto",
            # This is the critical "Proper" flag
            trust_remote_code=True,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            _attn_implementation="eager"
        ).eval()

    def ask(self, image: Image.Image, question: str) -> str:
        # Phi-3.5 uses a specific prompt format with indexed image tags
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{question}"}
        ]

        prompt = self._processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(prompt, [image], return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            generate_ids = self._model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_TO_GENERATE,
                do_sample=False
            )

        # Remove input tokens from output
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        answer = self._processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return _parse_yes_no(answer)


class InternVL25VQA(BaseVLM):
    """SOTA InternVL2.5 Support"""
    MODEL_ID = "OpenGVLab/InternVL2_5-8B"

    def load(self):
        if self._model is not None: return
        from transformers import AutoModel, AutoTokenizer
        print(f"\n[InternVL2.5] Loading {self.MODEL_ID} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # avoids OOM during loading
        ).eval()
        if "cuda" in self.device:
            self._model = self._model.cuda()  # ← single clean move, no device_map

    def ask(self, image: Image.Image, question: str) -> str:
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        transform = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = image.convert("RGB")
        dtype = torch.float16 if "cuda" in self.device else torch.float32
        pixel_values = transform(image).unsqueeze(0).to(self._model.device, dtype=dtype)

        prompt = f"<image>\n{question}"
        generation_config = {"max_new_tokens": MAX_NEW_TOKENS_TO_GENERATE, "do_sample": False}

        response, _ = self._model.chat(
            self._tokenizer,
            pixel_values,
            prompt,
            generation_config=generation_config
        )
        return _parse_yes_no(response)


class PixtralVQA(BaseVLM):
    """Mistral's Frontier Vision Model - Native Implementation"""
    # Keep using the community repo as it has the correct HF config files
    MODEL_ID = "mistral-community/pixtral-12b"

    def load(self):
        if self._model is not None: return
        # FRESH START: Use the native Auto classes, NO LlavaNext imports
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch

        print(f"\n[Pixtral] Loading {self.MODEL_ID} natively ...")

        # No trust_remote_code needed anymore!
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)

        # Load directly into the Vision2Seq pipeline
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16
        ).eval()

    def ask(self, image: Image.Image, question: str) -> str:
        import torch
        image = image.convert("RGB")

        # The native processor handles this flawlessly now
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]

        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)

        # Process the inputs (the native processor won't mess up the split math)
        inputs = self._processor(text=prompt, images=[image], return_tensors="pt").to(self._model.device)

        # Ensure image tensors match model dtype
        inputs = {k: v.to(self._model.dtype) if torch.is_floating_point(v) else v for k, v in inputs.items()}

        with torch.no_grad():
            generate_ids = self._model.generate(
                **inputs,
                max_new_tokens=256,  # Or your MAX_NEW_TOKENS_TO_GENERATE
                do_sample=False
            )

        input_len = inputs["input_ids"].shape[1]
        answer = self._processor.batch_decode(
            generate_ids[:, input_len:],
            skip_special_tokens=True
        )[0].strip()

        return _parse_yes_no(answer)  # Add back your _parse_yes_no() if needed


class Llama32VisionVQA(BaseVLM):
    """Meta Llama 3.2 Vision 11B"""
    MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    def load(self):
        if self._model is not None: return
        from transformers import AutoProcessor, AutoModelForVision2Seq
        print(f"\n[Llama-3.2-Vision] Loading {self.MODEL_ID} ...")
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
        ).eval()

    def ask(self, image: Image.Image, question: str) -> str:
        image = image.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=prompt, images=[image], return_tensors="pt").to(self._model.device)
        inputs = {k: v.to(self._model.dtype) if torch.is_floating_point(v) else v for k, v in inputs.items()}

        with torch.no_grad():
            generate_ids = self._model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_TO_GENERATE,
                do_sample=False
            )

        answer = self._processor.batch_decode(
            generate_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )[0].strip()
        return _parse_yes_no(answer)


class Gemma3VQA(BaseVLM):
    """Google Gemma 3 12B Vision"""
    MODEL_ID = "google/gemma-3-12b-it"

    def load(self):
        if self._model is not None: return
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        print(f"\n[Gemma3] Loading {self.MODEL_ID} ...")
        # padding_side="left" is required for Gemma 3 batch inference
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID, padding_side="left")
        self._model = Gemma3ForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32,
        ).eval()

    def ask(self, image: Image.Image, question: str) -> str:
        image = image.convert("RGB")
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant. Answer concisely."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},  # PIL image passed directly here
                    {"type": "text", "text": question},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device, dtype=torch.bfloat16 if "cuda" in self.device else torch.float32)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generate_ids = self._model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_TO_GENERATE,
                do_sample=False,
            )

        answer = self._processor.decode(
            generate_ids[0][input_len:], skip_special_tokens=True
        ).strip()
        return _parse_yes_no(answer)


class LlavaOneVisionVQA(BaseVLM):
    MODEL_ID = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

    def load(self):
        if self._model is not None: return
        from transformers import AutoProcessor, AutoModelForVision2Seq
        print(f"\n[LLaVA-OneVision] Loading {self.MODEL_ID} ...")
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
        ).eval()

    def ask(self, image: Image.Image, question: str) -> str:
        image = image.convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=prompt, images=[image], return_tensors="pt").to(self._model.device)
        inputs = {k: v.to(self._model.dtype) if torch.is_floating_point(v) else v for k, v in inputs.items()}
        with torch.no_grad():
            generate_ids = self._model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS_TO_GENERATE, do_sample=False)
        answer = self._processor.batch_decode(generate_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[
            0].strip()
        return _parse_yes_no(answer)


# ---------------------------------------------------------------------------
# VLM Classifier (Single Model)
# ---------------------------------------------------------------------------


class VLMClassifier:
    DEFAULT_PROMPT = "Analyze the provided image and answer the binary question below.\n{label_instruction}\n\nQUESTION: {question}\n\nANSWER: "
    BINARY_LABEL_INSTRUCTION = "Answer only YES or NO. If you are unsure, answer NO."

    def __init__(self, question: str, model_name: str, device: str = None):
        self.question = self.DEFAULT_PROMPT.format(
            question=question,
            label_instruction=self.BINARY_LABEL_INSTRUCTION
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Registry of supported models
        model_map = {
            "blip2": BLIP2VQA,
            "qwen2.5": Qwen25VLVQA,
            "qwen3": Qwen3VLVQA,
            "llava": LLaVAVQA,
            "internvl2.5": InternVL25VQA,
            "pixtral": PixtralVQA,
            "llama3.2": Llama32VisionVQA,
            "llama-onevision": LlavaOneVisionVQA,
            "gemma3": Gemma3VQA
            # "phi3.5": Phi35VisionVQA  # Does not work
        }

        self.model_name = model_name.lower()
        if self.model_name not in model_map:
            raise NotImplementedError(f"Model '{model_name}' is not supported. Choose from {list(model_map.keys())}")

        self.model_class = model_map[self.model_name]
        self.vlm = self.model_class(device=self.device)
        self.vlm.load()

    def __call__(self, image_path: str):
        try:
            img = Image.open(image_path).convert("RGB")
            answer = self.vlm.ask(img, self.question)
            return {"label": answer, "success": answer == "YES", "error": None}
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return {"label": "NO", "success": False, "error": str(e)}

    def close(self):
        self.vlm.unload()


# ---------------------------------------------------------------------------
# VLM Evaluator
# ---------------------------------------------------------------------------

class VLMEvaluator(BaseEvaluator):

    def __init__(self, question: str, model_name: str, **kwargs):

        super(VLMEvaluator, self).__init__()

        self.classifier = VLMClassifier(
            question=question,
            model_name=model_name,
            **kwargs
        )

    def process_images(self, image_file_paths: List[str], **kwargs) -> dict:
        summary = Counter()

        print(f"Scanning {len(image_file_paths)} images with {self.classifier.model_class.__name__}...")

        for image_path in tqdm(image_file_paths, desc="VLM Processing"):
            result = self.classifier(image_path)

            if result["error"]:
                summary['errors'] += 1
            else:
                summary['no_errors'] += 1

                if result['success']:
                    summary['yes_count'] += 1

        total_processed = summary['no_errors']
        yes_rate = (summary['yes_count'] / total_processed) if total_processed > 0 else 0.0

        return {
            'total': len(image_file_paths),
            'processed': total_processed,
            'errors': summary['errors'],
            'detection_rate': yes_rate,
        }


class EnsembleVLMEvaluator(BaseEvaluator):
    """
    Evaluates multiple VLMs sequentially.
    Calculates individual detection rates and an ensemble mean.
    """

    def __init__(self, question: str, model_names: List[str], device: str = None):
        super(EnsembleVLMEvaluator, self).__init__()
        self.question = question
        self.model_names = [m.lower() for m in model_names]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def process_images(self, image_file_paths: List[str], **kwargs) -> dict:
        individual_results = {}
        detection_rates = []

        print(f"Starting Ensemble Evaluation with models: {self.model_names}")

        for model_name in self.model_names:
            print(f"\n--- Initializing Model: {model_name} ---")

            # Initialize and load the specific model
            classifier = VLMClassifier(
                question=self.question,
                model_name=model_name,
                device=self.device
            )

            summary = Counter()

            # Process all images with this specific model
            for image_path in tqdm(image_file_paths, desc=f"Processing {model_name}"):
                result = classifier(image_path)

                if result["error"]:
                    summary['errors'] += 1
                else:
                    summary['processed'] += 1
                    if result['success']:
                        summary['yes_count'] += 1

            # Calculate stats for this model
            total_valid = summary['processed']
            rate = (summary['yes_count'] / total_valid) if total_valid > 0 else 0.0

            # Store results and cleanup memory
            individual_results[f"{model_name}_detection_rate"] = rate
            detection_rates.append(rate)

            print(f"Result for {model_name}: {rate:.2%}")
            classifier.close()
            # Force extra cleanup just in case
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Calculate Ensemble Mean
        ensemble_mean = sum(detection_rates) / len(detection_rates) if detection_rates else 0.0

        return {
            'total': len(image_file_paths),
            'errors': -1,
            **individual_results,
            "ensemble_mean": ensemble_mean
        }


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Options: "blip2", "qwen2.5", "qwen3", "llava"
    MODEL_TO_USE = "qwen3"

    evaluator = VLMEvaluator(
        question="Is there a person in this image?",
        model_name=MODEL_TO_USE
    )

    # Replace with your actual image paths
    image_paths = ["test_image_1.jpg", "test_image_2.jpg"]

    try:
        results = evaluator.process_images(image_paths)
        print("\n--- Evaluation Results ---")
        print(results)
    finally:
        evaluator.classifier.close()
