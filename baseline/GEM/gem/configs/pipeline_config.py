from dataclasses import dataclass, field
from typing import Optional

from gem.configs.base_config import BaseConfig

MODEL_CONFIGS = {
    "sd_1_4": {
        "model_name_or_path": "CompVis/stable-diffusion-v1-4",
        "num_inference_steps": 50,
        "inference_guidance_scale": 7.5,
        "subset_name": "uncond",
        "scheduler": "ddim"
    },
    "sd_1_5": {
        "model_name_or_path": "runwayml/stable-diffusion-v1-5",
        "num_inference_steps": 50,
        "inference_guidance_scale": 7.5,
        "subset_name": "uncond",
        "scheduler": "ddim"
    },
    "sd_2_1": {
        "model_name_or_path": "stabilityai/stable-diffusion-2-1-base",
        "num_inference_steps": 50,
        "inference_guidance_scale": 7.5,
        "subset_name": "uncond",
        "scheduler": "ddim"
    },
    "sd_3": {
        "model_name_or_path": "stabilityai/stable-diffusion-3-medium-diffusers",
        "num_inference_steps": 28,
        "inference_guidance_scale": 7.0,
        "subset_name": "qk_dual",
    },
    "sd_3_5": {
        "model_name_or_path": "stabilityai/stable-diffusion-3.5-medium",
        "num_inference_steps": 40,
        "inference_guidance_scale": 4.5,
        "subset_name": "qk_dual",
    },
    "flux": {
        "model_name_or_path": "black-forest-labs/FLUX.1-dev",
        "num_inference_steps": 28,
        "inference_guidance_scale": 3.5,  # was 7.0 in the paper
        "subset_name": "qk_dual",
        "model_dtype": "float16"
    },
    "qwen_image": {
        "model_name_or_path": "Qwen/Qwen-Image",
        "num_inference_steps": 50,
        "inference_guidance_scale": 4.0,
        "image_size": 512,
        "subset_name": "qk_dual",
        "model_dtype": "bfloat16"
    }
}


@dataclass
class PipelineConfig(BaseConfig):
    _allow_init: bool = field(default=False, repr=False, compare=False)

    model_type: str = None
    model_name_or_path: Optional[str] = None
    model_dtype: str = None

    scheduler: str = 'default'
    image_size: int = 512
    num_inference_steps: Optional[int] = None
    inference_guidance_scale: Optional[float] = None

    def get_pipe_config_string(self):
        return f"{self.model_type}_{self.model_dtype}_{self.scheduler}"

    def __post_init__(self):
        if not self._allow_init:
            raise RuntimeError("Use PipelineConfig.default_for_model_type(model_type) to instantiate this class.")

        # Apply defaults only if not provided manually
        if self.model_name_or_path is None:
            self.model_name_or_path = MODEL_CONFIGS[self.model_type]["model_name_or_path"]

        if self.num_inference_steps is None:
            self.num_inference_steps = MODEL_CONFIGS[self.model_type]["num_inference_steps"]

        if self.inference_guidance_scale is None:
            self.inference_guidance_scale = MODEL_CONFIGS[self.model_type]["inference_guidance_scale"]

        if self.model_dtype is None:
            self.model_dtype = MODEL_CONFIGS[self.model_type].get("model_dtype", "float32")

        if self.scheduler is None or self.scheduler == 'default':
            self.scheduler = MODEL_CONFIGS[self.model_type].get("scheduler", self.scheduler)

        print(self)

    @classmethod
    def default_for_model_type(cls, model_type: str):
        if model_type not in MODEL_CONFIGS:
            raise KeyError(f"Unknown model_type {model_type}")

        model_default_config = MODEL_CONFIGS[model_type]
        return cls(
            _allow_init=True,
            model_type=model_type,
            model_name_or_path=model_default_config["model_name_or_path"],
            num_inference_steps=model_default_config["num_inference_steps"],
            inference_guidance_scale=model_default_config["inference_guidance_scale"]
        )

    def is_intermediate_checkpoint(self):
        return 'checkpoints' in self.model_name_or_path.split("/")
