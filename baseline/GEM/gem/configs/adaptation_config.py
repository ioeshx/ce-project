from dataclasses import dataclass
from typing import Optional, Dict, Any

from gem.configs.base_config import BaseConfig
from gem.configs.pipeline_config import MODEL_CONFIGS


@dataclass
class AdaptationConfig(BaseConfig):
    subset_name: Optional[str] = None
    adapter_mode: Optional[str] = None

    lora_config: Optional[Dict[str, Any]] = None  # Declare it normally

    def __post_init__(self):
        if self.adapter_mode == "lora" and self.lora_config is None:
            # Default LoRA configuration
            self.lora_config = {
                "r": 16,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "bias": "none"
            }

    def default_for_model_type(self, model_type):
        model_default_config = MODEL_CONFIGS[model_type]
        self.subset_name = model_default_config["subset_name"] if self.subset_name is None else self.subset_name
        return self

