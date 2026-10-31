from dataclasses import dataclass

from gem.configs.base_config import BaseConfig
from gem.configs.pipeline_config import PipelineConfig


@dataclass
class InferenceConfig(BaseConfig):
    pipe_config: PipelineConfig = None
    dataset_name: str = None
    batch_size: int = 8
    test_seed: int = 0
