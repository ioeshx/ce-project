from typing import Optional, List

from gem.configs.base_config import BaseConfig

from dataclasses import dataclass, field
from abc import ABC


@dataclass
class OperatorConfig(BaseConfig, ABC):
    id: str = None
    method: str = None

    run_save_path: str = None
    train_seed: int = 0
    val_seed: int = 0

    use_wandb: bool = False
    initial_validation: bool = True
    steps_between_validation: int = 100

    save_pipeline_after_operator: bool = True
    steps_to_save_checkpoints: Optional[List[int]] = field(default_factory=lambda: [])

    max_gradient_norm: float = None

    def __post_init__(self):
        pass

    def identifier(self):
        """Return the method identifier for the operator."""
        return self.method


@dataclass
class AdhocOperatorConfig(OperatorConfig):
    """Configuration for adhoc operators that do not have a specific method"""
