from dataclasses import dataclass
from typing import Optional

from gem.configs import AdaptationConfig


@dataclass
class ModelAdaptationMixin:
    adaptation_config: AdaptationConfig = None

    @staticmethod
    def get_default_adaptation_config() -> Optional[AdaptationConfig]:
        return None

    def __post_init__(self):
        self.adaptation_config = self.adaptation_config or self.get_default_adaptation_config()