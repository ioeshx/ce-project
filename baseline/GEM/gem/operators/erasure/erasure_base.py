from dataclasses import dataclass, field
from typing import List, Optional

from gem.common.prompt_augmentation import PromptAugmentationConfig
from gem.configs import OperatorConfig
from gem.operators.model_adaptation_mixin import ModelAdaptationMixin
from gem.operators.operator import Operator
from gem.wrappers import BasePipelineWrapper


@dataclass
class ErasureBaseConfig(OperatorConfig, ModelAdaptationMixin):
    method: str = "erasure_base"

    targets: List[str] = field(default_factory=list)
    prompt_augmentation: Optional[PromptAugmentationConfig] = None

    validation_templates: List[str] = field(
        default_factory=lambda: ["{}", "an image of {}", "an artwork of {}"]
    )
    validation_concepts: List[str] = field(
        default_factory=lambda: ['a gem', 'a cat', 'a tree']
    )

    def __post_init__(self):
        OperatorConfig.__post_init__(self)
        ModelAdaptationMixin.__post_init__(self)

        assert isinstance(self.targets, list), f"Targets must always be a list of strings but got: {self.targets}"

        #  Once anchors (and targets) are specified, ensure they match in length
        if hasattr(self, 'anchors') and self.targets is not None and len(self.targets):
            if len(self.anchors) == 1:
                self.anchors = self.anchors * len(self.targets)

            # Throw error if lengths do not match for some reason
            if len(self.anchors) != len(self.targets):
                raise ValueError(
                    f"Number of anchors ({self.anchors}) must match number of targets ({self.targets})."
                )

    def identifier(self):
        parts = [self.method]
        if self.id:
            parts.append(self.id)
        if self.adaptation_config is not None:
            parts.extend([self.adaptation_config.adapter_mode, self.adaptation_config.subset_name])
        parts = map(str, parts)
        return '_'.join(parts)


class ErasureOperator(Operator):

    def __init__(self, name):
        super(ErasureOperator, self).__init__(name=name)

    def __call__(self, wrapper: BasePipelineWrapper, config: ErasureBaseConfig):
        super(ErasureOperator, self).__call__(wrapper, config)
