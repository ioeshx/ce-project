from dataclasses import dataclass

from gem.configs import OperatorConfig
from gem.operators.operator import Operator


@dataclass
class IdentityOperatorConfig(OperatorConfig):
    method: str = 'identity'
    use_wandb: bool = False
    save_pipeline_after_operator: bool = False


class IdentityOperator(Operator):

    def __init__(self):
        super(IdentityOperator, self).__init__(name='identity')

