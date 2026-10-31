import copy

from gem.configs import OperatorConfig
from gem.operators.identity_operator import IdentityOperatorConfig


class OperatorChain:
    def __init__(self, exp_name, initial_pipe_config):
        self.exp_name = exp_name
        self.initial_pipe_config = initial_pipe_config
        self.operator_configs = []

    def __iter__(self):
        return iter(self.operator_configs)

    def __getitem__(self, item):
        return self.operator_configs[item]

    def __len__(self):
        return len(self.operator_configs)

    def append_operator(self, operator_config):
        chain_copy = copy.deepcopy(self)
        if operator_config is None:
            return chain_copy

        chain_copy.operator_configs.append(operator_config)
        return chain_copy

    def __str__(self):
        base_name = self.exp_name if self.operator_configs else "original"

        id_base = '/'.join(
            [base_name, f"{self.initial_pipe_config.get_pipe_config_string()}"]
        )

        id_parts = []
        for op_config in self.operator_configs:
            op_id = op_config.identifier()
            if op_id:
                id_parts.append(f"[{op_id}]")

        return id_base + "/" + "->".join(id_parts)

    def get_empty_base_id_chain(self):
        base = copy.deepcopy(self)
        base.operator_configs.clear()
        return base

