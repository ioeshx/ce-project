from typing import Union, Tuple

from gem.configs import OperatorConfig
from gem.configs.pipeline_config import MODEL_CONFIGS
from gem.operators import Operator
from gem.operators.erasure import GEMConfig, GEM, GEMSD3Config

METHOD_REGISTRY = {
    # GEM
    "gem": (GEMConfig, GEM),
    "gem:sd3": (GEMSD3Config, GEM),
}

SUPPORTED_METHODS = list(METHOD_REGISTRY.keys())
SUPPORTED_MODEL_TYPES = list(MODEL_CONFIGS.keys())


def get_config_and_operator(method: str) -> Union[Tuple[OperatorConfig, Operator]]:
    """Factory function to return the appropriate config and its function based on the method."""
    method = method.lower()

    try:
        config_class, operator_class = METHOD_REGISTRY[method]
        print(f"Found config + operator for the provided method ({method}): {config_class.__name__} and {operator_class.__name__}")
        return config_class(), operator_class()
    except KeyError:
        raise ValueError(f"Unknown method: {method}. Please choose from: {', '.join(SUPPORTED_METHODS)}.")


def get_config(method: str):
    method = method.lower()
    return METHOD_REGISTRY[method][0]()


def get_operator(method: str):
    method = method.lower()
    return METHOD_REGISTRY[method][1]()
