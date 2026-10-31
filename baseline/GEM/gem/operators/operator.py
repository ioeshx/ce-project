from typing import Optional

import wandb
from gem.configs import OperatorConfig
from gem.utils.logging import get_logger
from gem.utils.memory import get_gpu_memory_percent
from gem.utils.seed import seed_everything
from gem.wrappers import BasePipelineWrapper


class Operator:
    def __init__(self, name):
        self.name = name

    @property
    def logger(self):
        return get_logger()

    def __call__(self, wrapper: BasePipelineWrapper, config: OperatorConfig) -> Optional[BasePipelineWrapper]:
        seed_everything(config.train_seed)
        return wrapper

    def log_loss_and_metrics_to_wandb(self, loss, step, metrics=None, step_metric='global_step', loss_name=None, use_wandb=False):
        if metrics is None:
            metrics = {}

        if loss_name is None:
            loss_name = self.name.upper()

        if use_wandb:
            wandb.define_metric(step_metric)
            wandb.define_metric(f"{self.name.upper()} Loss", step_metric=step_metric)

            wandb.log({
                f"{loss_name} Loss": loss, step_metric: step, **metrics,
                "GPU Memory %": get_gpu_memory_percent(),
            })
        else:
            if loss is not None:
                print(f"{loss_name} Loss [{step_metric.title().replace('_', ' ')}: {step}]", loss)


def translate_shared_fields_to_other_config(config, target_cls):

    shared_params = {}
    for attr in dir(config):
        # Skip private/protected attributes
        if attr.startswith("_"):
            continue

        if callable(getattr(config, attr)):
            # Do not transfer methods
            continue

        # Only include if the target config class actually has this parameter
        if hasattr(target_cls, attr) and getattr(config, attr) is not None:
            shared_params[attr] = getattr(config, attr)

    print("Shared non-null parameters to transfer:")
    for key, value in shared_params.items():
        print("> ", key, "=", value)

    new_config = target_cls()
    # Update with shared params
    for key, value in shared_params.items():
        new_config.__setattr__(key, value)

    return new_config
