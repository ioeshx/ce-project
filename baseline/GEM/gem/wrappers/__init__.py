import copy
import shutil
from dataclasses import replace

from .base_pipeline_wrapper import BasePipelineWrapper
from .pipeline_wrapper_sd import StableDiffusionPipelineWrapper
from .pipeline_wrapper_sd3 import StableDiffusion3PipelineWrapper
from .pipeline_wrapper_flux import FluxPipelineWrapper

from .factory import PipelineFactory

from .custom_adapter_utils import add_custom_adapters, assert_no_custom_adapters_left, merge_and_unload_custom_adapters

import uuid


__all__ = [
    "BasePipelineWrapper",
    "StableDiffusionPipelineWrapper",
    "StableDiffusion3PipelineWrapper",
    "FluxPipelineWrapper",
    "PipelineFactory",
    "add_custom_adapters", "assert_no_custom_adapters_left", "merge_and_unload_custom_adapters"
]

from ..utils.memory import memory_cleanup


def reload_wrapper(wrapper: BasePipelineWrapper, device='cuda', tmp_save_path=None, ignore=None) -> BasePipelineWrapper:

    if tmp_save_path is None:
        random_id = str(uuid.uuid4())
        tmp_save_path = f"models/tmp_{device}_{random_id}"

    # Save the current pipeline to a temporary directory
    old_pipe_config = copy.deepcopy(wrapper.pipe_config)
    tmp_pipe_config = replace(wrapper.pipe_config, model_name_or_path=tmp_save_path)

    wrapper.save_pipeline(tmp_save_path)
    print(f"Saved pipeline to temporary path: {tmp_pipe_config.model_name_or_path}")

    wrapper.teardown()
    del wrapper
    memory_cleanup()

    print(f"Saved pipeline with this pipe config: {tmp_pipe_config}")

    # Reload a fresh pipeline (and get the modified delta components)
    pipe, delta = PipelineFactory.create_pipeline(config=tmp_pipe_config, ignore=ignore, return_local_component_names=True)
    pipe.pipe_config = old_pipe_config

    # Delete the temporary directory
    shutil.rmtree(tmp_save_path)
    print(f"Deleted temporary path again: {tmp_pipe_config.model_name_or_path}")

    wrapper: BasePipelineWrapper = PipelineFactory.wrap_pipeline(pipe).to(device)
    wrapper.track_modified_components(delta)

    return wrapper

