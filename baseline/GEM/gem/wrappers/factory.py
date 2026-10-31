import os
from huggingface_hub import list_repo_tree, RepoFolder
from typing import Optional, List, Union

import torch

from diffusers import Flux2Pipeline, StableDiffusionPipeline, StableDiffusion3Pipeline, QwenImagePipeline
from diffusers import DDIMScheduler

from peft import get_peft_model, LoraConfig, PeftModel

from diffusers import FluxPipeline
from gem.configs.adaptation_config import AdaptationConfig
from gem.configs.pipeline_config import PipelineConfig, MODEL_CONFIGS
from gem.wrappers import FluxPipelineWrapper, StableDiffusion3PipelineWrapper, StableDiffusionPipelineWrapper, \
    BasePipelineWrapper
from gem.wrappers.custom_adapter_utils import add_custom_adapters
from gem.wrappers.pipeline_wrapper_qwenimage import QwenImagePipelineWrapper


def get_base_model_name_or_path(model_type):
    return MODEL_CONFIGS.get(model_type)['model_name_or_path']


def overwrite_pipeline_components_from_directory(pipe, ignore, checkpoint_dir: str):
    """
    Overwrites components of a pipeline with those saved in `checkpoint_dir`.
    Works with both local directories and Hugging Face Hub repo IDs (e.g. 'username/model').
    """
    is_local = os.path.isdir(checkpoint_dir)

    if is_local:
        component_names = [
            name for name in os.listdir(checkpoint_dir)
            if os.path.isdir(os.path.join(checkpoint_dir, name))
        ]
    else:
        # Treat checkpoint_dir as a HF Hub repo ID; list top-level folders
        try:
            tree = list_repo_tree(checkpoint_dir, repo_type="model", recursive=False)
            component_names = [entry.path for entry in tree if isinstance(entry, RepoFolder)]



        except Exception as e:
            print(f"[Overwrite] Could not access HF Hub repo '{checkpoint_dir}': {e}")
            raise ValueError("Checkpoint directory is neither a local directory nor an accessible HF Hub repo.")

    overwritten_components = []

    for component_name in component_names:

        if component_name in (ignore or []):
            continue
        try:
            existing_component = getattr(pipe, component_name, None)
            if existing_component is None:
                print(f"[Overwrite] Skipping '{component_name}': not found on pipeline.")
                continue
            component_class = existing_component.__class__

            if is_local:
                load_kwargs = dict(
                    pretrained_model_name_or_path=os.path.join(checkpoint_dir, component_name),
                )
            else:
                load_kwargs = dict(
                    pretrained_model_name_or_path=checkpoint_dir,
                    subfolder=component_name
                )

            loaded_component = component_class.from_pretrained(
                **load_kwargs,
                force_download=True,
                torch_dtype=getattr(existing_component, "dtype", None),
            )
            setattr(pipe, component_name, loaded_component)
            overwritten_components.append(component_name)
            source = component_name if not is_local else None
            print(f"[Overwrite] Successfully loaded and replaced '{component_name}' from: {checkpoint_dir!r} (subfolder={source!r})")
        except Exception as e:
            print(f"[Overwrite] Failed to load '{component_name}': {e}")

    return overwritten_components


class PipelineFactory:

    @staticmethod
    def create_pipeline(config: PipelineConfig, model_dtype: Optional[torch.dtype] = None,
                        ignore: Optional[List[str]] = None, return_local_component_names: bool = False):

        dtype = model_dtype
        if model_dtype is None:
            if config.model_dtype == "float16":
                dtype = torch.float16
            elif config.model_dtype == "float32":
                dtype = torch.float32
            elif config.model_dtype == "bfloat16":
                dtype = torch.bfloat16
            else:
                print("Unknown model_dtype in config:", config.model_dtype)

        base_model_name = get_base_model_name_or_path(model_type=config.model_type)

        ignored_components_dict = {ignored_component_name: None for ignored_component_name in (ignore or [])}

        if config.model_type in {"sd_1_4", "sd_1_5", "sd_2_1"}:
            pipe = StableDiffusionPipeline.from_pretrained(base_model_name, **ignored_components_dict, torch_dtype=dtype)
            if config.scheduler == 'ddim':
                pipe.scheduler = DDIMScheduler.from_pretrained(base_model_name, subfolder="scheduler")

        elif config.model_type in {"sd_3", "sd_3_5"}:
            pipe = StableDiffusion3Pipeline.from_pretrained(base_model_name, **ignored_components_dict, torch_dtype=dtype)

        elif config.model_type in {"flux", "flux_schnell"}:
            # assert isinstance(config, InferenceConfig) or dtype == torch.float16, ("FLUX currently only works with fp16 for fine-tuning!")
            pipe = FluxPipeline.from_pretrained(base_model_name, **ignored_components_dict, torch_dtype=dtype)

        elif config.model_type in {"flux_2"}:
            pipe = Flux2Pipeline.from_pretrained(base_model_name, **ignored_components_dict, torch_dtype=dtype)
            pipe.enable_model_cpu_offload()

        elif config.model_type == "qwen_image":
            pipe = QwenImagePipeline.from_pretrained(base_model_name, **ignored_components_dict, torch_dtype=dtype)

        else:
            raise NotImplementedError(f"Model type {config.model_type} is not supported!")

        # Now loading only the altered components (delta to base model)
        potentially_modified_components = []
        if base_model_name != config.model_name_or_path:
            potentially_modified_components = overwrite_pipeline_components_from_directory(
                pipe, ignore=ignore, checkpoint_dir=config.model_name_or_path
            )

        setattr(pipe, "pipe_config", config)
        pipe.safety_checker = None
        pipe.requires_safety_checker = False

        return (pipe, potentially_modified_components) if return_local_component_names else pipe

    @staticmethod
    def wrap_pipeline(pipe):

        if isinstance(pipe, StableDiffusionPipeline):
            wrapper = StableDiffusionPipelineWrapper(pipe)

        elif isinstance(pipe, StableDiffusion3Pipeline):
            wrapper = StableDiffusion3PipelineWrapper(pipe)

        elif isinstance(pipe, FluxPipeline):
            wrapper = FluxPipelineWrapper(pipe)

        elif isinstance(pipe, Flux2Pipeline):
            # TODO: Revisit once its available and check if we can reuse the same wrapper like this
            wrapper = FluxPipelineWrapper(pipe)

        elif isinstance(pipe, QwenImagePipeline):
            wrapper = QwenImagePipelineWrapper(pipe)

        else:
            raise NotImplementedError(f"Pipeline class {type(pipe).__name__} is not supported!")

        return wrapper

    @staticmethod
    def prepare_adaptation(adaptation_config: AdaptationConfig, wrapper, adapter_name='default'):

        if isinstance(wrapper, PeftModel):
            print("Warning: The given wrapper is already a PeftModel. The current logic does not yet support a second"
                  "adaptation step using the prepare_adaptation(...) method!"
                  "It works with the custom adapters through the MultiAdapterHandler logic, though."
                  "To be clear: LoRA adaptation via peft currently only works once per model."
                  "No multi-adapter support yet via this method!")

        assert isinstance(wrapper, BasePipelineWrapper)

        if not hasattr(adaptation_config, "adapter_mode") or not adaptation_config.adapter_mode:
            print("Warning: No 'adapter_mode' provided in the 'adaptation_config'. The model has no trainable "
                  "parameters! '_target_modules' might be empty.")
            return wrapper
        else:
            # Identify target modules
            if not wrapper._target_modules:
                target_modules = wrapper.set_and_get_target_modules(subset_name=adaptation_config.subset_name)
            else:
                target_modules = wrapper._target_modules

            # Either add LoRA adapters (with PEFT library)
            if adaptation_config.adapter_mode == 'lora':
                peft_config = LoraConfig(**adaptation_config.lora_config, target_modules=list(target_modules.keys()))
                wrapper = get_peft_model(wrapper, peft_config)
                assert isinstance(wrapper, PeftModel)
                wrapper.print_trainable_parameters()

            # Or add custom adapters
            elif adaptation_config.adapter_mode not in ['none', 'closed_form']:
                wrapper._target_modules = add_custom_adapters(
                    wrapper, target_modules=target_modules,
                    adapter_mode=adaptation_config.adapter_mode,
                    adapter_name=adapter_name
                )

        def cast_training_params(model: Union[torch.nn.Module, List[torch.nn.Module]], dtype=torch.float32):
            if not isinstance(model, list):
                model = [model]
            for m in model:
                for param in m.parameters():
                    # only upcast trainable parameters into fp32
                    if param.requires_grad and param.dtype != dtype:
                        print(f"Casting parameter from dtype {param.dtype} to dtype:", dtype)
                        param.data = param.to(dtype)

        cast_training_params(wrapper)

        print(f"Successfully added '{adaptation_config.adapter_mode}' adapters for parameter subset {adaptation_config.subset_name}!")
        return wrapper
