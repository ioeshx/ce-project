import torch.nn as nn

from gem.wrappers import BasePipelineWrapper
from peft import PeftModel

from gem.wrappers.custom_adapters.adapter_base import BaseAdapter, MultiAdapterHandler
from gem.wrappers.custom_adapters.adapter_normal import NormalAdapter

SUPPORTED_CUSTOM_ADAPTER_MODES = ['normal']
SUPPORTED_ADAPTER_MODES = ['normal'] + ['lora']


def add_custom_adapters(model: nn.Module, target_modules: dict, adapter_name='default', adapter_mode='normal'):

    if adapter_mode == 'lora':
        raise ValueError("LoRA training mode is not supported with a custom adapter type. Make sure to use the "
                         "official PEFT library instead of the `add_custom_adapters` function!")
    elif adapter_mode == 'normal':
        adapter_class = NormalAdapter
    else:
        raise NotImplementedError(f"The adapter_mode '{adapter_mode}' is not supported!")

    # Freeze model (just to make sure)
    for param in model.parameters():
        param.requires_grad = False

    new_target_modules = {}
    for full_name, module in target_modules.items():
        # Split the full name to traverse the module hierarchy
        name_parts = full_name.split('.')

        # Traverse to the parent module of the target module.
        parent_module = model
        for part in name_parts[:-1]:
            parent_module = getattr(parent_module, part)

        # The last part is the attribute name we want to replace.
        target_attr = name_parts[-1]

        # Get the original module (for reference, if needed)
        original_module = getattr(parent_module, target_attr)

        # Replace the original module with your custom adapter.
        if not isinstance(original_module, MultiAdapterHandler):
            # print(f"No MultiAdapterHandler exists yet, creating a new one.")
            handler = MultiAdapterHandler(original_module)
            setattr(parent_module, target_attr, handler)
        else:
            handler = original_module
            # print(f"MultiAdapterHandler already exists with adapters: {list(handler.adapters.keys())}")

        handler.add_adapter(adapter_class, adapter_name=adapter_name)
        new_target_modules[full_name] = handler

    return new_target_modules


def merge_and_unload_custom_adapters(model: nn.Module, adapted_modules: dict = None):

    if adapted_modules is None and isinstance(model, BasePipelineWrapper):
        adapted_modules = model._target_modules

    if adapted_modules is not None and len(adapted_modules):

        # The adapted_modules dictionary has keys that are 'full module names' within the given model:
        # This is something you get when you use model.named_modules
        for full_name in list(adapted_modules.keys()):

            # Split the full name to traverse the module hierarchy
            name_parts = full_name.split('.')

            # Traverse to the parent module of the target module.
            parent_module = model
            child_module = None
            part = None
            for part in name_parts:
                parent_module = child_module if child_module is not None else parent_module
                child_module = getattr(parent_module, part)

            if part is not None:

                # Merge adapter!
                if isinstance(child_module, MultiAdapterHandler):
                    adapter = child_module
                    adapter.merge()
                    setattr(parent_module, part, adapter.base_module)

    return model


def assert_no_custom_adapters_left(model: nn.Module):
    def check_module(module, name=''):
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child, BaseAdapter):
                raise AssertionError(f"Adapter still present in module: {full_name} ({type(child).__name__})")
            check_module(child, full_name)

    check_module(model)
    print("✅ No adapters left in the model.")


def merge_and_unload_all_adapters(wrapper):
    if isinstance(wrapper, PeftModel):
        print("Merging PEFT adapters ...")
        wrapper = wrapper.merge_and_unload()
        if hasattr(wrapper, "peft_config"):
            del wrapper.peft_config
    else:
        print("Merging Custom adapters ...")
        wrapper = merge_and_unload_custom_adapters(wrapper)

    assert_no_custom_adapters_left(wrapper)
    return wrapper
