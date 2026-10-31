import torch


class BaseAdapter(torch.nn.Module):

    def __init__(self, base_module: torch.nn.Module):
        super(BaseAdapter, self).__init__()
        self.base_module = base_module

        # Freeze the base module
        for param in self.base_module.parameters():
            param.requires_grad = False

        self.adapted_module = None
        self.adapter_enabled = True

    def disable_adapter(self):
        """Method to disable the adapter."""
        self.adapter_enabled = False

    def enable_adapter(self):
        """Method to enable the adapter."""
        self.adapter_enabled = True

    def get_adapter_stats(self, *args, **kwargs):
        return {}

    def merge(self):
        raise NotImplementedError

    def unfreeze(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def _get_adapted_params(self):
        raise NotImplementedError


class MultiAdapterHandler(torch.nn.Module):

    def __init__(self, base_module: torch.nn.Module):

        super(MultiAdapterHandler, self).__init__()
        self.base_module = base_module

        # Freeze the base module
        for param in self.base_module.parameters():
            param.requires_grad = False

        self.active_adapter_name = None
        self.adapters = torch.nn.ModuleDict()

    def __call__(self, *args, **kwargs):
        if self.active_adapter_name is None:
            return self.base_module(*args, **kwargs)

        return self.get_active_adapter()(*args, **kwargs)

    def add_adapter(self, adapter_cls, adapter_name):
        assert adapter_name not in self.adapters, f"Adapter {adapter_name} already exists."

        adapter = adapter_cls(self.base_module)
        assert isinstance(adapter, BaseAdapter)
        self.adapters[adapter_name] = adapter
        self.adapters[adapter_name].enable_adapter()
        self.active_adapter_name = adapter_name

        # print(f"Added new adapter ({adapter_name}) to the MultiAdapterHandler with base_module ({type(self.base_module)}).")

    def disable_adapter(self):
        """Method to disable the adapter."""
        if self.active_adapter_name:
            self.get_active_adapter().disable_adapter()
            self.active_adapter_name = None

    def enable_adapter(self, adapter_name: str = 'default'):
        """Method to enable the adapter."""
        self.active_adapter_name = adapter_name
        self.get_active_adapter().enable_adapter()

    def get_adapter_stats(self, *args, **kwargs):
        return self.get_active_adapter().get_adapter_states(*args, **kwargs)

    def merge(self, *args, **kwargs):
        self.get_active_adapter().merge(*args, **kwargs)

    def unfreeze(self):
        self.get_active_adapter().unfreeze()

    def reset(self):
        self.get_active_adapter().reset()

    def get_active_adapter(self):
        return self.adapters[self.active_adapter_name]

    def _get_adapted_params(self):
        return self.get_active_adapter()._get_adapted_params()
