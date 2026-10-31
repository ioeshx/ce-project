import torch
import copy

from gem.wrappers.custom_adapters.adapter_base import BaseAdapter


class NormalAdapter(BaseAdapter):

    def __init__(self, base_module: torch.nn.Module):
        super(NormalAdapter, self).__init__(base_module)
        self.reset()

    def forward(self, x, *args, **kwargs):

        # If the adapter is enabled, modify the output
        if self.adapter_enabled:
            # Forward pass through the adapted module
            out = self.adapted_module(x.float(), *args, **kwargs)
        else:
            # Forward pass through the original module
            out = self.base_module(x, *args, **kwargs)

        return out.to(x.dtype)

    def merge(self):
        base_dtype = next(self.base_module.parameters()).dtype
        self.base_module.load_state_dict(self.adapted_module.state_dict())
        self.base_module = self.base_module.to(base_dtype)

    def unfreeze(self):
        # Unfreeze the adapted module
        for param in self.adapted_module.parameters():
            param.requires_grad = True

    def reset(self):
        self.adapted_module = copy.deepcopy(self.base_module).float()
        self.unfreeze()

    def _get_adapted_params(self):
        return self.adapted_module.parameters()