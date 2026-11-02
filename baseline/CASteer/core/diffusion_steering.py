import logging
import torch
from collections import defaultdict
from typing import Optional, Dict, Any, List, Tuple

import enum

from .controller import (
    logger,
    VectorControl,
)


class DiffusionModelType(enum.StrEnum):
    SANA = 'sana'
    SD = 'sd'
    SD3 = 'sd3'
    FLUX = 'flux'

    @staticmethod
    def from_model(model: str) -> 'DiffusionModelType':
        """Convert string to DiffusionModelType enum"""
        model = model.lower().strip()

        if model == 'sd35_medium':
            return DiffusionModelType.SD3
        elif model == 'flux1_dev':
            return DiffusionModelType.FLUX
        elif model in ['sana', 'sana-sprint', 'sana-06', 'sana-sprint-06', 'sana15']:
            return DiffusionModelType.SANA
        elif model in ['sd14', 'sd21', 'sd21-turbo', 'sdxl', 'sdxl-turbo']:
            return DiffusionModelType.SD
        else:
            raise ValueError(f"Unknown model type: {model}. Supported types: {list(DiffusionModelType)}")


class HookManager:
    """Manages registration and cleanup of hooks for cross-attention output steering."""

    def __init__(self, model_type: DiffusionModelType):
        self.hooks = []
        self.controls = []
        self.module_info = {}
        self.model_type = model_type

    def register_vector_controls_with_hooks(self, model, *controls: VectorControl):
        """Register vector controls using PyTorch hooks on cross-attention outputs."""
        self.controls = list(controls)
        self._clear_hooks()

        block_count = self._register_hooks_recursive(model)

        for control in self.controls:
            control.num_attn_layers = block_count

        return self.hooks

    def _register_hooks_recursive(self, model) -> int:
        """Recursively find transformer blocks and register hooks."""
        block_count = 0

        if self.model_type in (DiffusionModelType.SD3, DiffusionModelType.FLUX):
            return self._register_dit_hooks(model)
        if self.model_type == DiffusionModelType.SANA:
            block_count += self._register_sana_model_hooks(model)
        elif self.model_type == DiffusionModelType.SD:
            for name, module in model.named_children():
                if "down" in name:
                    block_count += self._register_hooks_in_submodule(module, "down")
                elif "up" in name:
                    block_count += self._register_hooks_in_submodule(module, "up")
                elif "mid" in name:
                    block_count += self._register_hooks_in_submodule(module, "mid")
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return block_count

    def _register_dit_hooks(self, model):
        """Steer only text-value contributions to image tokens in joint attention.

        Native attention, its joint softmax denominator, text outputs, image
        self-attention and MLP remain intact. A second SDPA with zero image
        values extracts the cross-modal component without an attention matrix.
        """
        from diffusers.models.embeddings import apply_rotary_emb
        import torch.nn.functional as F

        flux = self.model_type == DiffusionModelType.FLUX
        text_length = 0

        def remember_text_length(module, args, kwargs):
            nonlocal text_length
            text_length = kwargs['encoder_hidden_states'].shape[1]

        self.hooks.append(model.register_forward_pre_hook(remember_text_length, with_kwargs=True))

        def register(block, single):
            attn = block.attn
            state = {}

            def collect_cross(module, args, kwargs):
                hidden = kwargs.get('hidden_states', args[0] if args else None)
                context = kwargs.get('encoder_hidden_states', args[1] if len(args) > 1 else None)
                batch = hidden.shape[0]
                heads = attn.heads
                query = attn.to_q(hidden).view(batch, -1, heads, attn.to_q.out_features // heads).transpose(1, 2)
                key = attn.to_k(hidden).view_as(query.transpose(1, 2)).transpose(1, 2)
                if attn.norm_q is not None:
                    query = attn.norm_q(query)
                if attn.norm_k is not None:
                    key = attn.norm_k(key)
                if single:
                    values = attn.to_v(hidden).view_as(query.transpose(1, 2)).transpose(1, 2)
                    values = torch.cat([values[:, :, :text_length], torch.zeros_like(values[:, :, text_length:])], dim=2)
                    offset = text_length
                else:
                    context_key = attn.add_k_proj(context).view(batch, -1, heads, query.shape[-1]).transpose(1, 2)
                    context_value = attn.add_v_proj(context).view_as(context_key.transpose(1, 2)).transpose(1, 2)
                    if attn.norm_added_k is not None:
                        context_key = attn.norm_added_k(context_key)
                    image_zeros = torch.zeros_like(key)
                    if flux:
                        context_query = attn.add_q_proj(context).view_as(context_key.transpose(1, 2)).transpose(1, 2)
                        if attn.norm_added_q is not None:
                            context_query = attn.norm_added_q(context_query)
                        query = torch.cat([context_query, query], dim=2)
                        key = torch.cat([context_key, key], dim=2)
                        values = torch.cat([context_value, image_zeros], dim=2)
                        offset = text_length
                    else:
                        key = torch.cat([key, context_key], dim=2)
                        values = torch.cat([image_zeros, context_value], dim=2)
                        offset = 0
                rotary = kwargs.get('image_rotary_emb')
                if rotary is not None:
                    query = apply_rotary_emb(query, rotary)
                    key = apply_rotary_emb(key, rotary)
                image_query = query[:, :, offset:] if flux else query
                mask = kwargs.get('attention_mask')
                if mask is not None and mask.shape[-2] not in (1, image_query.shape[2]):
                    mask = mask[..., offset:, :]
                cross = F.scaled_dot_product_attention(image_query, key, values, attn_mask=mask, dropout_p=0.)
                state['cross'] = cross.transpose(1, 2).reshape(batch, -1, heads * query.shape[-1])

            def steer_projected(module, args, output):
                # FLUX single blocks fuse attention+MLP in proj_out. Only its
                # attention columns contribute to this delta; MLP is untouched.
                cross = state.pop('cross')
                cross = F.linear(cross, module.weight[:, :cross.shape[-1]], bias=None)
                controlled = cross.unsqueeze(-2)
                for control in self.controls:
                    controlled = control(controlled, 'single' if single else 'joint')
                delta = controlled.squeeze(-2) - cross
                if single:
                    return torch.cat([output[:, :text_length], output[:, text_length:] + delta.to(output.dtype)], dim=1)
                return output + delta.to(output.dtype)

            self.hooks.append(attn.register_forward_pre_hook(collect_cross, with_kwargs=True))
            projection = block.proj_out if single else attn.to_out[0]
            self.hooks.append(projection.register_forward_hook(steer_projected))

        for block in model.transformer_blocks:
            register(block, False)
        single_blocks = model.single_transformer_blocks if flux else []
        for block in single_blocks:
            register(block, True)
        return len(model.transformer_blocks) + len(single_blocks)

    def _register_sana_model_hooks(self, model) -> int:
        """Register hooks for SANA model architecture."""
        block_count = 0

        if hasattr(model, 'blocks'):
            for block in model.blocks:
                self._register_hooks_for_sana_block(block, "sana")
                block_count += 1

        if hasattr(model, 'transformer') and hasattr(model.transformer, 'blocks'):
            for block in model.transformer.blocks:
                self._register_hooks_for_sana_block(block, "sana")
                block_count += 1

        # Fallback: search recursively
        if block_count == 0:
            for name, module in model.named_modules():
                class_name = module.__class__.__name__
                if any(sana_name in class_name for sana_name in ['SanaTransformerBlock']):
                    self._register_hooks_for_sana_block(module, "sana")
                    block_count += 1

        return block_count

    def _register_hooks_in_submodule(self, module, place_in_unet: str) -> int:
        """Register hooks in a submodule for a specific place in UNet (SD architecture)."""
        block_count = 0

        for name, submodule in module.named_modules():
            class_name = submodule.__class__.__name__

            if class_name == 'BasicTransformerBlock':
                self._register_hooks_for_sd_block(submodule, place_in_unet)
                block_count += 1
            elif any(sana_name in class_name for sana_name in ['SanaTransformerBlock']):
                self._register_hooks_for_sana_block(submodule, place_in_unet)
                block_count += 1

        return block_count

    def _register_hooks_for_sd_block(self, block, place_in_unet: str):
        """Register cross-attention output hook for a Stable Diffusion BasicTransformerBlock."""
        self.module_info[id(block)] = place_in_unet

        for control in self.controls:
            # Hook into the cross-attention output (attn2)
            if hasattr(block, 'attn2') and block.attn2 is not None:
                hook = block.attn2.register_forward_hook(
                    self._create_attn_output_hook(control, place_in_unet)
                )
                self.hooks.append(hook)

    def _register_hooks_for_sana_block(self, block, place_in_unet: str):
        """Register cross-attention output hook for a SANA transformer block."""
        self.module_info[id(block)] = place_in_unet

        for control in self.controls:
            attn_module = self._get_sana_attention_module(block)
            if attn_module is not None:
                hook = attn_module.register_forward_hook(
                    self._create_sana_attn_output_hook(control, place_in_unet)
                )
                self.hooks.append(hook)

    def _get_sana_attention_module(self, block):
        """Get the appropriate attention module from a SANA block."""
        possible_names = ['attn2', 'attention', 'self_attn', 'Attention']

        for name in possible_names:
            if hasattr(block, name):
                attn_module = getattr(block, name)
                if attn_module is not None and hasattr(attn_module, 'to_q'):
                    return attn_module

        return None

    def _create_attn_output_hook(self, control: VectorControl, place_in_unet: str):
        """Create a forward hook for SD cross-attention output."""
        def hook_fn(module, input, output):
            if not control.active:
                return output

            output_expanded = output[..., None, :]
            controlled_output = control(output_expanded, place_in_unet)
            return controlled_output[..., 0, :]

        return hook_fn

    def _create_sana_attn_output_hook(self, control: VectorControl, place_in_unet: str):
        """Create a forward hook for SANA cross-attention output."""
        def hook_fn(module, input, output):
            if not control.active:
                return output

            output_expanded = output[..., None, :]
            controlled_output = control(output_expanded, place_in_unet)
            controlled_output = controlled_output[..., 0, :].to(torch.bfloat16)

            return controlled_output

        return hook_fn

    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.module_info.clear()

    def remove_hooks(self):
        """Public method to remove all hooks."""
        self._clear_hooks()

    def reset_controls(self):
        """Reset all controls to initial state."""
        for control in self.controls:
            control.reset()


def diffusion_register_vector_controls_with_hooks(model, *controls: VectorControl, model_type: DiffusionModelType) -> HookManager:
    """
    Register vector controls using PyTorch hooks on cross-attention outputs.

    Args:
        model: The model to register controls on
        *controls: VectorControl instances to register
        model_type: The type of diffusion model

    Returns:
        HookManager: Manager object that can be used to remove hooks later
    """
    manager = HookManager(model_type=model_type)
    manager.register_vector_controls_with_hooks(model, *controls)
    return manager
