import numpy as np
from contextlib import contextmanager
from PIL import Image

import torch
import torchvision
import wandb

from diffusers import FluxPipeline, StableDiffusion3Pipeline, QwenImagePipeline
from diffusers.training_utils import compute_snr

from gem.configs import PipelineConfig
from gem.utils.image_utils import create_image_grid
from gem.utils.memory import memory_cleanup
from gem.wrappers.custom_adapters.adapter_base import MultiAdapterHandler
from gem.wrappers.custom_adapters.adapter_normal import NormalAdapter


class BasePipelineWrapper(torch.nn.Module):

    def __init__(self, pipe):
        super(BasePipelineWrapper, self).__init__()
        self.pipe = pipe
        self.pipe.safety_checker = None
        self.pipe.requires_safety_checker = False

        # Assign all attributes of the given object, ensuring nn.Modules are registered
        for name, attr in pipe.__dict__.items():
            if isinstance(attr, torch.nn.Module):
                self.add_module(name, attr)
            else:
                setattr(self, name, attr)

        # Inherit the following methods
        self.prepare_latents = pipe.prepare_latents

        # Just to make it explicit (this attribute is set by create_pipeline(...) of the PipelineFactory class)
        self.pipe_config: PipelineConfig = pipe.pipe_config

        self.verbose = False
        self._target_modules = {}
        self.modified_components = []

        # Initially freeze everything
        for param in self.parameters():
            param.requires_grad = False

    # noinspection PyProtectedMember
    @property
    def device(self):
        return self.pipe._execution_device

    def calculate_image_seq_len(self, height, width):
        """
        Calculate the image sequence length (number of patches) for set_timesteps.
        This is model-specific and should be overridden if needed.
        
        For Flux: Includes packing (divide by 2)
        For SD/SD3: No packing
        """
        # Default: no packing (SD models)
        return (height // self.vae_scale_factor) * (width // self.vae_scale_factor)

    def set_timesteps(self, image_seq_len, num_timesteps):
        raise NotImplementedError

    def predict_noise(self, timestep_idx, latents, embeds, num_images_per_prompt=1, image_size=512, generator=None):
        raise NotImplementedError

    def encode_prompt(self, prompt, do_classifier_free_guidance=False):
        raise NotImplementedError

    def encode_image_to_latents(self, image, generator=None):
        raise NotImplementedError

    def create_optional_embeds_dict(self, embeds):
        """Creates a dictionary with the specific embeds-related keyword arguments for the pipeline"""
        raise NotImplementedError

    def _get_modules_for_subset(self, subset_name):
        raise NotImplementedError

    def set_and_get_target_modules(self, subset_name):
        target_modules = self._get_modules_for_subset(subset_name)
        self._target_modules = target_modules

        print("Target modules:")
        for module_name in target_modules:
            print(">", module_name, "->", type(target_modules[module_name]))
        assert target_modules
        return target_modules

    def track_modified_components(self, component_names):
        if len(component_names) == 0:
            print("No modified components need to be tracked...")
            return

        print("Adding modified components:")
        for component_name in component_names:
            print(" - ", component_name)
        print("to the already existing modified components:")
        for component_name in self.modified_components:
            print(" - ", component_name)
        self.modified_components.extend(component_names)
        self.modified_components = list(set(self.modified_components))

    def __call__(self, *args, **kwargs):
        result = self.pipe(*args, **kwargs)

        # Assume result["images"] or result.images is a list or batch of tensors/arrays
        images = result["images"] if isinstance(result, dict) else result.images

        for idx, img in enumerate(images):
            # Convert to tensor if it's a PIL Image or NumPy array
            if not torch.is_tensor(img):
                img = torch.from_numpy(np.array(img))\
                    if hasattr(img, "numpy") else torchvision.transforms.ToTensor()(img)

            # Assert no NaN or Inf
            assert not torch.isnan(img).any(), f"Image at index {idx} contains NaNs"
            assert not torch.isinf(img).any(), f"Image at index {idx} contains Infs"

            # Assert image is not completely black
            if not img.abs().sum() > 1e-5:
                print(f"Critical Warning: Image at index {idx} appears to be completely black (sum close to zero)")

        return result

    def partially_denoise_until_timestep(self, prompts, end_at_timestep_idx=None, embeds=None, guidance_scale=None,
                                         image_size=512, num_inference_steps=28, generator=None,
                                         return_all_latents=False):
        height, width = image_size, image_size

        # Validate the end_at_timestep_idx value
        if end_at_timestep_idx is not None:
            assert end_at_timestep_idx <= num_inference_steps, "end_at_timestep_idx must be less than or equal to num_inference_steps"

        # Save the number of inference steps from before this function was called
        num_steps_before = len(
            self.scheduler.timesteps) if self.scheduler.timesteps is not None else num_inference_steps
        # Calculate image_seq_len using model-specific method
        image_seq_len = self.calculate_image_seq_len(image_size, image_size)
        self.set_timesteps(image_seq_len, num_inference_steps)

        latents_per_step = []

        def controlled_callback(self, step, timestep, callback_kwargs):
            latents = callback_kwargs["latents"]

            # Store latents per step if required
            with torch.no_grad():
                latents_per_step.append(latents.clone().detach())

            # Stop when end_at_timestep_idx is reached
            if end_at_timestep_idx is not None and step >= end_at_timestep_idx:
                raise StopIteration

            return {}  # Continue the loop

        optional_embeds_dict = self.create_optional_embeds_dict(embeds)

        # Use the pipeline's __call__ method with the custom callback
        with torch.no_grad():
            try:
                _ = self(
                    prompt=prompts,
                    **optional_embeds_dict,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    callback_on_step_end=controlled_callback,
                )
            except StopIteration:
                if self.verbose:
                    print(
                        f"[Partial Denoising] Stopped inference at timestep: {num_inference_steps - end_at_timestep_idx}/{num_inference_steps}")

            finally:
                # Restore the previous timestep configuration
                self.set_timesteps(image_seq_len, num_steps_before)

        latents = latents_per_step[-1]

        with torch.no_grad():
            # Unpack first for Flux
            if isinstance(self.pipe, (FluxPipeline, QwenImagePipeline)):
                latents_for_decoding = self.pipe._unpack_latents(latents, height, width, self.vae_scale_factor)
            else:
                latents_for_decoding = latents

            if isinstance(self.pipe, QwenImagePipeline):
                vae_shift = (torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1)
                             .to(latents_for_decoding.device, latents_for_decoding.dtype)
                             )
                vae_scaling = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                    1, self.vae.config.z_dim, 1, 1, 1).to(latents_for_decoding.device, latents_for_decoding.dtype)
            else:
                vae_shift = self.vae.config.shift_factor or 0.0
                vae_scaling = self.vae.config.scaling_factor

            # Then apply scaling and shift
            latents_for_decoding = (latents_for_decoding / vae_scaling) + vae_shift

            image = self.vae.decode(latents_for_decoding, return_dict=False)[0]

            if isinstance(self.pipe, QwenImagePipeline):
                image = image[:, :, 0]

            image = self.image_processor.postprocess(image, output_type='pil')[0]

        return image, latents if not return_all_latents else latents_per_step

    @torch.no_grad()
    def log_samples_to_wandb(
            self, step, operator_config, concepts=None, templates=None, embeds_list=None,
            num_images_per_prompt=1, generator=None, log_prefix='samples/train', n_cols: int = None):

        # TODO Fix bug with num_images_per_prompt > 1 as it leads to num_images_per_prompt^2 outputs sometimes

        # **Ensure at least 1 target is present**
        if not concepts and embeds_list is None:
            raise ValueError("At least one concept or embeds is required.")

        # Generate samples
        torch.cuda.empty_cache()

        if templates is None:
            templates = ["{}"]

        # The invocation differs based on whether embeds_list is provided
        if embeds_list is not None:
            prompts = [None]  # Dummy list to satisfy the loop structure below
            optional_embeds_dict_list = []
            for embeds in embeds_list:
                optional_embeds_dict_list.append(self.create_optional_embeds_dict(embeds))

            n_cols = n_cols or num_images_per_prompt
            print(f"Generating Samples for given embeds.")
        else:
            optional_embeds_dict_list = [{}]  # Dummy list to satisfy the loop structure below
            concepts = concepts

            prompts = [
                template.format(concept)
                for concept in concepts
                for template in templates
                for _ in range(num_images_per_prompt)
            ]
            n_cols = n_cols or (len(templates) * num_images_per_prompt)

            print(f"Generating Samples for the following {len(prompts)} prompts:")
            for i, prompt in enumerate(prompts):
                print(f"({i}) ->", prompt)

        with torch.no_grad():
            self.eval()
            images = []
            batch_size = 8 if self.pipe_config.model_type in ["sd_3", "sd_3_5", "flux"] else 16

            # Manual prompt batching (to avoid OOMs)
            num_prompts = len(prompts)
            for batch_start in range(0, num_prompts, batch_size):
                batch_end = batch_start + batch_size
                prompt_batch = prompts[batch_start:batch_end]

                # The prompt_batch can now be [None] if embeds_list is provided
                if embeds_list is not None:
                    assert len(prompt_batch) == 1 and prompt_batch[0] is None
                    prompt_batch = None

                # Loop over the embeds dicts exactly as before
                for optional_embeds_dict in optional_embeds_dict_list:

                    # Branch preserved EXACTLY as in your original code
                    if self.pipe_config.model_type not in ["sd_3", "sd_3_5"] or embeds_list is None:

                        # NON-SD3 path — keep your original multi-image call
                        result = self(
                            prompt=prompt_batch,
                            **optional_embeds_dict,
                            height=self.pipe_config.image_size,
                            width=self.pipe_config.image_size,
                            num_inference_steps=self.pipe_config.num_inference_steps,
                            guidance_scale=self.pipe_config.inference_guidance_scale,
                            num_images_per_prompt=num_images_per_prompt,
                            generator=(
                                generator if generator is not None
                                else torch.Generator().manual_seed(operator_config.val_seed)
                            ),
                        )
                        images.extend(result.images)

                    else:
                        # SD3 / SD3.5 path — keep your manual per-image loop

                        # ==============================================================================================
                        # SD3 leads to problems when 'num_images_per_prompt' is > 1 together with the embeds-style usage
                        # therefore we handle it manually for now. This is highly likely a bug in the official diffusers
                        # code from huggingface.
                        # ==============================================================================================
                        for _ in range(num_images_per_prompt):
                            result = self(
                                prompt=prompt_batch,
                                **optional_embeds_dict,
                                height=self.pipe_config.image_size,
                                width=self.pipe_config.image_size,
                                num_inference_steps=self.pipe_config.num_inference_steps,
                                guidance_scale=self.pipe_config.inference_guidance_scale,
                                num_images_per_prompt=1,
                                generator=(
                                    generator if generator is not None
                                    else torch.Generator().manual_seed(operator_config.val_seed)
                                ),
                            )
                            images.extend(result.images)

            actual_batch_size = len(images) // num_images_per_prompt
            reordered_images = []

            for i in range(actual_batch_size):
                for j in range(num_images_per_prompt):
                    index = j * actual_batch_size + i
                    reordered_images.append(images[index])

            images = reordered_images

            # Create an image grid with adjusted columns
            print("Produced:", len(images), f"in {n_cols} columns")
            image_grid = create_image_grid(images, cols=n_cols)

            width, height = image_grid.size
            resized_image_grid = image_grid.resize((width // 2, height // 2), Image.LANCZOS)

            # Log to WandB
            wandb.log({log_prefix: [wandb.Image(resized_image_grid, caption=f"(Step: {step})")]})

    @contextmanager
    def disable_adapter(self):
        """
        Context manager that disables all target modules' adapters temporarily.
        Upon exiting the context, the adapters are re-enabled.
        Important: This is overridden by PeftModel's disable_adapter method when using LoRA, so do not rename!
        """
        # Disable all adapters
        active_adapter_name = None
        for module in self._target_modules.values():
            if hasattr(module, "disable_adapter"):
                active_adapter_name = module.active_adapter_name
                module.disable_adapter()
        try:
            yield
        finally:
            # Re-enable all adapters after context block
            for module in self._target_modules.values():
                if hasattr(module, "enable_adapter"):
                    module.enable_adapter(adapter_name=active_adapter_name)

    @contextmanager
    def normal_adapters_temporarily_merged(self):
        # TODO: Extend this for all adapter types including LoRA from peft

        def _get_parent_module_and_attr(obj, module_name):
            """
            For a dotted module name like 'encoder.layer.0.attention.dense',
            returns the parent module object and the last attribute name:
                parent, attr_name = _get_parent_module_and_attr(model, 'encoder.layer.0.attention.dense')
                getattr(parent, attr_name)  # is the module we want
            """
            tokens = module_name.split(".")
            # Traverse to the parent of the last attribute
            for token in tokens[:-1]:
                # If token is an integer string (e.g., layer.0), interpret as list index
                if token.isdigit():
                    obj = obj[int(token)]
                else:
                    obj = getattr(obj, token)
            return obj, tokens[-1]

        print("Temporarily merging adapters ...")
        adapters = {}
        # Replace each target module with its base_module
        for name, module in self._target_modules.items():

            if isinstance(module, MultiAdapterHandler):
                assert isinstance(module.adapters[module.active_adapter_name], NormalAdapter)
                adapted_module = module.adapters[module.active_adapter_name].adapted_module
            else:
                assert isinstance(module, NormalAdapter), \
                    f"Temporary merging only available for NormalAdapter, not instances of class {type(module).__name__}"
                adapted_module = module.adapted_module

            adapters[name] = module  # Save original adapter module

            parent, attr_name = _get_parent_module_and_attr(self.pipe, name)
            setattr(parent, attr_name, adapted_module)  # Swap to base

        try:
            yield
        finally:
            print("Restoring adapters again ...")
            # Restore original adapter modules
            for name, module in adapters.items():
                parent, attr_name = _get_parent_module_and_attr(self.pipe, name)
                setattr(parent, attr_name, module)  # Restore original adapter

    def reset_all_adapters(self):
        # Reset all adapters
        for module in self._target_modules.values():
            module.reset()

    def unfreeze_all_adapters(self):
        # Reset all adapters
        for module in self._target_modules.values():
            module.unfreeze()

    def save_pipeline(self, path, components_to_save=None, save_full=False, verbose=True):
        import os

        os.makedirs(path, exist_ok=True)

        # Extract module names from target modules
        inferred_components_from_target_modules = set(k.split('.')[0] for k in self._target_modules.keys())
        other_modified_components = self.modified_components

        inferred_components = list(inferred_components_from_target_modules) + list(other_modified_components)

        if save_full:
            print("Saving the full entire pipeline.")
            self.pipe.save_pretrained(path)
        else:
            # Add any explicitly provided modules
            if components_to_save:
                inferred_components.append(components_to_save)

            if verbose:
                for module_name in inferred_components:
                    print(f"About to save module: {module_name}")

            saved_any = False

            for module_name in inferred_components:
                component = getattr(self.pipe, module_name, None)
                if component is not None and hasattr(component, "save_pretrained"):
                    component.save_pretrained(os.path.join(path, module_name))
                    saved_any = True
                    print(f"Successfully saved '{module_name}' under {os.path.join(path, module_name)}.")
                else:
                    print(f"Warning: Component '{module_name}' not found or can't be saved.")

            # Fallback: save entire pipeline if nothing was saved
            if not saved_any:
                print("No individual components saved; the checkpoint directory will be empty.")

    def teardown(self):
        if self.verbose:
            print("Tearing down wrapper references to submodels and clearing memory...")

        # Remove all submodules
        for name in list(self._modules.keys()):
            delattr(self, name)

        # Nullify everything by name
        attr_names = list(vars(self).keys())
        for attr in attr_names:
            setattr(self, attr, None)

        # Final purge (only effective when there is no other reference in outer scope)
        memory_cleanup()

    def add_noise_to_latents(self, latents, generator, timestep_idx=None, return_noise=False):
        timesteps = torch.full((len(latents),), self.scheduler.timesteps[timestep_idx], device=latents.device)

        noise = torch.randn(latents.shape, device=latents.device, generator=generator, dtype=latents.dtype)

        # Different schedulers have different forward methods
        if hasattr(self.scheduler, "scale_noise"):
            noisy_latents = self.scheduler.scale_noise(sample=latents, noise=noise, timestep=timesteps)
        else:
            noisy_latents = self.scheduler.add_noise(original_samples=latents, noise=noise, timesteps=timesteps)

        noisy_latents = noisy_latents.to(latents.dtype)
        return (noisy_latents, noise) if return_noise else noisy_latents

    def compute_standard_diffusion_loss(self, latents, timestep_idx, embeddings, generator, snr_gamma=None):

        # Generate noise and add it to the latents
        noisy_latents, noise = self.add_noise_to_latents(
            latents, timestep_idx=timestep_idx, generator=generator, return_noise=True
        )

        # Predict score with the model
        predicted_score = self.predict_noise(timestep_idx, noisy_latents, embeddings)

        timesteps = torch.full((len(latents),), self.scheduler.timesteps[timestep_idx], device=latents.device)

        if isinstance(self.pipe, (StableDiffusion3Pipeline, FluxPipeline)):
            # Rectified flow velocity
            target = noise - latents
        else:
            # Get the target for loss depending on the prediction type
            if self.scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.scheduler.config.prediction_type == "v_prediction":
                target = self.scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

        if snr_gamma is None:
            loss = torch.nn.functional.mse_loss(predicted_score.float(), target.float())
        else:
            # Compute loss-weights as per Section 3.4 of https://huggingface.co/papers/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.scheduler, timesteps)
            mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]

            if self.scheduler.config.prediction_type == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif self.scheduler.config.prediction_type == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)

            loss = torch.nn.functional.mse_loss(self.scheduler.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        # Garbage collection
        del predicted_score, latents

        return loss
