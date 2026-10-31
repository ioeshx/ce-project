import torch
import numpy as np

from gem.wrappers.base_pipeline_wrapper import BasePipelineWrapper
from gem.wrappers.parameter_subset_selector import ParameterSubsetSelector
from gem.wrappers.pipeline_wrapper_flux import calculate_shift


class QwenImagePipelineWrapper(BasePipelineWrapper):

    # Qwen-Image is NOT a guidance-distilled model but it still relies on the true_cfg_scale parameter.
    def __call__(self, *args, **kwargs):
        if kwargs.get("guidance_scale", 1.0) > 1.0:
            kwargs['negative_prompt'] = " "
            kwargs['true_cfg_scale'] = kwargs.pop("guidance_scale")

        return super(QwenImagePipelineWrapper, self).__call__(*args, **kwargs)

    def calculate_image_seq_len(self, height, width):
        """
        Flux-specific: includes packing (2x2 patches).
        Returns the number of patches after VAE compression and packing.
        """
        return (height // self.vae_scale_factor // 2) * (width // self.vae_scale_factor // 2)

    def set_timesteps(self, image_seq_len, num_timesteps):
        sigmas = np.linspace(1.0, 1 / num_timesteps, num_timesteps)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.set_timesteps(num_timesteps, sigmas=sigmas, mu=mu)
        self.pipe._num_timesteps = len(self.scheduler.timesteps)

    def predict_noise(self, timestep_idx, latents, embeds, num_images_per_prompt=1, image_size=512, generator=None, num_denoising_steps=200):
        height, width = image_size, image_size
        prompt_embeds, prompt_embeds_mask = embeds

        batch_size = len(latents)
        num_channels_latents = self.transformer.config.in_channels // 4

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            latents.dtype,
            self.device,
            generator,
            latents,
        )
        img_shapes = [[(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)]] * batch_size

        guidance_scale = 1.0  # No guidance effectively
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=self.device)
            guidance = guidance.expand(latents.shape[0]).to(latents.dtype)
        else:
            guidance = None

        timestep = self.scheduler.timesteps[timestep_idx].expand(latents.shape[0]).to(latents.dtype) / 1000

        noise_pred = self.transformer(
            hidden_states=latents,
            timestep=timestep.to(self.device),
            guidance=guidance,
            encoder_hidden_states=prompt_embeds.to(latents.dtype),
            encoder_hidden_states_mask=prompt_embeds_mask.to(latents.dtype),
            img_shapes=img_shapes,
            txt_seq_lens=prompt_embeds_mask.sum(dim=1).tolist(),
            return_dict=False,
        )[0]

        return noise_pred

    def encode_prompt(self, prompt, do_classifier_free_guidance=False):
        """
        This method returns the following:
        0: prompt_embeds = CLIP embeddings
        1: prompt_embeds_mask
        """
        assert not do_classifier_free_guidance
        return self.pipe.encode_prompt(prompt=prompt)

    def encode_image_to_latents(self, image, generator=None):
        batch_size, _, height, width = image.shape
        num_channels_latents = self.transformer.config.in_channels // 4

        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        latents = self.vae.encode(image.to(next(self.parameters()).dtype).to(self.device)).latent_dist.sample(generator=generator)
        latents = self.pipe._pack_latents(latents, batch_size, num_channels_latents, height, width)

        return (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

    def create_optional_embeds_dict(self, embeds):
        """Creates a dictionary with the specific embeds-related keyword arguments for the pipeline"""

        if embeds is not None:
            prompt_embeds, prompt_embeds_mask = embeds
        else:
            prompt_embeds, prompt_embeds_mask = [None] * 2

        return {
            'prompt_embeds': prompt_embeds,
            'prompt_embeds_mask': prompt_embeds_mask
        }

    def _get_modules_for_subset(self, subset_name):
        supported_subsets = [
            "full", "uce_subset", "qk_dual"
        ]

        subset_selector = ParameterSubsetSelector(self)

        if subset_name == "full":
            target_modules = subset_selector.find(startswith=["transformer"])

        elif subset_name == "qk_dual":
            target_modules = subset_selector.find(startswith=["transformer"],
                                                  endswith=["add_q_proj", "add_k_proj", "to_q", "to_k", ])

        elif subset_name == "uce_subset":
            print(self.pipe.transformer)
            target_modules = subset_selector.find(startswith=["transformer"], endswith=['txt_in'])

        else:
            raise NotImplementedError(
                f"The provided subset_name {subset_name} is not (yet) supported. Choose one of: {supported_subsets}.")

        print(self.pipe)

        return target_modules
