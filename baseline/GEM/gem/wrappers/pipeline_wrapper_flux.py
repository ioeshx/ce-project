import torch
import numpy as np

from gem.wrappers.base_pipeline_wrapper import BasePipelineWrapper
from gem.wrappers.parameter_subset_selector import ParameterSubsetSelector


def calculate_shift(image_seq_len, base_seq_len, max_seq_len, base_shift, max_shift):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class FluxPipelineWrapper(BasePipelineWrapper):

    # Uncomment the following override to automatically use true classifier-free-guidance instead of the
    # internalized guidance. FLUX is a guidance-distilled model!
    #def __call__(self, *args, **kwargs):
    #    if kwargs.get("guidance_scale", 1.0) > 1.0:
    #        kwargs['negative_prompt'] = " "
    #        kwargs['true_cfg_scale'] = kwargs.pop("guidance_scale")

    #    return super(FluxPipelineWrapper, self).__call__(*args, **kwargs)

    def calculate_image_seq_len(self, height, width):
        """
        Flux-specific: includes packing (2x2 patches).
        Returns the number of patches after VAE compression and packing.
        """
        return (height // self.vae_scale_factor // 2) * (width // self.vae_scale_factor // 2)

    def set_timesteps(self, image_seq_len, num_timesteps):
        # Validate image_seq_len: should be number of patches, not batch size!
        # Valid range: 64 (128x128 image) to 16384 (2048x2048 image)
        assert 64 <= image_seq_len <= 16384, f"Invalid image_seq_len={image_seq_len}. Expected 64-16384 (number of patches), not batch size!"
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

    def predict_noise(self, timestep_idx, latents, embeds, num_images_per_prompt=1, image_size=512, generator=None,
                      num_denoising_steps=200, guidance_scale=1.0):
        height, width = image_size, image_size
        prompt_embeds, pooled_prompt_embeds, text_ids = embeds

        batch_size = len(latents)
        num_channels_latents = self.transformer.config.in_channels // 4

        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            latents.dtype,
            self.device,
            generator,
            latents,
        )

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
            pooled_projections=pooled_prompt_embeds.to(latents.dtype),
            encoder_hidden_states=prompt_embeds.to(latents.dtype),
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        return noise_pred

    def encode_prompt(self, prompt, do_classifier_free_guidance=False):
        """
        This method returns the following:
        0: prompt_embeds = T5 embeddings
        1: pooled_prompt_embeds = CLIP embeddings
        2: text_ids = ?
        """
        assert not do_classifier_free_guidance
        return self.pipe.encode_prompt(prompt=prompt, prompt_2=prompt)

    def encode_image_to_latents(self, image, generator=None):
        batch_size, _, height, width = image.shape
        num_channels_latents = self.transformer.config.in_channels // 4

        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        latents = self.vae.encode(image.to(next(self.parameters()).dtype).to(self.device)).latent_dist.sample(
            generator=generator)
        latents = self.pipe._pack_latents(latents, batch_size, num_channels_latents, height, width)

        return (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

    def decode_latent_to_image(self, latent, image_size=512):
        """
        Decode a Flux latent tensor to a PIL image.
        
        Args:
            latent: Packed latent tensor from Flux (shape: [batch, seq_len, channels])
            image_size: Image size (height and width)
        
        Returns:
            PIL Image
        """
        height, width = image_size, image_size

        # Unpack latents
        unpacked_latents = self.pipe._unpack_latents(latent, height, width, self.vae_scale_factor)

        # Undo VAE scaling
        vae_shift_factor = self.vae.config.shift_factor or 0.0
        unpacked_latents = (unpacked_latents / self.vae.config.scaling_factor) + vae_shift_factor

        # Decode with VAE
        with torch.no_grad():
            decoded = self.vae.decode(unpacked_latents, return_dict=False)[0]
            pil_image = self.image_processor.postprocess(decoded, output_type='pil')[0]

        return pil_image

    def create_optional_embeds_dict(self, embeds):
        """Creates a dictionary with the specific embeds-related keyword arguments for the pipeline"""

        if embeds is not None:
            prompt_embeds, pooled_prompt_embeds, _ = embeds
        else:
            prompt_embeds, pooled_prompt_embeds, _ = [None] * 3

        return {
            'prompt_embeds': prompt_embeds,
            'pooled_prompt_embeds': pooled_prompt_embeds,
        }

    def _get_modules_for_subset(self, subset_name):
        supported_subsets = [
            "full", "text_encoder", "qk_single", "qk_dual_add_proj", "qk_dual", "qkv_dual", "qk", "qkv", "uce_subset",
            "esd_x", "eraseflow_subset",
        ]

        subset_selector = ParameterSubsetSelector(self)

        if subset_name == "full":
            target_modules = subset_selector.find(startswith=["transformer"])

        elif subset_name == "text_encoder":
            target_modules = subset_selector.find(startswith=["text_encoder"])

        elif subset_name == "qk_single":
            target_modules = subset_selector.find(startswith=["transformer"], contains=["single_transformer_blocks"],
                                                  endswith=["add_q_proj", "add_k_proj", "to_q", "to_k", ])

        elif subset_name == "qk_dual_add_proj":
            # EraseAnything QK subset (for FLUX[dev])
            target_modules = subset_selector.find(startswith=["transformer"], exclude=["single_transformer_blocks"],
                                                  endswith=["add_q_proj", "add_k_proj"])
        elif subset_name == "qk_dual":
            target_modules = subset_selector.find(startswith=["transformer"], exclude=["single_transformer_blocks"],
                                                  endswith=["add_q_proj", "add_k_proj", "to_q", "to_k", ])

        elif subset_name == "qkv_dual":
            target_modules = subset_selector.find(startswith=["transformer"], exclude=["single_transformer_blocks"],
                                                  endswith=["add_q_proj", "add_k_proj", "add_v_proj", "to_q", "to_k",
                                                            "to_v"])

        elif subset_name == "qk":
            target_modules = subset_selector.find(startswith=["transformer"],
                                                  endswith=["add_q_proj", "add_k_proj", "to_q", "to_k", ])

        elif subset_name == "qkv":
            target_modules = subset_selector.find(startswith=["transformer"],
                                                  endswith=["add_q_proj", "add_k_proj", "add_v_proj", "to_q", "to_k",
                                                            "to_v"])

        elif subset_name == "esd_x":
            target_modules = subset_selector.find(startswith=["transformer"], contains=["attn"],
                                                  endswith=["to_k", "to_v"])

        elif subset_name == "uce_subset":
            target_modules = subset_selector.find(startswith=["transformer"], contains=['context_embedder'])
            target_modules.update(
                subset_selector.find(startswith=["transformer"], contains_substring=['text_embedder.linear_1']))

        elif subset_name == "eraseflow_subset":
            target_modules = subset_selector.find(contains_substring=['attn.add_k_proj'])
            target_modules.update(subset_selector.find(contains_substring=['attn.add_q_proj']))
            target_modules.update(subset_selector.find(contains_substring=['attn.add_v_proj']))
            target_modules.update(subset_selector.find(contains_substring=['attn.to_add_out']))
            target_modules.update(subset_selector.find(contains_substring=['attn.to_out.0']))
            target_modules.update(subset_selector.find(contains_substring=['attn.to_k']))
            target_modules.update(subset_selector.find(contains_substring=['attn.to_q']))
            target_modules.update(subset_selector.find(contains_substring=['attn.to_v']))

        else:
            raise NotImplementedError(
                f"The provided subset_name {subset_name} is not (yet) supported. Choose one of: {supported_subsets}.")

        return target_modules
