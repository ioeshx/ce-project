from gem.wrappers.base_pipeline_wrapper import BasePipelineWrapper
from gem.wrappers.parameter_subset_selector import ParameterSubsetSelector


class StableDiffusionPipelineWrapper(BasePipelineWrapper):

    def set_timesteps(self, image_seq_len, num_timesteps):
        self.scheduler.set_timesteps(num_timesteps)
        self.pipe._num_timesteps = len(self.scheduler.timesteps)

    def predict_noise(self, timestep_idx, latents, embeds, num_images_per_prompt=1, image_size=512, generator=None):
        height, width = image_size, image_size
        prompt_embeds, negative_prompt_embeds = embeds

        latents = self.scheduler.scale_model_input(latents, self.scheduler.timesteps[timestep_idx])

        batch_size = len(latents)
        num_channels_latents = self.unet.config.in_channels

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

        noise_pred = self.unet(
            latents, self.scheduler.timesteps[timestep_idx].unsqueeze(0).to(self.device),
            encoder_hidden_states=prompt_embeds.to(latents.dtype),
            return_dict=False,
        )[0]

        return noise_pred

    def encode_prompt(self, prompt, do_classifier_free_guidance=False):
        """
        This method returns the following:
        prompt_embeds = CLIP embeddings
        negative_prompt_embeds = CLIP embeddings for negative prompts
        """
        return self.pipe.encode_prompt(
            prompt=prompt,
            num_images_per_prompt=1,
            device=self.device,
            do_classifier_free_guidance=do_classifier_free_guidance
        )

    def encode_image_to_latents(self, image, generator=None):
        latents = self.vae.encode(image.to(next(self.parameters()).dtype).to(self.device)).latent_dist.sample(generator=generator)
        return latents * self.vae.config.scaling_factor

    def create_optional_embeds_dict(self, embeds):
        if embeds is not None:
            prompt_embeds, negative_prompt_embeds = embeds
        else:
            prompt_embeds, negative_prompt_embeds = [None] * 2

        return {
            'prompt_embeds': prompt_embeds,
            'negative_prompt_embeds': negative_prompt_embeds
        }

    def _get_modules_for_subset(self, subset_name):
        supported_subsets = [
            "full",
            "uncond",
            "attn",
            "xattn",
            "xattn_k",
            "xattn_v",
            "xattn_kv",
            "xattn_qkv",
            "uce_subset"
        ]

        subset_selector = ParameterSubsetSelector(self)

        if subset_name == "full":
            target_modules = subset_selector.find(startswith=["unet"])

        elif subset_name == "uncond":
            target_modules = subset_selector.find(startswith=["unet"], exclude=["attn2"])

        elif subset_name == "attn":
            target_modules = subset_selector.find(startswith=["unet"], endswith=["to_k", "to_v", "to_out"])

        elif subset_name == "xattn":
            target_modules = subset_selector.find(startswith=["unet"], contains=["attn2"])

        elif subset_name == "xattn_k":
            target_modules = subset_selector.find(startswith=["unet"], contains=["attn2"], endswith=["to_k"])

        elif subset_name == "xattn_v":
            target_modules = subset_selector.find(startswith=["unet"], contains=["attn2"], endswith=["to_v"])

        elif subset_name in ["xattn_kv", "uce_subset"]:
            target_modules = subset_selector.find(startswith=["unet"], contains=["attn2"], endswith=["to_k", "to_v"])

        elif subset_name == "xattn_qkv":
            target_modules = subset_selector.find(startswith=["unet"], contains=["attn2"], endswith=["to_q", "to_k", "to_v"])

        else:
            raise NotImplementedError(
                f"The provided subset_name {subset_name} is not (yet) supported. Choose one of: {supported_subsets}.")

        return target_modules
