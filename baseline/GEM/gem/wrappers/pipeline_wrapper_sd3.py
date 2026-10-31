import numpy as np

from gem.wrappers.base_pipeline_wrapper import BasePipelineWrapper
from gem.wrappers.parameter_subset_selector import ParameterSubsetSelector


class StableDiffusion3PipelineWrapper(BasePipelineWrapper):

    def set_timesteps(self, image_seq_len, num_timesteps):
        sigmas = np.linspace(1.0, 1 / num_timesteps, num_timesteps)
        self.scheduler.set_timesteps(num_timesteps, sigmas=sigmas)
        self.pipe._num_timesteps = len(self.scheduler.timesteps)

    def predict_noise(self, timestep_idx, latents, embeds, num_images_per_prompt=1, image_size=512, generator=None):
        height, width = image_size, image_size
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = embeds

        batch_size = len(latents)
        num_channels_latents = self.transformer.config.in_channels

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

        noise_pred = self.transformer(
            hidden_states=latents,
            timestep=self.scheduler.timesteps[timestep_idx].unsqueeze(0).to(self.device),
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False
        )[0]

        return noise_pred

    def encode_prompt(self, prompt, do_classifier_free_guidance=False):
        """
        This method returns the following:
        CLIP_1 and CLIP_2 have separate tokenizers!
        0: prompt_embeds = (CLIP_1 + CLIP_2 + T5) embeddings
        1: negative_prompt_embeds
        2: pooled_prompt_embeds = (CLIP_1 + CLIP_2)
        3: negative_pooled_prompt_embeds
        """
        return self.pipe.encode_prompt(prompt=prompt, prompt_2=prompt, prompt_3=prompt, do_classifier_free_guidance=do_classifier_free_guidance)

    def encode_image_to_latents(self, image, generator=None):
        latents = self.vae.encode(image.to(next(self.parameters()).dtype).to(self.device)).latent_dist.sample(generator=generator)
        return (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

    def create_optional_embeds_dict(self, embeds):
        if embeds is not None:
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = embeds
        else:
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = [None] * 4

        return {
            'prompt_embeds': prompt_embeds,
            'negative_prompt_embeds': negative_prompt_embeds,
            'pooled_prompt_embeds': pooled_prompt_embeds,
            'negative_pooled_prompt_embeds': negative_pooled_prompt_embeds
        }

    def _get_modules_for_subset(self, subset_name):
        supported_subsets = [
            "full", "qk_dual_add_proj", "qk_dual", "qkv_dual", "uce_subset"
        ]

        subset_selector = ParameterSubsetSelector(self)

        if subset_name == "full":
            target_modules = subset_selector.find(startswith=["transformer"])

        elif subset_name == "qk_dual_add_proj":
            target_modules = subset_selector.find(startswith=["transformer"],
                                                  endswith=["add_q_proj", "add_k_proj"])

        elif subset_name == "qk_dual":
            target_modules = subset_selector.find(startswith=["transformer"],
                                                  endswith=["add_q_proj", "add_k_proj", "to_q", "to_k", ])

        elif subset_name == "qkv_dual":
            target_modules = subset_selector.find(startswith=["transformer"],
                                                  endswith=["add_q_proj", "add_k_proj", "add_v_proj", "to_q", "to_k",
                                                            "to_v"])

        elif subset_name == "uce_subset":
            target_modules = subset_selector.find(startswith=["transformer"],
                                                  contains=["context_embedder"])
        else:
            raise NotImplementedError(
                f"The provided subset_name {subset_name} is not (yet) supported. Choose one of: {supported_subsets}.")

        return target_modules