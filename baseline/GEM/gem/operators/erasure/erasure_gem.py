import contextlib
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from tqdm import tqdm

from gem.common.stratified import stratified_sampling
from gem.configs import AdaptationConfig
from gem.operators.erasure import ErasureBaseConfig, ErasureOperator
from gem.operators.utils import EMPTY_CONCEPT, detach_all
from gem.wrappers import BasePipelineWrapper

from gem.utils.debug import assert_model_gradients_valid, assert_tensor_valid
from gem.utils.memory import memory_cleanup


@dataclass
class GEMConfig(ErasureBaseConfig):
    method: str = "gem"

    # General settings
    anchors: List[str] = field(default_factory=lambda: [EMPTY_CONCEPT])

    # Training steps
    max_step_budget: int = 250  # if None, use n_iterations x sum of partial denoising timesteps
    n_iterations: int = 30  # only used if max_step_budget is None
    eta: float = 1.0
    negative_guidance_scale: float = 1.0  # zero means no negative guidance

    # Trajectory settings
    use_inference_timesteps: bool = True
    partial_denoising_mode: str = "student"
    partial_denoising_guidance_scale: float = 3.0
    partial_denoising_end_at_timestep: int = 10  # Defaults to inference time steps of pipeline
    partial_denoising_ignore_until_timestep: int = 0
    partial_denoising_inference_timesteps: int = None  # Defaults to pipe_config.num_inference_steps

    # General loss settings
    learning_rate: float = 1e-3
    gamma: float = 100.0
    gradient_accumulation_steps: int = 1
    steps_between_validation: int = 10

    max_parallel_timesteps: int = 6

    @staticmethod
    def get_default_adaptation_config() -> Optional[AdaptationConfig]:
        return AdaptationConfig(adapter_mode="lora", subset_name='qk_dual')


@dataclass
class GEMSD3Config(GEMConfig):
    learning_rate: float = 2e-4

    partial_denoising_end_at_timestep: int = 10
    partial_denoising_inference_timesteps: int = 40

    @staticmethod
    def get_default_adaptation_config() -> Optional[AdaptationConfig]:
        # Default to ESD-x
        return AdaptationConfig(adapter_mode="lora", subset_name='qk_dual')


class GEM(ErasureOperator):
    """
    GEM implementation.
    """

    def __init__(self, name: str = "gem"):
        super(GEM, self).__init__(name=name)

    def __call__(self, wrapper: BasePipelineWrapper, config: GEMConfig):
        super(GEM, self).__call__(wrapper, config)
        wrapper.eval()

        assert config.gradient_accumulation_steps == 1, "Gradient accumulation is not yet implemented for GEM."
        prompt_augmentation = config.prompt_augmentation.create() if config.prompt_augmentation else None

        if config.use_wandb and config.initial_validation:
            wrapper.log_samples_to_wandb(concepts=config.targets + list(config.validation_concepts),
                                         templates=config.validation_templates, step=0, operator_config=config,
                                         generator=torch.Generator().manual_seed(config.val_seed))

        # Define the optimizer and loss function
        optimizer = torch.optim.AdamW(wrapper.parameters(), lr=config.learning_rate)

        # Determine number of inference steps used for GEM internal denoising
        internal_gem_num_inference_steps = config.partial_denoising_inference_timesteps or wrapper.pipe_config.num_inference_steps

        # Determine max timestep until which to denoise
        if config.partial_denoising_end_at_timestep is not None:
            max_timestep = config.partial_denoising_end_at_timestep
        else:
            max_timestep = internal_gem_num_inference_steps

        if config.max_step_budget is not None:
            expected_n_iterations = 1 + config.max_step_budget // max_timestep
        elif config.n_iterations is not None:
            # Use fixed number of iterations
            expected_n_iterations = config.n_iterations
        else:
            raise ValueError("Either max_step_budget or n_iterations must be specified.")

        # Pre-encode the concepts
        neutral_embeddings = wrapper.encode_prompt(EMPTY_CONCEPT)

        target_embeddings_list = [wrapper.encode_prompt(target) for target in config.targets]

        if len(config.anchors) == 1:
            config.anchors = config.anchors * len(target_embeddings_list)

        if len(target_embeddings_list) != len(config.anchors):
            raise ValueError(
                f"Number of anchors ({len(target_embeddings_list)}) must match number of anchors ({len(config.anchors)})."
            )

        anchor_embeddings_list = [wrapper.encode_prompt(anchor) for anchor in config.anchors]

        # Stratified sampling for anchors and retention concepts
        local_generator = torch.Generator().manual_seed(config.train_seed)
        balanced_anchor_indices = stratified_sampling(range(len(anchor_embeddings_list)), expected_n_iterations, local_generator)

        # Set the random seed for training
        train_generator = torch.Generator(device=wrapper.device).manual_seed(config.train_seed)

        if config.use_inference_timesteps:
            print(f"Using inference schedule ({internal_gem_num_inference_steps} steps) instead of training schedule ({wrapper.scheduler.config.num_train_timesteps} steps)")

        # Training loop
        progress_bar = tqdm(range(expected_n_iterations), desc="Training GEM", unit="step")

        # Initialize step budget
        step_budget = config.max_step_budget if config.max_step_budget is not None else (expected_n_iterations * max_timestep)
        print("Starting training with step budget:", step_budget)

        print(f"Using {config.max_parallel_timesteps} as the max. number of timesteps to process in parallel.")

        # Main training loop
        global_step = 0
        for global_step in progress_bar:
            wrapper.train()

            # Sample an anchor and a retention concept (if any)
            target_anchor_idx = balanced_anchor_indices[global_step]

            # Apply prompt augmentation (if provided)
            if prompt_augmentation is None:
                # Create a detached copy of the embeddings
                neutral_embeddings = detach_all(neutral_embeddings)
                target_embeddings = detach_all(target_embeddings_list[target_anchor_idx])
                anchor_embeddings = detach_all(anchor_embeddings_list[target_anchor_idx])

            else:
                anchor = config.anchors[target_anchor_idx]
                target = config.targets[target_anchor_idx]

                anchor_prompt, target_prompt = prompt_augmentation.apply(
                    anchor, target
                )
                neutral_embeddings = wrapper.encode_prompt(EMPTY_CONCEPT)
                target_embeddings = wrapper.encode_prompt(target_prompt)
                anchor_embeddings = wrapper.encode_prompt(anchor_prompt)

            # Perform partial denoising to generate a latent
            with torch.no_grad():
                if config.partial_denoising_mode == 'teacher':
                    context_manager = wrapper.disable_adapter()
                elif config.partial_denoising_mode == 'student':
                    context_manager = contextlib.nullcontext()
                else:
                    raise ValueError(f"Unknown partial denoising mode: {config.partial_denoising_mode}")

                # Sample end timestep if enabled
                sampled_end_timestep = max_timestep

                print(f"Partial denoising will end at timestep index: {sampled_end_timestep} / {internal_gem_num_inference_steps}")

                with context_manager:
                    _, all_latents_of_trajectory = wrapper.partially_denoise_until_timestep(
                        prompts=None,
                        embeds=target_embeddings,
                        image_size=wrapper.pipe_config.image_size,
                        num_inference_steps=internal_gem_num_inference_steps,
                        guidance_scale=config.partial_denoising_guidance_scale,
                        end_at_timestep_idx=sampled_end_timestep,
                        generator=train_generator,
                        return_all_latents=True
                    )

                assert not torch.isnan(all_latents_of_trajectory[-1]).any(), "Partially Denoised Latents contain NaN values"
                assert not torch.isinf(all_latents_of_trajectory[-1]).any(), "Partially Denoised Latents contain Inf values"

            traj_loss = 0.0
            optimizer.zero_grad()

            n_latents = len(all_latents_of_trajectory)

            if n_latents > config.max_parallel_timesteps:
                print(f"Processing {n_latents} latents in batches of {config.max_parallel_timesteps} due to GPU memory constraints.")

            for batch_start in range(0, n_latents, config.max_parallel_timesteps):
                batch_end = min(batch_start + config.max_parallel_timesteps, n_latents)
                batch_latents = all_latents_of_trajectory[batch_start:batch_end]

                # Stack latents for parallel processing
                stacked_latents = torch.stack(batch_latents, dim=1).squeeze(0)

                # Set timesteps for training (use inference timesteps if flag is enabled)
                num_timesteps = internal_gem_num_inference_steps if config.use_inference_timesteps else wrapper.scheduler.config.num_train_timesteps
                wrapper.set_timesteps(stacked_latents.shape[1], num_timesteps)

                # Create timestep indices for parallel processing (keep on CPU for scheduler indexing)
                base_idx = config.partial_denoising_ignore_until_timestep + batch_start
                active_timestep_idx = torch.arange(base_idx, base_idx + stacked_latents.shape[0])

                print(f"Processing batch [{batch_start}:{batch_end}] with timestep indices {active_timestep_idx.tolist()}")

                # Teacher model predictions (no gradient-tracking)
                with torch.no_grad():
                    with wrapper.disable_adapter():
                        anchor_score = wrapper.predict_noise(active_timestep_idx, stacked_latents, anchor_embeddings)

                predicted_target_score = wrapper.predict_noise(active_timestep_idx, stacked_latents, target_embeddings)

                assert_tensor_valid(predicted_target_score)

                # Teacher model prediction for target (needed for negative guidance and triplet loss)
                with torch.no_grad():
                    with wrapper.disable_adapter():
                        target_score = wrapper.predict_noise(active_timestep_idx, stacked_latents, target_embeddings)

                # Target Loss
                pred_flat = predicted_target_score.float().reshape(predicted_target_score.shape[0], -1)
                pos_flat = anchor_score.float().reshape(anchor_score.shape[0], -1)
                neg_flat = target_score.float().reshape(target_score.shape[0], -1)

                # Calculate Distances
                dist_pos = torch.nn.functional.pairwise_distance(pred_flat, pos_flat, p=2)
                dist_neg = torch.nn.functional.pairwise_distance(pred_flat, neg_flat, p=2)

                target_loss = torch.relu(dist_pos - config.eta * dist_neg).sum()

                # Reset timesteps (skip if already using inference timesteps)
                if not config.use_inference_timesteps:
                    wrapper.set_timesteps(stacked_latents.shape[1], internal_gem_num_inference_steps)

                loss = target_loss
                loss = (config.gamma * loss) / n_latents
                assert_tensor_valid(loss)

                torch.cuda.empty_cache()

                num_parallel_steps = predicted_target_score.shape[0]
                print(
                    f"Processed {num_parallel_steps} timesteps in parallel. Budget left: {step_budget}, Global Step: {global_step}")

                self.log_loss_and_metrics_to_wandb(loss=float(loss.detach()), metrics={
                    'target_loss': float(target_loss.detach()),
                    'num_parallel_timesteps': num_parallel_steps
                }, step=global_step + 1, use_wandb=config.use_wandb)

                loss.backward()
                step_budget -= num_parallel_steps
                traj_loss += float(loss.detach())

                # Clean up batch memory
                del anchor_score, predicted_target_score, stacked_latents

                if target_score is not None:
                    del target_score

                memory_cleanup()

                if step_budget <= 0:
                    print(f"Step budget ({config.max_step_budget}) exhausted, breaking out of batch loop.")
                    break

            self.log_loss_and_metrics_to_wandb(
                loss=traj_loss, step=global_step+1, use_wandb=config.use_wandb
            )

            if config.max_gradient_norm:
                torch.nn.utils.clip_grad_norm_(wrapper.parameters(), max_norm=config.max_gradient_norm)

            assert_model_gradients_valid(wrapper)

            optimizer.step()

            # Log to WandB
            if config.use_wandb and (step_budget <= 0 or (global_step + 1) % config.steps_between_validation == 0):
                wrapper.log_samples_to_wandb(concepts=config.targets + list(config.validation_concepts),
                                             templates=config.validation_templates,
                                             step=global_step + 1,
                                             operator_config=config,
                                             generator=torch.Generator().manual_seed(config.val_seed))

            if step_budget <= 0:
                print(f"Step budget ({config.max_step_budget}) exhausted. Stopping at global_step {global_step}.")
                break

        if config.use_wandb:
            wrapper.log_samples_to_wandb(concepts=config.targets + list(config.validation_concepts),
                                         templates=config.validation_templates,
                                         step=global_step + 1,
                                         operator_config=config,
                                         generator=torch.Generator().manual_seed(config.val_seed))

        del optimizer, neutral_embeddings, anchor_embeddings_list, target_embeddings_list
        memory_cleanup()
