import json
import time
from dataclasses import asdict, replace

import wandb
import torch
import os

import argparse

from gem.configs import AdaptationConfig
from gem.configs.pipeline_config import PipelineConfig

from gem.operators.operator import Operator
from gem.operators.registry import get_config_and_operator, SUPPORTED_METHODS, SUPPORTED_MODEL_TYPES
from gem.utils.logging import get_logger, log_rng_states, setup_logger, generate_random_run_id
from gem.utils.memory import memory_cleanup

from gem.wrappers.factory import PipelineFactory
from gem.wrappers.custom_adapter_utils import merge_and_unload_all_adapters, assert_no_custom_adapters_left

from gem.utils.seed import seed_everything
from gem.utils.cli import add_args_from_dataclass
from gem.utils.debug import assert_model_params_valid

from dotenv import load_dotenv
load_dotenv()


def run_operator(operator: Operator, operator_config, pipe_config: PipelineConfig, run_id: str, run_save_path: str = None):

    if run_save_path is None:
        run_save_path = os.path.join("models", run_id)

    if os.path.exists(run_save_path) and any(
            os.path.isfile(os.path.join(run_save_path, f)) and not f.endswith('.log')
            for f in os.listdir(run_save_path)
    ):
        print(f"Fine-tuned {operator.name.upper()} model artifacts already exist under: {run_save_path}")
        print("Check the directory and/or specify --run_id and retry! Skipping fine-tuning ...")
    else:
        if operator_config.use_wandb:
            wandb.init(project=os.getenv("WANDB_PROJECT"), config=vars(operator_config), name=run_id)

        log_file_path = f"{run_save_path}/train.log"

        #with logger_context(name=run_id, log_file=log_file_path):
        setup_logger(name=run_id, log_file=log_file_path)
        print(f"Using logger for run_id: {run_id}, log_file: {log_file_path}")
        logger = get_logger()

        logger.info("Pipe Configuration Loaded:")
        logger.info(pipe_config)

        # Print configuration
        logger.info("Operator Configuration Loaded:")
        logger.info(operator_config)

        logger.info(f"Seeding everything with seed: {operator_config.train_seed}")
        seed_everything(operator_config.train_seed)

        #pipe.enable_model_cpu_offload()

        # Important that we do not have dedicated extra pipe = ... variable so that we can delete it in local operator
        wrapper = PipelineFactory.wrap_pipeline(
            PipelineFactory.create_pipeline(pipe_config)
        ).to('cuda')

        # Prepare the wrapper for adaptation (if adaptation_config is provided)
        if hasattr(operator_config, "adaptation_config"):
            wrapper = PipelineFactory.prepare_adaptation(operator_config.adaptation_config, wrapper)
        else:
            logger.warning("No 'adaptation_config' provided in the 'operator_config'. The model has no (new) trainable parameters!")
            assert_no_custom_adapters_left(wrapper)

        if operator_config.use_wandb:
            wandb.watch(wrapper)

        # Log number of trainable parameters
        num_trainable_params = sum(param.numel() for param in wrapper.parameters() if param.requires_grad)
        logger.info(f"# Trainable Parameters: {num_trainable_params}")

        if operator_config.use_wandb:
            wandb.log({"# Trainable Parameters": num_trainable_params}, commit=False)

        # Apply the method
        seed_everything(operator_config.train_seed)
        wrapper.train()

        # ==== Start time and memory tracking ====
        torch.cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()

        operator_config.run_save_path = run_save_path
        logger.info(f"Logging RNG states directly before operator is applied:")
        log_rng_states(logger)

        maybe_new_wrapper = operator(wrapper, config=operator_config)

        # Some operators may return a new wrapper (e.g., STEREO)
        if maybe_new_wrapper is not None:
            wrapper = maybe_new_wrapper

        # Existing tracking code
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # in GB

        # Convert elapsed time to hours, minutes, and seconds
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time  # in seconds
        hours, remainder = divmod(int(elapsed_time), 3600)
        minutes, seconds = divmod(remainder, 60)

        logger.info(f"Training time: {hours}h {minutes}m {seconds}s")
        logger.info(f"Peak GPU memory usage: {peak_memory:.2f} GB")

        if operator_config.use_wandb:
            wandb.log({
                "Training Time (s)": elapsed_time,
                "Training Time (h:m:s)": f"{hours}:{minutes:02}:{seconds:02}",
                "Peak GPU Memory (GB)": peak_memory
            }, commit=False)

        assert_model_params_valid(wrapper)
        torch.cuda.empty_cache()

        wandb.finish()

        # Save the model
        if operator_config.save_pipeline_after_operator:

            wrapper = merge_and_unload_all_adapters(wrapper)
            assert_model_params_valid(wrapper)
            wrapper.save_pipeline(run_save_path)

        # Save the operator and pipe configurations
        with open(os.path.join(run_save_path, "operator_config.json"), 'w') as f:
            json.dump(asdict(operator_config), f)

        with open(os.path.join(run_save_path, "pipe_config.json"), 'w') as f:
            json.dump(asdict(pipe_config), f)

        if operator_config.use_wandb:
            wandb.finish()

        del wrapper
        memory_cleanup()

    new_pipe_config = replace(pipe_config, model_name_or_path=run_save_path)
    assert new_pipe_config.model_name_or_path == run_save_path, f"Expected {run_save_path}, got {new_pipe_config.model_name_or_path}"
    return new_pipe_config


if __name__ == "__main__":

    # Load configuration
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--method", "-m", required=True, choices=SUPPORTED_METHODS)
    pre.add_argument("--model_type", "-t", required=True, choices=SUPPORTED_MODEL_TYPES)
    pre.add_argument("--model_dtype", "-d", default=None)
    pre.add_argument("--model_name_or_path", required=False)
    pre.add_argument("--adapter_mode", "-a", default=None)
    pre.add_argument("--subset_name", "-s", default=None)
    pre.add_argument("--use_wandb", "--wb", action='store_true', default=False)
    pre.add_argument("--run_id", "-p", default=None)
    args, remaining = pre.parse_known_args()
    
    if args.run_id is None:
        args.run_id = os.path.join("adhoc_runs", generate_random_run_id(args.method))

    pipe_config = PipelineConfig.default_for_model_type(model_type=args.model_type)

    if args.model_name_or_path is not None:
        print(f"Overriding model_name_or_path to: {args.model_name_or_path}.")
        pipe_config = replace(pipe_config, model_name_or_path=args.model_name_or_path)

    if args.model_dtype is not None:
        print(f"Overriding model_dtype to: {args.model_dtype}.")
        pipe_config = replace(pipe_config, model_dtype=args.model_dtype)

    # Get the right config class and operator
    operator_config, operator = get_config_and_operator(args.method)
    operator_config = replace(operator_config, use_wandb=args.use_wandb)

    if hasattr(operator_config, "adaptation_config") and args.adapter_mode is not None:
        adaptation_config = AdaptationConfig(adapter_mode=args.adapter_mode).default_for_model_type(model_type=args.model_type)

        if args.subset_name is not None:
            # Override default subset_name if provided
            adaptation_config = replace(adaptation_config, subset_name=args.subset_name)

        operator_config = replace(operator_config, adaptation_config=adaptation_config)

    # Now build a full parser that includes all of config’s fields
    parser = argparse.ArgumentParser(description=f"Train with operator={operator_config.method}, model={pipe_config.model_type}")
    add_args_from_dataclass(parser, operator_config)
    parsed, remaining = parser.parse_known_args(remaining)

    # Update our config instance
    for k, v in vars(parsed).items():
        if v is not None:
            setattr(operator_config, k, v)

    if remaining:
        print(f"Warning: Unrecognized unused arguments: {remaining}")

    run_operator(operator, operator_config, pipe_config, args.run_id)
