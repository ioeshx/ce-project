import argparse
import os
from dataclasses import replace

from gem.common.sampling import generate_and_save_images
from gem.configs import InferenceConfig, PipelineConfig

from gem.datasets import get_prompt_dataset
from gem.utils.logging import setup_logger

from gem.wrappers.factory import PipelineFactory

from gem.utils.seed import seed_everything
from gem.utils.cli import add_args_from_dataclass

from dotenv import load_dotenv
load_dotenv()


def build_inference_output_path(output_folder, dataset_name):
    return os.path.join(os.getenv("OUTPUT_DIR"), "images", output_folder, dataset_name)


def inference(inference_config, output_folder, dataset):
    seed_everything(inference_config.test_seed)

    image_folder = build_inference_output_path(output_folder, dataset.name)

    wrapper = PipelineFactory.wrap_pipeline(PipelineFactory.create_pipeline(inference_config.pipe_config)).to("cuda")
    wrapper.eval()
    print("Loaded the following pipe for inference:")
    print(inference_config.pipe_config)

    generate_and_save_images(dataset=dataset, wrapper=wrapper, inference_config=inference_config, image_folder=image_folder)

    return image_folder


def main():
    # Load and adapt configuration
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--model_type", "-t", required=True)
    base.add_argument("--model_name_or_path", "-p", required=False, default=None)
    base.add_argument("--dataset_name", "-d", required=False, default=None)
    base.add_argument("--guidance_scale", "-gs", required=False, type=float, default=None)
    base.add_argument("--prompts", nargs='+', default=["a cute beaver"])
    args, _ = base.parse_known_args()

    logger = setup_logger(name="inference", log_file=f"inference.log", level="INFO")

    pipe_config = PipelineConfig.default_for_model_type(model_type=args.model_type)

    if args.model_name_or_path is not None:
        pipe_config = replace(pipe_config, model_name_or_path=args.model_name_or_path)

    if args.guidance_scale is not None:
        pipe_config.inference_guidance_scale = args.guidance_scale

    # Create default config for this model type
    inference_config = InferenceConfig(pipe_config=pipe_config, dataset_name=args.dataset_name)

    # Parse model-specific command line arguments
    parser = argparse.ArgumentParser("Run inference")
    add_args_from_dataclass(parser, inference_config)
    parsed, _ = parser.parse_known_args()

    # Override config with command line arguments
    inference_config = replace(inference_config, **vars(parsed))
    for k, v in vars(parsed).items():
        assert getattr(inference_config, k) == v

    logger.info("Inference Configuration Loaded:")
    logger.info(inference_config)

    from pathlib import Path

    # Robust output_folder construction
    if inference_config.pipe_config.model_name_or_path is None:
        output_folder = inference_config.pipe_config.model_type
    else:
        path = Path(inference_config.pipe_config.model_name_or_path)
        try:
            # Find the index of the "models" directory
            models_index = path.parts.index("models")
            # Get everything after "models/"
            output_folder = Path(*path.parts[models_index + 1:]).as_posix()
        except ValueError:
            # If "models" is not in the path, fallback to using the name of the final directory or file
            output_folder = path.name

    # Call inference with the computed output folder
    dataset = get_prompt_dataset(args.dataset_name, prompts=args.prompts)
    inference(inference_config, output_folder=os.path.join('adhoc', output_folder), dataset=dataset)


if __name__ == "__main__":
    main()

