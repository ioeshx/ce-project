import csv
import logging
import time

import torch
import os
from tqdm import tqdm

from gem.configs import InferenceConfig
from gem.datasets.prompt_datasets.adhoc_prompt_embed_dataset import AdhocPromptEmbedDataset
from gem.utils.memory import memory_cleanup
from gem.utils.slugify import slugify


def build_filename(prompt: str, num_prompts_total: int, idx: int, seed: int, sweep_idx: int, num_images_per_prompt: int = 1, ext: str = "png") -> str:
    total_seed = seed + sweep_idx
    slug = slugify(prompt) if prompt is not None else "none"
    return f"{idx:0{len(str(num_prompts_total-1))}d}_{total_seed:0{len(str(num_images_per_prompt-1))}d}_{slug}.{ext}"


@torch.no_grad()
def generate_and_save_images(dataset, wrapper, inference_config, image_folder):
    assert isinstance(inference_config, InferenceConfig)

    logger = logging.getLogger()
    logger.info(f"Generating images and saving them to: {image_folder}")

    os.makedirs(image_folder, exist_ok=True)

    wrapper.pipe.enable_attention_slicing()
    wrapper.pipe.enable_model_cpu_offload()
    wrapper.eval()

    assert wrapper.pipe.safety_checker is None

    # Extract prompts and seeds from dataset
    items = list(dataset)

    # Track pending generations and skipped files
    pending = []
    skipped = []

    # ==== Start time and memory tracking ====
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    for idx in range(len(items)):
        for sweep_idx in range(dataset.num_images_per_prompt):
            seed = items[idx].seed or (inference_config.test_seed + sweep_idx)
            fname = build_filename(items[idx].prompt, len(items), idx, seed, sweep_idx, dataset.num_images_per_prompt)
            fpath = os.path.join(image_folder, fname)

            if os.path.isfile(fpath):
                skipped.append((idx, sweep_idx, fname))
            else:
                pending.append((idx, sweep_idx, fname))

    # Log all skips
    if skipped:
        tqdm.write(f"⏭️Skipping {len(skipped)} already-existing images at {image_folder}:")
        for idx, sweep_idx, fname in skipped:
            tqdm.write(f"    - idx={idx}, sweep={sweep_idx} → {fname}")

    if not pending:
        tqdm.write("🎉 All requested images already exist. Nothing to do.")
        return

    # Batch generation only for pending items
    batches = [pending[i:i + inference_config.batch_size] for i in range(0, len(pending), inference_config.batch_size)]

    memory_cleanup()

    # Generate images in batches
    index_save_path = os.path.join(image_folder, 'index.csv')
    # Create the CSV header only once if the file doesn't exist
    if not os.path.exists(index_save_path):
        with open(index_save_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['idx', 'sweep_idx', 'filename', 'prompt'])

    for batch in tqdm(batches, desc="Generating Batches"):
        batch_prompts = [items[idx].prompt for idx, _, _ in batch]
        batch_seeds = [items[idx].seed if items[idx].seed is not None else inference_config.test_seed for idx, _, _ in batch]
        sweep_idxs = [sweep_idx for _, sweep_idx, _ in batch]
        generators = [torch.Generator().manual_seed(seed + sweep_idx) for seed, sweep_idx in zip(batch_seeds, sweep_idxs)]

        optional_embeds_dict = {}
        if isinstance(dataset, AdhocPromptEmbedDataset):
            batch_prompts = None
            optional_embeds_dict_list = [wrapper.create_optional_embeds_dict(item.prompt_embeds) for item in items for _ in range(len(batch_seeds))]
            for key in optional_embeds_dict_list[0].keys():
                optional_embeds_dict[key] = torch.cat([d[key] for d in optional_embeds_dict_list], dim=0) if optional_embeds_dict_list[0][key] is not None else None

        result = wrapper(
            prompt=batch_prompts,
            **optional_embeds_dict,
            num_inference_steps=inference_config.pipe_config.num_inference_steps,
            guidance_scale=inference_config.pipe_config.inference_guidance_scale,
            height=inference_config.pipe_config.image_size,
            width=inference_config.pipe_config.image_size,
            generator=generators
        )
        for img, (idx, sweep_idx, fname) in zip(result.images, batch):
            filepath = os.path.join(image_folder, fname)
            img.save(filepath)

            #  Save prompt to CSV
            prompt_text = items[idx].prompt if batch_prompts is not None else '[embedded]'
            with open(index_save_path, mode='a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([idx, sweep_idx, fname, prompt_text])

            tqdm.write(f"✅ Generated & saved: idx={idx}, sweep={sweep_idx} → {fname}, folder: {image_folder}")

        del result

    end_time = time.perf_counter()
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # in GB
    elapsed_time = end_time - start_time  # in seconds

    logger.info(f"Sampling time: {elapsed_time:.2f} seconds")
    logger.info(f"Peak GPU memory usage: {peak_memory:.2f} GB")

    # ==== End time and memory tracking ====