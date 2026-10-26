import os
import time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel

# flux 1.dev
# 512*512, step=28, max sequence length=256
# Inference time: 10.04 seconds
# GPU 0 (NVIDIA GeForce RTX 3090): peak allocated = 18.82 GiB, peak reserved = 19.36 GiB
# GPU 1 (NVIDIA GeForce RTX 3090): peak allocated = 13.36 GiB, peak reserved = 13.49 GiB

# 1024*1024
# Inference time: 33.51 seconds
# GPU 0 (NVIDIA GeForce RTX 3090): peak allocated = 20.60 GiB, peak reserved = 22.81 GiB
# GPU 1 (NVIDIA GeForce RTX 3090): peak allocated = 13.66 GiB, peak reserved = 14.20 GiB

model_id = "black-forest-labs/FLUX.1-dev"

# Reserve more room on GPU 0 for the text encoders and VAE.
transformer = FluxTransformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={
        0: "16GiB",
        1: "16GiB",
    },
)

pipe = FluxPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)

# Move only the remaining components.
pipe.text_encoder.to("cuda:0")
pipe.text_encoder_2.to("cuda:0")
pipe.vae.to("cuda:0")
pipe.vae.enable_tiling()

# Print module placement.
for module_name, device in transformer.hf_device_map.items():
    print(f"{module_name} -> {device}")

devices = [0, 1]
GiB = 1024 ** 3

# Wait for model loading and reset peak counters on both GPUs.
for device in devices:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

with torch.inference_mode():
    start = time.perf_counter()

    image = pipe(
        prompt="A beautiful scene",
        height=512,
        width=512,
        guidance_scale=3.5,
        num_inference_steps=28,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(42),
    ).images[0]

    # CUDA operations are asynchronous; wait before stopping the timer.
    for device in devices:
        torch.cuda.synchronize(device)

    total_time = time.perf_counter() - start

# Record and print per-GPU memory peaks.
memory_stats = {}

print(f"\nInference time: {total_time:.2f} seconds")

for device in devices:
    peak_allocated = torch.cuda.max_memory_allocated(device) / GiB
    peak_reserved = torch.cuda.max_memory_reserved(device) / GiB

    memory_stats[device] = {
        "peak_allocated_GiB": peak_allocated,
        "peak_reserved_GiB": peak_reserved,
    }

    print(
        f"GPU {device} ({torch.cuda.get_device_name(device)}): "
        f"peak allocated = {peak_allocated:.2f} GiB, "
        f"peak reserved = {peak_reserved:.2f} GiB"
    )

image.save("flux_2.png")