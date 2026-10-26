import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time
import torch
from diffusers import StableDiffusion3Pipeline

device = torch.device("cuda:0")
torch.cuda.synchronize(device)
torch.cuda.reset_peak_memory_stats(device)

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium", 
    torch_dtype=torch.bfloat16
)
pipe = pipe.to(device)

start = time.perf_counter()
image = pipe(
    "A capybara holding a sign that reads Hello World",
    num_inference_steps=40,
    guidance_scale=4.5,
).images[0]
torch.cuda.synchronize(device)
runTime = time.perf_counter() - start

peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3

print(f"推理耗时: {runTime:.2f} 秒")
print(f"峰值实际使用显存: {peak_allocated:.2f} GiB")
print(f"峰值 PyTorch 保留显存: {peak_reserved:.2f} GiB")


image.save("sd35medium_output.png")
