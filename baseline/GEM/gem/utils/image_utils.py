import math
from io import BytesIO
from typing import Union

import torch

from PIL import Image


def create_image_grid(images, spacing=10, cols=None, background_color=(255, 255, 255)):
    # Normalize and convert images to PIL format
    if isinstance(images, torch.Tensor):
        images = images.to("cpu")
        images = ((images / 2) + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        pil_images = [Image.fromarray((img * 255).astype("uint8")) for img in images]
    else:
        pil_images = images  # Assume it is a list of lists of PIL images

    n_images = len(pil_images)

    cols = int(math.sqrt(n_images)) if cols is None else cols
    rows = math.ceil(n_images / cols)

    img_width, img_height = pil_images[0].size
    grid_width = cols * img_width + (cols - 1) * spacing
    grid_height = rows * img_height + (rows - 1) * spacing

    # Create the grid image with the specified background color
    grid_image = Image.new("RGB", (grid_width, grid_height), background_color)

    # Paste each image into the grid
    for idx, img in enumerate(pil_images):
        x = (idx % cols) * (img_width + spacing)
        y = (idx // cols) * (img_height + spacing)
        grid_image.paste(img, (x, y))

    return grid_image


def load_image_from_source(image_source: str, allow_web_scraping: bool = False) -> Union[torch.Tensor, Image.Image]:
    if image_source.startswith(("http://", "https://")):
        assert allow_web_scraping, "Web scraping is not allowed in this context."
        response = requests.get(image_source, timeout=5)
        image_source = BytesIO(response.content)
    return Image.open(image_source).convert("RGB")
