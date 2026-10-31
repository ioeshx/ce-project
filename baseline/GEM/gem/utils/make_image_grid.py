import argparse
import os
import platform

from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid
import torchvision.transforms as T
import torch
from pathlib import Path
from dotenv import load_dotenv

from gem.utils.make_eval_table import sort_groups

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Create image grid using torchvision")
    parser.add_argument('--exp_name', '-e', required=True, help='Experiment name')
    parser.add_argument('--base_model_name', '-b', required=True, help='Base model name')
    parser.add_argument('--prompt_indices', '-i', type=int, nargs='+', required=True, help='Prompt indices to include as columns')
    parser.add_argument('--dataset', '-d', required=True, help='Single dataset to use')
    parser.add_argument('--censored', '-c', action='store_true', help='Use censored version')
    parser.add_argument('--output', '-o', default='grid_output.png', help='Output grid image filename')
    parser.add_argument('--label_models', action='store_true', help='Render model names on the first column')
    parser.add_argument('--exclude', required=False, type=str, default=None, help='List of keywords when found in a folder path exclude it')
    parser.add_argument('--filter', required=False, type=str, default=None, help='List of keywords that we filter for')
    parser.add_argument('-k', required=False, default=1, help='Number of supercolumns')
    return parser.parse_args()


def get_image_tensor(img_path):
    img = Image.open(img_path).convert("RGB")
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])
    return transform(img)


def get_labelled_image(tensor_img, label, font_size=20):
    pil_img = T.ToPILImage()(tensor_img)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    draw.text((5, 5), label, fill="lime", font=font)
    return T.ToTensor()(pil_img)


def find_image(path, prompt_idx):
    if not os.path.isdir(path):
        print("Not a directory:", path)
        return None
    fnames = sorted(os.listdir(path))  # sort to ensure consistent indexing
    if prompt_idx >= len(fnames):
        print(f"Prompt index {prompt_idx} out of range in {path}")
        return None
    print("Loading image from: ", fnames[prompt_idx])
    return get_image_tensor(os.path.join(path, fnames[prompt_idx]))


def find_runs(exp_path):
    runs = []
    for base_model_dir in os.listdir(exp_path):
        model_path = os.path.join(exp_path, base_model_dir)
        if os.path.isdir(model_path):
            for run_id in os.listdir(model_path):
                run_path = os.path.join(model_path, run_id)
                if os.path.isdir(run_path):
                    runs.append(run_id)

    runs = sort_groups(runs)
    return runs


def get_image_from_path(base_path, exp_name, run_id, base_model_str, dataset_folder, idx, prompt_idx, label_models, interpret_model_run_ids_absolute):
    if run_id is None:
        img_dir = base_path / 'original' / base_model_str / dataset_folder
    else:
        if not interpret_model_run_ids_absolute:
            img_dir = base_path / exp_name / base_model_str / run_id / dataset_folder
        else:
            img_dir = base_path / run_id / dataset_folder

    img_tensor = find_image(img_dir, prompt_idx)

    return img_tensor


def get_current_box_color(label, box_color_dict, default_color=(30, 30, 30)):

    if not box_color_dict:
        return default_color

    for k, v in box_color_dict.items():
        if str(k).lower() in str(label).lower():
            return v
    else:
        return default_color


def add_grid_labels(grid_pil, row_labels, col_labels, cell_size=256, padding=2, col_margins=None, box_color_dict=None):

    def get_bold_font(size):
        """Helper to find a bold font across different OS."""
        system = platform.system()
        if system == "Windows":
            paths = ["C:\\Windows\\Fonts\\arialbd.ttf", "C:\\Windows\\Fonts\\SegoeUIb.ttf"]
        elif system == "Darwin":  # macOS
            paths = ["/Library/Fonts/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"]
        else:  # Linux
            paths = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                     "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]

        for path in paths:
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
        return ImageFont.load_default()

    font_size = 28
    font = get_bold_font(font_size)

    col_margins = col_margins or []

    text_color = (255, 255, 255)

    box_margin = 3
    label_margin_top = 60
    label_margin_left = 60

    new_img = Image.new("RGB", (grid_pil.width + label_margin_left,
                                grid_pil.height + label_margin_top), (255, 255, 255))
    new_img.paste(grid_pil, (label_margin_left, label_margin_top))
    draw = ImageDraw.Draw(new_img)

    # --- Column Headers ---
    for i, label in enumerate(col_labels):
        if label is None:
            continue
        # Precise center calculation accounting for padding
        x_start = label_margin_left + sum(col_margins[:i+1]) + padding // 2 + (i * (cell_size + padding))
        x_end = x_start + cell_size
        draw.rectangle([x_start, box_margin, x_end, label_margin_top - box_margin], fill=get_current_box_color(label, box_color_dict))

        bbox = draw.textbbox((0, 0), str(label), font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x_start + (cell_size // 2) - (tw // 2),
                   (label_margin_top // 2) - (th // 2)), str(label), fill=text_color, font=font)

    # --- Row Headers (Fixed Rotation & Clipping) ---
    for i, label in enumerate(row_labels):
        if label is None:
            continue
        y_start = label_margin_top + padding + (i * (cell_size + padding))
        y_end = y_start + cell_size

        # Color logic for "Ours"
        current_box_color = get_current_box_color(label, box_color_dict)
        draw.rectangle([box_margin, y_start, label_margin_left - box_margin, y_end], fill=current_box_color)

        # 1. Measure text
        text_str = str(label)
        bbox = draw.textbbox((0, 0), text_str, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # 2. CREATE OVERSIZED CANVAS
        # Using a large safe_pad ensures no part of the bold font is cut off before rotation
        safe_pad = 40
        txt_canvas = Image.new("RGBA", (tw + safe_pad * 2, th + safe_pad * 2), (0, 0, 0, 0))
        d_txt = ImageDraw.Draw(txt_canvas)
        d_txt.text((safe_pad, safe_pad), text_str, font=font, fill=text_color)

        # 3. ROTATE
        rotated_txt = txt_canvas.rotate(90, expand=True, resample=Image.BICUBIC)

        # 4. AUTO-CROP
        # This removes the safe_pad and trims the image to the exact pixels of the text
        content_bbox = rotated_txt.getbbox()
        if content_bbox:
            rotated_txt = rotated_txt.crop(content_bbox)

        # 5. PASTE WITH CENTERING
        rw, rh = rotated_txt.size
        paste_x = (label_margin_left // 2) - (rw // 2)
        paste_y = y_start + (cell_size // 2) - (rh // 2)

        new_img.paste(rotated_txt, (paste_x, paste_y), rotated_txt)

    return new_img


def create_grid(base_model_str, exp_name, datasets_and_prompt_indices, output_file, row_labels=None, col_labels=None, output_folder=None,
            exclude_keywords=None, filter_keywords=None, censored=False, label_models=False, n_supercolumns=2,
            model_run_ids=None, interpret_model_run_ids_absolute=False, transpose=False, box_color_dict=None):

    print("Creating grid with the following args:")
    # --- The Printout Section ---
    print("=" * 60)
    print(f"{'GRID GENERATION CONFIGURATION':^60}")
    print("=" * 60)

    if transpose:
        col_labels, row_labels = row_labels, col_labels

    datasets = [d for d, p in datasets_and_prompt_indices.items() for _ in range(len(p))]
    prompt_indices = [idx for p in datasets_and_prompt_indices.values() for idx in p]

    config = {
        "Base Model": base_model_str,
        "Experiment": exp_name,
        "Datasets": ', '.join(datasets),
        "Output Folder": f"{output_folder or './'}/{output_file}",
        "Supercolumns": n_supercolumns,
        "Censored Mode": "ON" if censored else "OFF",
        "Label Models": label_models,
        "Filter Kws": filter_keywords or "None",
        "Exclude Kws": exclude_keywords or "None",
        "Run IDs": model_run_ids or "All",
        "Transpose": "ON" if transpose else "OFF"
    }

    for key, value in config.items():
        print(f"  {key:<20} : {value}")

    print("=" * 60)
    # --- End Printout Section ---

    base_path = Path(f"{os.getenv('OUTPUT_DIR')}/images")
    models = [None] + (find_runs(base_path / exp_name) if model_run_ids is None else model_run_ids)

    original_rows = []
    method_rows = []

    if isinstance(datasets, str):
        datasets = [datasets] * len(prompt_indices)

    assert len(datasets) == len(prompt_indices), f"Length of datasets ({len(datasets)}) must match length of prompt_indices ({len(prompt_indices)})"

    for run_id in models:
        row_images = []

        if run_id and exclude_keywords and any(e in run_id for e in exclude_keywords):
            continue

        if run_id and filter_keywords and any(f not in run_id for f in filter_keywords):
            continue

        print(run_id)

        for idx, prompt_idx in enumerate(prompt_indices):

            dataset_folder_censored = f"{datasets[idx]}_censored"
            dataset_folder_uncensored = datasets[idx]

            img_tensor = None

            # First try to find censored one if required
            if censored:
                img_tensor = get_image_from_path(base_path, exp_name, run_id, base_model_str, dataset_folder_censored, idx, prompt_idx, label_models, interpret_model_run_ids_absolute)

            # Try again to find uncensored one
            if not censored or img_tensor is None:
                img_tensor = get_image_from_path(base_path, exp_name, run_id, base_model_str, dataset_folder_uncensored, idx, prompt_idx, label_models, interpret_model_run_ids_absolute)

            if img_tensor is None:
                img_tensor = torch.ones(3, 256, 256)  # white placeholder

            if label_models and idx == 0:
                label = base_model_str if run_id is None else f"{base_model_str}/{run_id}"
                img_tensor = get_labelled_image(img_tensor, label)

            row_images.append(img_tensor)

        if run_id is None:
            original_rows.extend(row_images)
        else:
            method_rows.extend(row_images)

    num_cols = len(prompt_indices)

    # Group the flat original/method lists back into their respective rows
    grid_2d = []

    # Add the "Original" row(s)
    for i in range(0, len(original_rows), num_cols):
        grid_2d.append(original_rows[i: i + num_cols])

    # Add the "Method" rows
    for i in range(0, len(method_rows), num_cols):
        grid_2d.append(method_rows[i: i + num_cols])

    # 2. Handle Transpose
    if transpose:
        # zip(*grid_2d) turns [Rows x Cols] into [Cols x Rows]
        grid_2d = [list(row) for row in zip(*grid_2d)]
        # Update nrow for make_grid: it should now be the number of models
        nrow = n_supercolumns * len(models)
    else:
        # Standard: nrow is the number of prompts
        nrow = n_supercolumns * num_cols

    # 3. Flatten for make_grid
    flat_grid = [img for row in grid_2d for img in row]

    grid = make_grid(flat_grid, nrow=nrow, padding=2, pad_value=1.0)

    if output_file is not None:
        output_folder = output_folder or os.path.join(os.getenv("OUTPUT_DIR"), "grids", exp_name)
        output_path = os.path.join(output_folder, output_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        pil_image = T.ToPILImage()(grid)
        pil_image = add_grid_labels(pil_image, row_labels or [], col_labels or [], box_color_dict=box_color_dict)
        pil_image.save(output_path)

        if not output_path.endswith(".pdf"):
            pil_image.save(output_path.split(".")[0] + ".pdf")
        print(f"Saved grid to {output_path}")

    return grid


if __name__ == "__main__":
    args = parse_args()

    exclude_keywords = [e.strip() for e in args.exclude.split(",")] if args.exclude else None
    print("Excluding keywords:", exclude_keywords)

    filter_keywords = [f.strip() for f in args.filter.split(",")] if args.filter else None
    print("Filtering keywords:", filter_keywords)

    create_grid(
        base_model_str=args.base_model_name,
        exp_name=args.exp_name,
        exclude_keywords=exclude_keywords,
        filter_keywords=filter_keywords,
        prompt_indices=args.prompt_indices,
        datasets=args.dataset,
        censored=args.censored,
        output_file=args.output,
        label_models=args.label_models,
        n_supercolumns=args.k
    )


def join_grids_with_margin(grid1, grid2, margin_width=10, vertical=False):
    # Get dimensions (C, H, W)
    channels, height, width = grid1.shape

    # Determine fill value based on data type
    fill_value = 1.0 if grid1.is_floating_point() else 255

    if vertical:
        # Create spacer: width matches grid1, height is the margin
        spacer = torch.full((channels, margin_width, width), fill_value, dtype=grid1.dtype)
        # Concatenate along height (dim 1)
        combined = torch.cat((grid1, spacer, grid2), dim=1)
    else:
        # Create spacer: height matches grid1, width is the margin
        spacer = torch.full((channels, height, margin_width), fill_value, dtype=grid1.dtype)
        # Concatenate along width (dim 2)
        combined = torch.cat((grid1, spacer, grid2), dim=2)

    return combined