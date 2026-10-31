import re

from gem.datasets.prompt_dataset import PromptItem, PromptDataset
from gem.datasets.prompt_datasets.adhoc_prompt_dataset import AdhocPromptDataset
from gem.datasets.prompt_datasets.coco_30k_dataset import COCOC30KPromptDataset
from gem.datasets.prompt_datasets.csv_prompt_dataset import CSVPromptDataset
from gem.datasets.prompt_datasets.file_prompt_dataset import FilePromptDataset
from gem.datasets.prompt_datasets.mjhq_30k_dataset import MJHQPromptDataset
from gem.datasets.prompt_datasets.mma_nudity_prompt_dataset import MMANudityPromptDataset
from gem.datasets.prompt_datasets.p4d_prompt_dataset import P4DNudityPromptDataset
from gem.datasets.prompt_datasets.rab_nudity_prompt_dataset import RABNudityPromptDataset
from gem.datasets.unlabeled_image_dataset import UnlabeledImageFolderDataset
from ..common import TemplatePromptAugmentation
from ..common.templates.templates_basic import BASIC_SUBJECT_TEMPLATES, BASIC_GENERAL_TEMPLATES


def get_prompt_dataset(dataset_name=None, prompts=None):
    dataset_name = dataset_name or "none"

    # 1. Extract max_samples if present (e.g., "name_n100" -> samples=100, base="name")
    max_samples = None
    base_name = dataset_name
    if "_n" in dataset_name and dataset_name[-1].isdigit():
        parts = dataset_name.rsplit("_n", 1)
        base_name = parts[0]
        max_samples = int(parts[1])

    # 2. Simple Class Mappings
    simple_map = {
        "rab_nudity": RABNudityPromptDataset,
        "mma_nudity": MMANudityPromptDataset,
        "p4d_nudity": P4DNudityPromptDataset,
        "coco_30k": COCOC30KPromptDataset,
        "mjhq_30k": MJHQPromptDataset,
    }

    if base_name in simple_map:
        cls = simple_map[base_name]
        return cls(max_samples=max_samples) if max_samples else cls()

    # 3. CSV Datasets
    csv_configs = {
        "i2p_nudity": "i2p/i2p_nudity.csv",
        "i2p_udiff_subset": "i2p/i2p_unlearndiff_subset.csv",
        "t2i-rp_pornography": "t2i-rp/t2i-rp_pornography.csv",
        "t2i-rp_bloody_gore": "t2i-rp/t2i-rp_disturbing_content_blood.csv"
    }

    if base_name in csv_configs:
        return CSVPromptDataset(filepath=f"gem/datasets/prompt_datasets/data/{csv_configs[base_name]}",
            name=dataset_name, **({"max_samples": max_samples} if max_samples else {}))

    # 4. Dynamic Templated Datasets
    template_configs = {
        "basic_subject": (BASIC_SUBJECT_TEMPLATES, 10),
        "basic_general": (BASIC_GENERAL_TEMPLATES, 10)
    }

    for prefix, config in template_configs.items():
        if dataset_name.startswith(prefix):
            templates, img_per_prompt = config[0], config[1]

            match = re.search(r"<([^>]+)>", dataset_name)
            target = match.group(1) if match else None

            aug_prompts = TemplatePromptAugmentation(templates=templates).apply(target, randomize=False,
                n_templates=len(templates))

            kwargs = {"name": dataset_name, "seed": 0, "prompts": aug_prompts}
            if img_per_prompt:
                kwargs["num_images_per_prompt"] = img_per_prompt
            return AdhocPromptDataset(**kwargs)

    # 5. Fallback
    assert prompts is not None, "Prompts must be provided for AdhocPromptDataset"
    return AdhocPromptDataset(prompts=prompts)


if __name__ == "__main__":

    dataset = get_prompt_dataset(dataset_name="basic_large_person<Albert Einstein>")
    for item in dataset:
        print(item.prompt, item.seed)
