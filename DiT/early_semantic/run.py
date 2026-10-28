"""Sample with an anchor prompt first, then switch to the target prompt."""
from __future__ import annotations
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


import argparse
from itertools import product
import json
import math
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def encode_prompts(model, prompts):
    """Return token embeddings, pooled embeddings and FLUX text position IDs."""
    pipe, opts = model.pipe, model.config["sampling"]
    kwargs = dict(prompt=prompts, prompt_2=None, device=model.device,
                  num_images_per_prompt=1, max_sequence_length=opts["max_sequence_length"])
    if model.flux:
        return pipe.encode_prompt(**kwargs)
    tokens, _, pooled, _ = pipe.encode_prompt(
        **kwargs, prompt_3=None, do_classifier_free_guidance=False)
    return tokens, pooled, None


def decode_image(model, latent):
    """Decode either a clean estimate or the current noisy sampling state."""
    from diffusers.utils.torch_utils import get_module_execution_device

    pipe, opts = model.pipe, model.config["sampling"]
    if model.flux:
        latent = pipe._unpack_latents(latent, opts["height"], opts["width"], pipe.vae_scale_factor)
    latent = latent.to(device=get_module_execution_device(pipe.vae), dtype=pipe.vae.dtype)
    latent = latent / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    decoded = pipe.vae.decode(latent, return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")[0]


def sample_prompt(model, record, anchor_steps, output_dir, negative):
    """The only intervention is choosing the condition at each Euler step."""
    import numpy as np
    import torch

    pipe, opts = model.pipe, model.config["sampling"]
    tokens, pooled, text_ids = encode_prompts(
        model, [record["anchor_prompt"], record["target_prompt"]])
    channels = pipe.transformer.config.in_channels // (4 if model.flux else 1)
    shape = (channels, opts["height"] // pipe.vae_scale_factor, opts["width"] // pipe.vae_scale_factor)
    generator = torch.Generator("cpu").manual_seed(record["seed"])
    z = torch.randn((1, *shape), generator=generator, dtype=torch.float32).to(model.dtype)
    if model.flux:
        z = pipe._pack_latents(z, 1, *shape)
    latent_args = (1, channels, opts["height"], opts["width"], tokens.dtype, model.device, None, z)

    # Match the native pipeline's timestep schedule, including resolution shift.
    schedule = pipe.scheduler.config
    schedule_kwargs = {}
    if model.flux:
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

        z, image_ids = pipe.prepare_latents(*latent_args)
        length = z.shape[1]
        schedule_kwargs["sigmas"] = np.linspace(1, 1 / opts["steps"], opts["steps"])
    else:
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
            calculate_shift, retrieve_timesteps,
        )

        z = pipe.prepare_latents(*latent_args)
        patch = pipe.transformer.config.patch_size
        length = (z.shape[-2] // patch) * (z.shape[-1] // patch)
    if model.flux or schedule.get("use_dynamic_shifting"):
        schedule_kwargs["mu"] = calculate_shift(
            length, schedule.get("base_image_seq_len", 256), schedule.get("max_image_seq_len", 4096),
            schedule.get("base_shift", 0.5), schedule.get("max_shift", 1.15 if model.flux else 1.16))
    timesteps, _ = retrieve_timesteps(pipe.scheduler, opts["steps"], model.device, **schedule_kwargs)
    pipe.scheduler.set_begin_index(0)
    sigmas = pipe.scheduler.sigmas.cpu().tolist()
    time_values = timesteps.cpu().tolist()
    use_cfg = not model.flux and opts["guidance_scale"] > 1
    guidance = None
    if model.flux and pipe.transformer.config.guidance_embeds:
        guidance = torch.full((1,), opts["guidance_scale"], device=model.device, dtype=torch.float32)

    snapshots, frames = [], []
    for i, timestep in enumerate(timesteps):
        # Step indices count actual model evaluations, not a fraction of sigma.
        condition = 0 if i < anchor_steps else 1
        phase = "anchor" if condition == 0 else "target"
        current_tokens = tokens[condition:condition + 1]
        current_pooled = pooled[condition:condition + 1]
        model_input = z
        if use_cfg:
            current_tokens = torch.cat((negative[0], current_tokens))
            current_pooled = torch.cat((negative[1], current_pooled))
            model_input = torch.cat((z, z))

        kwargs = dict(hidden_states=model_input, timestep=timestep.expand(model_input.shape[0]),
                      encoder_hidden_states=current_tokens, pooled_projections=current_pooled,
                      joint_attention_kwargs={}, return_dict=False)
        if model.flux:
            kwargs.update(timestep=kwargs["timestep"].to(z.dtype) / 1000,
                          guidance=guidance, txt_ids=text_ids, img_ids=image_ids)
        velocity = pipe.transformer(**kwargs)[0]
        if use_cfg:
            v_uncond, v_cond = velocity.chunk(2)
            velocity = v_uncond + opts["guidance_scale"] * (v_cond - v_uncond)

        # Clean prediction uses the PRE-STEP latent and this step's velocity.
        clean = z.float() - sigmas[i] * velocity.float()
        # Euler: z_next = z + (sigma_next - sigma) * velocity.
        z = pipe.scheduler.step(velocity, timestep, z, return_dict=False)[0]
        snapshots.append((clean.cpu(), z.cpu()))
        stem = f"step_{i + 1:03d}_{phase}"
        frames.append(dict(step=i + 1, phase=phase, timestep=time_values[i],
                           sigma=sigmas[i], sigma_next=sigmas[i + 1],
                           clean_image=f"clean/{stem}_clean.png", latent_image=f"latent/{stem}_latent.png"))

    # Decode after sampling: VAE CPU-offload hooks must not move the transformer
    # off the GPU between every pair of denoising steps. Snapshots stay on CPU.
    for name in ("clean", "latent"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    for frame, (clean, latent) in zip(frames, snapshots):
        decode_image(model, clean).save(output_dir / frame["clean_image"])
        image = decode_image(model, latent)
        image.save(output_dir / frame["latent_image"])
    image.save(output_dir / "final.png")
    metadata = {**record, "anchor_steps": anchor_steps, "target_steps": len(frames) - anchor_steps,
                "effective_anchor_percent": 100 * anchor_steps / len(frames),
                "first_target_step": anchor_steps + 1 if anchor_steps < len(frames) else None,
                "frames": frames, "final_image": "final.png"}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    pipe.maybe_free_model_hooks()


def main(argv=None):
    from benchmark.model_profiles import profile

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", choices=("sd35_medium", "flux1_dev"), default="sd35_medium")
    parser.add_argument("--percent", type=float, default=10, help="Percentage of steps using the anchor prompt.")
    parser.add_argument("--concepts", type=Path, default=ROOT / "benchmark/configs/dit_target_anchor.json")
    parser.add_argument("--targets", nargs="+", help="Optional subset of target names, e.g. Snoopy 'Van Gogh'.")
    parser.add_argument("--prompt-count", type=int, default=10, help="Use the first N of the ten generated templates.")
    parser.add_argument("--template-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="Image seed for prompt 1; prompt N uses seed + N - 1.")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device", default="cuda:0")
    placement = parser.add_mutually_exclusive_group()
    placement.add_argument("--device-map", choices=("balanced",), help="Shard the model across visible GPUs.")
    placement.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Load cached weights only.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts and step allocation without loading a model.")
    args = parser.parse_args(argv)
    pairs = json.loads(args.concepts.read_text(encoding="utf-8"))
    config = profile(args.model)
    config.update(model_name=args.model, adapter={"kind": "original"})
    opts = config["sampling"]
    for key in ("steps", "height", "width", "guidance_scale", "dtype"):
        if getattr(args, key) is not None:
            opts[key] = getattr(args, key)
    opts.update(device=args.device, device_map=args.device_map,
                cpu_offload=args.cpu_offload, local_files_only=args.offline)

    # Validate user parameters once, before creating outputs or loading weights.
    if not 0 <= args.percent <= 100:
        parser.error("--percent must be between 0 and 100.")
    if opts["steps"] < 1 or not 1 <= args.prompt_count <= 10:
        parser.error("--steps must be positive; --prompt-count must be between 1 and 10.")
    if min(opts["height"], opts["width"]) < 16 or opts["height"] % 16 or opts["width"] % 16:
        parser.error("--height and --width must be positive multiples of 16.")
    if args.targets and set(args.targets) - pairs.keys():
        parser.error(f"Unknown targets: {sorted(set(args.targets) - pairs.keys())}")
    if args.targets:
        pairs = {target: pairs[target] for target in args.targets}

    scenes = ("in a quiet garden", "beside a lake", "on a city street", "in a bright studio",
              "against a mountain landscape", "in a sunlit room", "near a forest", "by the seaside")
    lights = ("soft daylight", "warm evening light", "diffuse overcast light", "gentle side lighting")
    views = ("a close view", "a wide composition", "a centered composition", "a low-angle view")
    contexts = random.Random(args.template_seed).sample(list(product(scenes, lights, views)), 10)
    templates = ["{concept}, " + ", ".join(context) + "." for context in contexts]
    anchor_steps = math.ceil(opts["steps"] * args.percent / 100)
    result_root = args.output_dir / f"{args.percent:g}"
    records = []
    for target, anchor in pairs.items():
        prefix = "An image of "
        if anchor == "Artist Style":
            prefix = "A painting in the style of "
        elif target == "nudity":
            prefix = "An image of an adult person, "
        pair_dir = re.sub(r"[^\w.-]+", "_", f"{target}_{anchor}").strip("_.")
        for prompt_no, template in enumerate(templates[:args.prompt_count], start=1):
            template = prefix + template
            records.append(dict(target=target, anchor=anchor, prompt_no=prompt_no,
                                template=template, target_prompt=template.format(concept=target),
                                anchor_prompt=template.format(concept=anchor), seed=args.seed + prompt_no - 1,
                                directory=f"{pair_dir}/{prompt_no:02d}"))

    print(f"{args.model}: {len(pairs)} pairs x {args.prompt_count} prompts = {len(records)} samples; "
          f"{anchor_steps} anchor + {opts['steps'] - anchor_steps} target steps", flush=True)
    print(f"Results: {result_root}", flush=True)
    if args.dry_run:
        print(json.dumps(records, ensure_ascii=False, indent=2))
        return 0

    import torch
    from benchmark.dit_adapter import DiTAdapter

    model = DiTAdapter(config)
    model.pipe.transformer.eval()
    result_root.mkdir(parents=True, exist_ok=True)
    experiment = dict(model=args.model, base_model=config["base_model"], sampling=model.config["sampling"],
                      requested_anchor_percent=args.percent, anchor_steps=anchor_steps,
                      template_seed=args.template_seed, templates=templates, concepts=pairs,
                      samples=records, frame_description={
                          "clean": "Decode(z_i - sigma_i * guided_velocity_i), before the Euler step.",
                          "latent": "Decode(z_{i+1}), after the Euler step; still noisy until sigma=0.",
                          "final": "Decode(z_N), identical to the last latent frame."})
    (result_root / "experiment.json").write_text(
        json.dumps(experiment, ensure_ascii=False, indent=2), encoding="utf-8")
    with torch.inference_mode():
        negative = encode_prompts(model, [""]) if not model.flux and opts["guidance_scale"] > 1 else None
        for number, record in enumerate(records, start=1):
            directory = result_root / record["directory"]
            directory.mkdir(parents=True, exist_ok=True)
            print(f"[{number}/{len(records)}] {record['target']} / prompt {record['prompt_no']:02d}", flush=True)
            sample_prompt(model, record, anchor_steps, directory, negative)
    model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
