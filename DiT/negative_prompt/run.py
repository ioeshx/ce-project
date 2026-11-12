"""Generate and evaluate four DiT benchmark groups using negative prompts."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.common import ROOT as BENCHMARK_ROOT, SCHEMA, concept_key, configuration, write_json
from benchmark.data import prompt_records
from benchmark.dataset.prompt_templates import template_dict
from benchmark.dit_cases import definitions, select_cases
from benchmark.layout import timestamp, unique_directory
from benchmark.model_profiles import DIT_MODELS, profile
from benchmark.sampling_profile import canonical_protocol


def run_experiments(run_root, config, experiments, args):
    """Generate missing images, release the diffusion model, then run benchmark metrics."""
    model = None
    for experiment in experiments:
        case_root = run_root / experiment["case_id"]
        records = experiment["records"]
        if experiment["task"] != "i2p":
            for record in records:
                record["reference"] = f"../references/{experiment['task']}/{record['sample_id']}.png"
        if args.phase == "evaluate":
            continue

        experiment["sampling"] = {**config["sampling"], "negative_prompt": experiment["negative_prompt"]}
        case_root.mkdir(parents=True, exist_ok=True)
        write_json(case_root / "experiment.json", experiment)
        for field in (["image"] if experiment["task"] == "i2p" else ["image", "reference"]):
            missing = [record for record in records if not (case_root / record[field]).is_file()]
            print(f"[{experiment['case_id']}:{field}] {len(missing)} missing, {len(records) - len(missing)} reused", flush=True)
            if not missing:
                continue
            if model is None:
                from benchmark.dit_adapter import DiTAdapter
                model = DiTAdapter(config)
                model.pipe.transformer.eval()
                config["sampling"] = model.config["sampling"]
                info = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
                info.update(sampling=model.config["sampling"], runtime=model.runtime_info)
                write_json(run_root / "run.json", info)
            experiment["sampling"] = {**config["sampling"], "negative_prompt": experiment["negative_prompt"]}
            write_json(case_root / "experiment.json", experiment)
            # The reference differs ONLY in its empty negative prompt. In particular,
            # FLUX keeps the same true_cfg_scale, rather than reverting to distilled CFG.
            sampling = {**experiment["sampling"], "negative_prompt": experiment["negative_prompt"] if field == "image" else ""}
            (case_root / missing[0][field]).parent.mkdir(parents=True, exist_ok=True)
            for start in range(0, len(missing), args.batch_size):
                batch = missing[start:start + args.batch_size]
                images = model.generate(batch, sampling)
                for record, image in zip(batch, images, strict=True):
                    image.save(case_root / record[field])
                print(f"[{experiment['case_id']}:{field}] {start + len(batch)}/{len(missing)}", flush=True)
    if model is not None:
        model.close()
    if args.phase == "sample":
        return

    from benchmark.evaluate import evaluate

    evaluation = dict(device=args.metric_device or args.device, batch_size=args.metric_batch_size,
                      local_files_only=args.offline, q16_enabled=args.with_q16)
    for key in ("clip_model", "vlm_model", "vlm_device", "nudenet_weights", "resnet_weights_path", "q16_prompts", "q16_clip_path"):
        value = getattr(args, key)
        if value is not None:
            local = Path(value).expanduser()
            evaluation[key] = str(local.resolve()) if local.exists() else value
    for key, name in (("vlm_model", "Qwen3-VL-2B-Instruct"), ("resnet_weights_path", "resnet50_v2.pth")):
        local = BENCHMARK_ROOT / "weights" / name
        if key not in evaluation and local.exists():
            evaluation[key] = str(local)
    for experiment in experiments:
        case_root = run_root / experiment["case_id"]
        sampling = experiment["sampling"]
        cfg = configuration(task=experiment["task"], base={
            "case_id": experiment["case_id"], "model_name": config["model_name"], "base_model": config["base_model"],
            "method": "NegativePrompt", "adapter": {"kind": "existing"}, "targets": experiment["targets"],
            "sampling": {**sampling, **{key: config["sampling"][key] for key in
                         ("device", "device_map", "cpu_offload", "max_memory", "local_files_only", "batch_size")}},
            "dataset": experiment["dataset"]["options"], "cache_dir": str(args.cache_dir.expanduser().resolve()),
            "output_dir": str(run_root / "metrics" / config["model_name"]),
            "evaluation": {**evaluation, "vlm_enabled": experiment["group"] in {"ip", "style"} and not args.skip_vlm}})
        records = [{**record, **{field: str((case_root / record[field]).resolve())
                    for field in ("image", "reference") if field in record}} for record in experiment["records"]]
        manifest = dict(schema=SCHEMA, task=experiment["task"], case_id=experiment["case_id"],
                        method=cfg["method"], model_name=cfg["model_name"], base_model=cfg["base_model"],
                        targets=experiment["targets"], sampling=canonical_protocol(sampling),
                        dataset=experiment["dataset"], config=cfg, records=records,
                        adapter_identity={"kind": "negative_prompt", "negative_prompt": experiment["negative_prompt"]})
        if experiment["task"] != "i2p":
            manifest["reference_sampling"] = canonical_protocol({**sampling, "negative_prompt": ""})
        path = case_root / "manifest.json"
        write_json(path, manifest, portable=True)
        print(f"[evaluate:{experiment['case_id']}] {path}", flush=True)
        row = evaluate(path)
        print(f"[metrics:{experiment['case_id']}] {row['details_dir']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", choices=DIT_MODELS, default="sd35_medium")
    parser.add_argument("--cases", default="all", help="all/i2p/ip/style/object or comma-separated case IDs/groups")
    parser.add_argument("--phase", choices=("all", "sample", "evaluate"), default="all",
                        help="all: generate then evaluate; sample: images only; evaluate: metrics only")
    parser.add_argument("--run-dir", type=Path, help="Existing timestamped run; reuse its prompts, seeds and sampling settings")
    parser.add_argument("--i2p-prompt-count", type=int, help="First N prompts after selecting the I2P subset; default: all")
    parser.add_argument("--i2p-subset", choices=("sexual", "all"), default="sexual")
    parser.add_argument("--ip-template-count", type=int, default=80, help="First N templates per character (1-80)")
    parser.add_argument("--style-template-count", type=int, default=30, help="First N templates per artist (1-30)")
    parser.add_argument("--object-template-count", type=int, default=1,
                        help="First N templates per object: benchmark prompt, then 18 existing object templates (1-19)")
    parser.add_argument("--seeds", nargs="+", type=int, help="Override IP/style/object seeds; I2P keeps its CSV seeds")
    parser.add_argument("--base-model", help="Local Diffusers directory or Hub model ID")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--guidance-scale", type=float, help="SD3 CFG scale; FLUX embedded guidance")
    parser.add_argument("--true-cfg-scale", type=float, help="FLUX true CFG scale, must be > 1; default: 2")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    placement = parser.add_mutually_exclusive_group()
    placement.add_argument("--device-map", choices=("balanced", "auto"))
    placement.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--max-memory", type=json.loads, help='Transformer memory budget, e.g. \'{"0":"20GiB"}\'')
    parser.add_argument("--gpu-memory-reserve", type=float, default=2.0)
    parser.add_argument("--metric-device", help="Metric device; default: --device")
    parser.add_argument("--metric-batch-size", type=int, default=32)
    parser.add_argument("--cache-dir", type=Path, default=BENCHMARK_ROOT / "cache")
    parser.add_argument("--skip-vlm", action="store_true", help="IP/style: compute CS/FID without VLM classification")
    parser.add_argument("--with-q16", action="store_true", help="I2P: also compute Q16; new runs use the full I2P dataset")
    parser.add_argument("--clip-model", help="CLIP model ID or local directory")
    parser.add_argument("--vlm-model", help="Qwen3-VL model ID or local directory")
    parser.add_argument("--vlm-device", help="VLM device; default: --metric-device")
    parser.add_argument("--nudenet-weights", help="Local NudeNet ONNX weights")
    parser.add_argument("--resnet-weights-path", help="Local official ResNet-50 V2 weights")
    parser.add_argument("--q16-prompts", help="Official Q16 prompts.p")
    parser.add_argument("--q16-clip-path", help="Local OpenAI CLIP weights for Q16")
    parser.add_argument("--offline", action="store_true", help="Only load locally cached weights")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--dry-run", action="store_true", help="Preview counts and prompts without loading weights or writing files")
    args = parser.parse_args(argv)

    if args.phase == "evaluate" and args.run_dir is None:
        parser.error("--phase evaluate requires --run-dir pointing to an existing run.")
    if args.run_dir and any(getattr(args, key) is not None for key in
                            ("base_model", "steps", "height", "width", "guidance_scale", "true_cfg_scale", "dtype", "seeds", "i2p_prompt_count")):
        parser.error("--run-dir uses saved prompts, seeds and sampling settings; omit generation overrides.")
    run_root = args.run_dir.expanduser().resolve() if args.run_dir else None
    info = json.loads((run_root / "run.json").read_text(encoding="utf-8")) if run_root else None
    if info:
        args.model = info["model"]
    config = profile(args.model)
    config.update(model_name=args.model, adapter={"kind": "original"})
    if info:
        config.update(base_model=info["base_model"], sampling=info["sampling"])
    if args.base_model:
        local = Path(args.base_model).expanduser()
        config["base_model"] = str(local.resolve()) if local.exists() else args.base_model
    sampling = config["sampling"]
    for key in ("steps", "height", "width", "guidance_scale", "dtype"):
        if getattr(args, key) is not None:
            sampling[key] = getattr(args, key)
    sampling.update(device=args.device, batch_size=args.batch_size, device_map=args.device_map,
                    cpu_offload=args.cpu_offload, max_memory=args.max_memory,
                    gpu_memory_reserve_gib=args.gpu_memory_reserve, local_files_only=args.offline,
                    execution_backend=args.device.split(":", 1)[0])
    flux = args.model.startswith("flux1_")
    if flux and not info:
        sampling["true_cfg_scale"] = args.true_cfg_scale if args.true_cfg_scale is not None else 2.0

    # Check CLI values once, before expanding data or loading the model.
    object_templates = ["a photo of {}", *template_dict["object"]]
    for group, templates in (("ip", template_dict["instance"]), ("style", template_dict["style"]),
                             ("object", object_templates)):
        count = getattr(args, f"{group}_template_count")
        if not 1 <= count <= len(templates):
            parser.error(f"--{group}-template-count must be between 1 and {len(templates)}.")
    if args.i2p_prompt_count is not None and args.i2p_prompt_count < 1:
        parser.error("--i2p-prompt-count must be positive.")
    if args.seeds is not None and (len(args.seeds) != len(set(args.seeds)) or
                                   any(not 0 <= seed < 2**63 for seed in args.seeds)):
        parser.error("--seeds must contain distinct integers in [0, 2**63).")
    if min(sampling["steps"], args.batch_size, args.metric_batch_size) < 1:
        parser.error("--steps, --batch-size and --metric-batch-size must be positive.")
    if min(sampling["height"], sampling["width"]) < 16 or sampling["height"] % 16 or sampling["width"] % 16:
        parser.error("--height and --width must be positive multiples of 16.")
    if flux and (not math.isfinite(sampling["true_cfg_scale"]) or sampling["true_cfg_scale"] <= 1):
        parser.error("FLUX needs --true-cfg-scale > 1 for negative_prompt to participate in CFG.")
    if not math.isfinite(sampling["guidance_scale"]) or (not flux and sampling["guidance_scale"] <= 1):
        parser.error("SD3 needs --guidance-scale > 1 for negative_prompt to participate in CFG.")
    if (args.device_map or args.cpu_offload) and not args.device.startswith("cuda"):
        parser.error("--device-map and --cpu-offload require a CUDA device.")
    if args.max_memory is not None and (not args.device_map or not isinstance(args.max_memory, dict)):
        parser.error("--max-memory requires --device-map and a JSON object.")
    if not math.isfinite(args.gpu_memory_reserve) or args.gpu_memory_reserve < 0:
        parser.error("--gpu-memory-reserve must be a finite nonnegative number.")

    cases = definitions()["cases"]
    experiments = []
    selected = select_cases(args.cases) if not info or args.cases != "all" else info["cases"]
    if info and set(selected) - set(info["cases"]):
        parser.error(f"Cases not present in --run-dir: {sorted(set(selected) - set(info['cases']))}")
    for case_id in selected:
        if info:
            experiment = json.loads((run_root / case_id / "experiment.json").read_text(encoding="utf-8"))
            experiments.append(experiment)
            continue
        case = cases[case_id]
        group = case["group"]
        task = "instance" if group == "ip" else group
        dataset = dict(case["tasks"][task]["dataset"])
        if dataset.get("path"):
            dataset["path"] = str(BENCHMARK_ROOT / dataset["path"])
        if group == "i2p":
            dataset.update(i2p_subset="all" if args.with_q16 else args.i2p_subset, limit_prompts=args.i2p_prompt_count)
        else:
            if args.seeds is not None:
                dataset["seeds"] = args.seeds
            if group in {"ip", "style"}:
                dataset["limit_prompts_per_concept"] = getattr(args, f"{group}_template_count")
        records, source = prompt_records({"task": task, "dataset": dataset})
        if group == "object":
            # Template 0 exactly matches imagenet10.csv; extra templates are opt-in.
            records = [{**record, "prompt": template.format(record["concept"]), "template_index": index,
                        "sample_id": record["sample_id"].replace("template-000", f"template-{index:03d}")}
                       for record in records
                       for index, template in enumerate(object_templates[:args.object_template_count])]
            source["templates"] = object_templates[:args.object_template_count]
        targets = {concept_key(target) for target in case["targets"]}
        records.sort(key=lambda record: concept_key(record["concept"]) not in targets)
        # Every prompt in a case, including retained concepts, uses the SAME targets.
        negative_prompt = ", ".join(case["targets"])
        for record in records:
            record.update(negative_prompt=negative_prompt, image=f"images/{record['sample_id']}.png")
        experiment = dict(case_id=case_id, group=group, task=task, targets=case["targets"],
                          negative_prompt=negative_prompt, dataset=source, records=records)
        experiments.append(experiment)
    for experiment in experiments:
        records = experiment["records"]
        print(f"[{experiment['case_id']}] {len(records)} images; negative_prompt={experiment['negative_prompt']!r}", flush=True)
        for record in records[:2]:
            print(f"  seed={record['seed']}  prompt={record['prompt']!r}", flush=True)
        if args.phase != "sample" and experiment["group"] in {"ip", "style"}:
            counts = Counter(record["concept"] for record in records)
            if min(counts.values()) < 2:
                parser.error(f"{experiment['case_id']}: FID needs at least 2 images per concept; "
                             "increase template/seed counts, or use --phase sample for images only.")
        if args.with_q16 and experiment["group"] == "i2p" and experiment["dataset"]["options"].get("i2p_subset") == "sexual":
            parser.error("Q16 requires an I2P run prepared with --i2p-subset all.")

    print(f"{args.model}: {sum(len(exp['records']) for exp in experiments)} images total; "
          f"guidance_scale={sampling['guidance_scale']}" +
          (f", true_cfg_scale={sampling['true_cfg_scale']}" if flux else ""), flush=True)
    if args.dry_run:
        print(f"Phase: {args.phase}; IP/style/object references use negative_prompt=''.", flush=True)
        return 0

    if run_root is None:
        run_root = unique_directory(args.output_dir.expanduser().resolve() / args.model, timestamp())
        write_json(run_root / "run.json", {"model": args.model, "base_model": config["base_model"],
                   "sampling": sampling, "cases": [exp["case_id"] for exp in experiments],
                   "image_count": sum(len(exp["records"]) for exp in experiments)})
        for experiment in experiments:
            experiment["sampling"] = {**sampling, "negative_prompt": experiment["negative_prompt"]}
            write_json(run_root / experiment["case_id"] / "experiment.json", experiment)
    print(f"Results: {run_root}", flush=True)
    run_experiments(run_root, config, experiments, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
