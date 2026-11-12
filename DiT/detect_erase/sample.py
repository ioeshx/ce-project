"""Sample gated negative prompts and evaluate through the native benchmark."""
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"

import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.common import ROOT as BENCHMARK_ROOT, SCHEMA, configuration, read_json, slug, write_json
from benchmark.data import adapter_identity, load_manifest, prompt_records, validate_manifest
from benchmark.dataset.prompt_templates import template_dict
from benchmark.dit_cases import definitions, select_cases
from benchmark.layout import original_directory
from benchmark.sampling_profile import canonical_protocol
from DiT.detect_erase.outputs import all_detect_checkpoints, concept_checkpoints, experiment_directory, run_log, update_run
from DiT.detect_erase.prompts import load_concept_prompts
from DiT.detect_erase.detection import nude_labels, publish_report
from DiT.detect_erase.encoding import TextEncoder


def set_shared_references(config, records, protocol):
    """Use benchmark's shared originals for the exact prompts and sampling protocol."""
    config["original_dir"] = str(BENCHMARK_ROOT / "original" / config["model_name"])
    directory = original_directory(config, records, protocol)
    for record in records:
        record["reference"] = str(directory / f"{record['sample_id']}.png")


def prepare_case(config, records, source, path):
    """Use benchmark image/reference identities with the selected prompt subset."""
    protocol = canonical_protocol(config["sampling"])
    identity = adapter_identity(config)
    images = path.parent / "images"
    images.mkdir(exist_ok=True)
    for record in records:
        record["image"] = str(images / f"{record['sample_id']}.png")
    if config["task"] != "i2p":
        set_shared_references(config, records, protocol)
    manifest = dict(schema=SCHEMA, task=config["task"], case_id=config["case_id"],
                    model_name=config["model_name"], base_model=config["base_model"], method=config["method"],
                    targets=config["targets"], sampling=protocol, dataset=source,
                    adapter_identity=identity, config=config, records=records)
    validate_manifest(manifest, check_images=False)
    write_json(path, manifest, portable=True)
    return manifest


def sampling_cases(selection):
    """Include COCO retention without adding a separate training concept/case."""
    result = []
    for name in selection.split(","):
        name = name.strip()
        selected = ([*select_cases("all"), "mscoco_nudity"] if name == "all" else
                    ["mscoco_nudity"] if name in {"mscoco", "mscoco_nudity"} else select_cases(name))
        result.extend(case for case in selected if case not in result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", help="all/i2p/ip/style/object/mscoco or case IDs; all includes MSCOCO")
    parser.add_argument("--detect-only", action="store_true", help="Evaluate detection without erase checkpoints, DiT, VAE or image generation")
    parser.add_argument("--phase", choices=("all", "sample", "evaluate"), default="all")
    parser.add_argument("--run-dir", type=Path, required=True, help="Experiment root containing per-concept files and run.json")
    parser.add_argument("--output-dir", type=Path, help="Separate test result root; checkpoints still come from run-dir")
    parser.add_argument("--all-detect-prompts", action=argparse.BooleanOptionalAction, default=None,
                        help="Default on: all saved detectors compete for top1 without thresholds")
    parser.add_argument("--similarity", choices=("cosine", "l2"))
    parser.add_argument("--tau", type=float, help="Cosine threshold; default: trained value (0.2)")
    parser.add_argument("--l2-threshold", type=float)
    parser.add_argument("--multi-concept", choices=("top1", "mean"))
    parser.add_argument("--i2p-prompt-count", type=int)
    parser.add_argument("--i2p-subset", choices=("sexual", "all"))
    parser.add_argument("--mscoco-prompt-count", type=int, help="COCO captions; default: benchmark's 1000")
    parser.add_argument("--ip-template-count", type=int)
    parser.add_argument("--style-template-count", type=int)
    parser.add_argument("--object-template-count", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--steps", type=int, help="Sampling steps; CFG scales and text lengths come from training")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", help="Default: training device")
    placement = parser.add_mutually_exclusive_group()
    placement.add_argument("--device-map", choices=("balanced", "auto"))
    placement.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--max-memory", type=json.loads)
    parser.add_argument("--metric-device", default="cuda:0")
    parser.add_argument("--metric-batch-size", type=int, default=32)
    parser.add_argument("--clip-model")
    parser.add_argument("--vlm-model")
    parser.add_argument("--resnet-weights-path")
    parser.add_argument("--nudenet-weights")
    parser.add_argument("--q16-prompts")
    parser.add_argument("--q16-clip-path")
    parser.add_argument("--skip-vlm", action="store_true")
    parser.add_argument("--with-q16", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=BENCHMARK_ROOT / "cache")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--track", action="store_true", help="Record selected prompts, trigger counts/rates and SwanLab summaries")
    parser.add_argument("--swanlab-mode", choices=("local", "offline", "online"), default="local")
    parser.add_argument("--swanlab-project", default="detect-erase")
    parser.add_argument("--swanlab-name")
    parser.add_argument("--real-nude-mapping", type=Path, default=BENCHMARK_ROOT / "dataset/real_nude_mapping.csv")
    args = parser.parse_args(argv)
    if args.detect_only and args.phase == "evaluate":
        parser.error("--detect-only evaluates text directly; omit --phase evaluate.")
    if args.track and not args.dry_run and args.swanlab_mode == "local":
        from importlib.util import find_spec
        if find_spec("swanboard") is None:
            parser.error("Local SwanLab requires: pip install 'swanlab[dashboard]'; or use --swanlab-mode offline.")
    generation_fields = ("steps", "height", "width", "seeds", "i2p_prompt_count", "i2p_subset",
                         "tau", "similarity", "l2_threshold", "multi_concept", "ip_template_count",
                         "style_template_count", "object_template_count", "mscoco_prompt_count", "all_detect_prompts")
    generation_overrides = any(getattr(args, key) is not None for key in generation_fields)
    for key, default in dict(similarity="cosine", l2_threshold=0.2, multi_concept="top1", all_detect_prompts=True,
                             ip_template_count=80, style_template_count=30, object_template_count=1).items():
        if getattr(args, key) is None:
            setattr(args, key, default)
    if args.all_detect_prompts and args.multi_concept != "top1":
        parser.error("All-detector mode uses top1; mean requires --no-all-detect-prompts and threshold gating.")
    object_templates = ["a photo of {}", *template_dict["object"]]
    if args.batch_size <= 0 or args.metric_batch_size <= 0 or args.l2_threshold < 0:
        parser.error("Batch sizes must be positive; l2-threshold must be nonnegative.")
    if args.tau is not None and not -1 <= args.tau <= 1:
        parser.error("tau must lie in [-1, 1].")
    for group, maximum in (("ip", 80), ("style", 30), ("object", len(object_templates))):
        if not 1 <= getattr(args, f"{group}_template_count") <= maximum:
            parser.error(f"--{group}-template-count must be in [1, {maximum}].")
    if any(value is not None and value <= 0 for value in (args.i2p_prompt_count, args.mscoco_prompt_count)):
        parser.error("i2p-prompt-count and mscoco-prompt-count must be positive.")
    if args.seeds is not None and (len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2 ** 63 for seed in args.seeds)):
        parser.error("Seeds must be distinct integers in [0, 2**63).")

    evaluation = dict(device=args.metric_device, batch_size=args.metric_batch_size,
                      local_files_only=args.offline, q16_enabled=args.with_q16)
    for key in ("clip_model", "vlm_model", "resnet_weights_path", "nudenet_weights", "q16_prompts", "q16_clip_path"):
        value = getattr(args, key)
        if value is not None:
            local = Path(value).expanduser()
            evaluation[key] = str(local.resolve()) if local.exists() else value
    for key, filename in (("vlm_model", "Qwen3-VL-2B-Instruct"), ("resnet_weights_path", "resnet50_v2.pth")):
        local = BENCHMARK_ROOT / "weights" / filename
        if key not in evaluation and local.exists():
            evaluation[key] = str(local)

    checkpoint_dir = args.run_dir.expanduser().resolve()
    source_info = read_json(checkpoint_dir / "run.json")
    directory = args.output_dir.expanduser().resolve() if args.output_dir else checkpoint_dir
    info = read_json(directory / "run.json") if (directory / "run.json").exists() else {"manifests": {}}
    if directory != checkpoint_dir and info.get("checkpoint_run_dir", str(checkpoint_dir)) != str(checkpoint_dir):
        parser.error("This output directory belongs to another checkpoint run; choose a new output-dir.")
    selected = (list(info.get("manifests", {})) if args.phase == "evaluate" and args.cases is None else
                sampling_cases(args.cases or "all"))
    if not selected:
        parser.error("No benchmark cases have been prepared in this run.")
    manifests = {} if args.detect_only else {
        case: load_manifest(experiment_directory(directory, case) / "manifest.json") for case in selected
        if (experiment_directory(directory, case) / "manifest.json").exists()}
    if manifests and generation_overrides:
        parser.error("Existing cases reuse their saved prompts/protocol; omit generation overrides or start a new run.")
    if args.phase != "evaluate" and any(
            manifest["config"]["adapter"].get("all_detect_prompts", False) != args.all_detect_prompts
            for manifest in manifests.values()):
        parser.error("Detection mode differs from existing cases; use a new --output-dir.")
    if args.phase == "evaluate" and set(selected) - set(manifests):
        parser.error("Evaluate requires completed manifests for all selected cases.")
    if args.device and args.phase != "evaluate" and any(
            args.device.split(":")[0] != manifest["config"]["sampling"]["device"].split(":")[0]
            for manifest in manifests.values()):
        parser.error("Resume sampling on the original execution backend, or create a new run.")
    new_cases = [case for case in selected if case not in manifests]
    cases = definitions()["cases"]
    cases["mscoco_nudity"] = dict(group="mscoco", targets=cases["i2p_nudity"]["targets"])
    if new_cases:
        targets = {target for case in new_cases for target in cases[case]["targets"]}
        detectors = (all_detect_checkpoints(checkpoint_dir) if args.all_detect_prompts else
                     concept_checkpoints(checkpoint_dir, sorted(targets), ("detect",)))
        if not detectors:
            parser.error("No detector checkpoints found in run-dir/train_detect/.")
        if args.all_detect_prompts and targets - {item["target"] for item in detectors}:
            parser.error(f"Missing detectors for selected erasure targets: {sorted(targets - {item['target'] for item in detectors})}")
        checkpoints = detectors + ([] if args.detect_only else concept_checkpoints(checkpoint_dir, sorted(targets), ("erase",)))
        missing = [f"{item['target']}:{item['stage']}" for item in checkpoints if not Path(item["checkpoint"]).is_file()]
        if missing:
            parser.error(f"Missing concept checkpoints: {', '.join(missing)}")
        entries = load_concept_prompts(checkpoints)
        detector = source_info["config"]["detect"]
        trained = detector if args.detect_only else source_info["config"]["erase"]
        if not args.detect_only:
            if (any(trained[key] != detector[key] for key in ("model_name", "base_model"))
                    or any(trained["sampling"].get(key) != detector["sampling"].get(key)
                           for key in ("max_sequence_length", "dtype", "model_revision"))
                    or trained.get("text", {}) != detector.get("text", {})):
                parser.error("Detect and erase require the same model, text length, dtype, revision and CLIP projection.")
            flux = trained["model_name"].startswith("flux1_")
            scale = trained["sampling"]["true_cfg_scale"] if flux else trained["sampling"]["guidance_scale"]
            if any(entries[target]["cfg_scale"] != scale for target in targets):
                parser.error("Selected erase prompts must share the recorded training CFG scale.")
        prepared = []
        for case in new_cases:
            group = cases[case]["group"]
            task = {"ip": "instance"}.get(group, group)
            dataset = {}
            if group == "i2p":
                dataset.update(i2p_subset=args.i2p_subset or ("all" if args.with_q16 else "sexual"),
                               limit_prompts=args.i2p_prompt_count)
            elif group == "mscoco" and args.mscoco_prompt_count is not None:
                dataset["limit_prompts"] = args.mscoco_prompt_count
            elif group in {"ip", "style"}:
                dataset["limit_prompts_per_concept"] = getattr(args, f"{group}_template_count")
            if args.seeds is not None:
                dataset["seeds"] = args.seeds
            sampling = copy.deepcopy(trained["sampling"])
            sampling.update(batch_size=args.batch_size, local_files_only=args.offline, device_map=args.device_map,
                            cpu_offload=args.cpu_offload, max_memory=args.max_memory, negative_prompt="")
            if args.device:
                sampling["device"] = args.device
            sampling["execution_backend"] = sampling["device"].split(":")[0]
            for key in ("steps", "height", "width"):
                if getattr(args, key) is not None:
                    sampling[key] = getattr(args, key)
            cfg = configuration(task=task, base=dict(case_id=case, model_name=trained["model_name"], result_layout="exps",
                                base_model=trained["base_model"], method="DetectErase", sampling=sampling,
                                dataset=dataset, cache_dir=str(args.cache_dir.resolve()),
                                evaluation={**evaluation, "vlm_enabled": group in {"ip", "style"} and not args.skip_vlm}))
            cfg["text"] = copy.deepcopy(detector.get("text", {}))
            records, source = prompt_records(cfg)
            if group == "object":
                expanded = []
                for record in records:
                    for index, template in enumerate(object_templates[:args.object_template_count]):
                        expanded.append({**record, "prompt": template.format(record["concept"]), "template_index": index,
                                         "sample_id": f"object-{slug(record['concept']).lower()}-template-{index:03d}-seed-{record['seed']}"})
                records = expanded
                source["templates"] = object_templates[:args.object_template_count]
            if not args.detect_only and args.phase != "sample" and group in {"ip", "style"}:
                if min(Counter(record["concept"] for record in records).values()) < 2:
                    parser.error("FID needs >= 2 images per concept; increase templates/seeds or use --phase sample.")
            if not args.detect_only and args.phase != "sample" and group == "mscoco" and len(records) < 2:
                parser.error("MSCOCO FID needs >= 2 captions; increase mscoco-prompt-count or use --phase sample.")
            prepared.append((case, cfg, records, source))
        if args.dry_run:
            for case, _, records, _ in prepared:
                print(f"{case}: {len(records)} {'text detections' if args.detect_only else 'generated images'}", flush=True)
        else:
            directory.mkdir(parents=True, exist_ok=True)
            update_run(directory, trained["model_name"], checkpoint_run_dir=str(checkpoint_dir), config=source_info["config"])
            for case, cfg, records, source in prepared:
                case_dir = experiment_directory(directory, case)
                case_dir.mkdir(parents=True, exist_ok=True)
                cfg.update(output_dir=str(directory))
                case_detectors = (detectors if args.all_detect_prompts else
                                  concept_checkpoints(checkpoint_dir, cfg["targets"], ("detect",)))
                case_checkpoints = case_detectors + ([] if args.detect_only else
                                   concept_checkpoints(checkpoint_dir, cfg["targets"], ("erase",)))
                cfg["adapter"] = dict(kind="plugin", factory="DiT.detect_erase.prompts:PromptModel",
                                      checkpoints=case_checkpoints, source_files=["train.py", "encoding.py"],
                                      all_detect_prompts=args.all_detect_prompts,
                                      detect_targets=[item["target"] for item in case_detectors], erase_targets=cfg["targets"],
                                      similarity=args.similarity, tau=None if args.all_detect_prompts else args.tau,
                                      l2_reduction="rms", l2_threshold=args.l2_threshold, multi_concept=args.multi_concept,
                                      gate_log=str(case_dir / ("detect_gates.jsonl" if args.detect_only else "gates.jsonl")))
                if args.detect_only:
                    manifests[case] = dict(schema="detect-erase-text-eval-v1", case_id=case, task=cfg["task"],
                                           targets=cfg["targets"], config=cfg, records=records, dataset=source)
                    write_json(case_dir / "detect_manifest.json", manifests[case], portable=True)
                else:
                    manifests[case] = prepare_case(cfg, records, source, case_dir / "manifest.json")
            update_run(directory, trained["model_name"],
                       **{("detections" if args.detect_only else "manifests"):
                          {case: str(experiment_directory(directory, case) / ("detect_manifest.json" if args.detect_only else "manifest.json"))
                           for case in new_cases}})

    if args.dry_run:
        for case, manifest in manifests.items():
            print(f"{case}: {len(manifest['records'])} images, phase={args.phase}")
        return

    groups = list(dict.fromkeys(cases[case]["group"] for case in selected))
    with run_log(directory, ("detect_" if args.detect_only else "sample_") + "_".join(groups)):
        run_cases(args, directory, [manifests[case] for case in selected], evaluation)


def run_cases(args, directory, manifests, evaluation):
    """Sample/evaluate the selected cases and publish shared summaries."""

    tracker, nude_keys = None, set()
    if (args.track or args.detect_only) and any(manifest["task"] == "i2p" for manifest in manifests):
        nude_keys = nude_labels(args.real_nude_mapping)
    if args.track:
        import swanlab
        tracker = swanlab.init(project=args.swanlab_project,
                               name=args.swanlab_name or "sample-" + "+".join(m["case_id"] for m in manifests),
                               mode=args.swanlab_mode, log_dir=str(directory / "swanlog"),
                               config={"run_dir": str(directory), "cases": [m["case_id"] for m in manifests],
                                       "real_nude_mapping": str(args.real_nude_mapping.resolve()),
                                       "i2p_positive_records": len(nude_keys)})

    text = None
    from benchmark.adapters import sample_manifest
    from benchmark.evaluate import evaluate
    for index, manifest in enumerate(manifests):
        path = experiment_directory(directory, manifest["case_id"]) / "manifest.json"
        config = manifest["config"]
        # Telemetry is outside adapter identity and never changes sampling or cache keys.
        config["tracking"] = {"enabled": args.track}
        config["evaluation"].update(evaluation)
        config["evaluation"]["vlm_enabled"] = config["task"] in {"instance", "style"} and not args.skip_vlm
        if args.device:
            config["sampling"]["device"] = args.device
        if args.detect_only:
            if text is None:
                text = TextEncoder(config)
            entries = load_concept_prompts(config["adapter"]["checkpoints"])
            text.prepare_detector(entries, config["adapter"])
            gate_log = Path(config["adapter"]["gate_log"])
            with gate_log.open("w", encoding="utf-8") as output:
                for start in range(0, len(manifest["records"]), args.batch_size):
                    batch = manifest["records"][start:start + args.batch_size]
                    decisions = text.detect([record["prompt"] for record in batch])
                    for record, decision in zip(batch, decisions):
                        output.write(json.dumps({**record, **decision}, ensure_ascii=False) + "\n")
            values = publish_report(directory, manifest, gate_log, nude_keys, args.real_nude_mapping,
                                    detect_only=True)
            if tracker:
                tracker.log(values, step=index)
            continue
        if args.phase != "evaluate":
            config["sampling"].update(batch_size=args.batch_size, local_files_only=args.offline)
            if args.device_map or args.cpu_offload:
                config["sampling"].update(device_map=args.device_map, cpu_offload=args.cpu_offload,
                                           max_memory=args.max_memory)
            sample_manifest(manifest, mode="image" if config["task"] == "i2p" else "both", manifest_path=path)
        if tracker:
            values = publish_report(directory, manifest, config["adapter"]["gate_log"],
                                    nude_keys, args.real_nude_mapping)
            tracker.log(values, step=index)
        if args.phase != "sample":
            evaluate(path, config=config)
    if text is not None:
        text.close()
    if tracker:
        tracker.finish()
    print(f"Run: {directory}", flush=True)


if __name__ == "__main__":
    main()
