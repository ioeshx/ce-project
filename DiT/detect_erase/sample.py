"""Sample gated negative prompts and evaluate through the native benchmark."""
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"

import argparse
import copy
import csv
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.common import ROOT as BENCHMARK_ROOT, SCHEMA, concept_key, configuration, read_json, slug, write_csv, write_json
from benchmark.data import adapter_identity, load_manifest, prompt_records, validate_manifest
from benchmark.dataset.prompt_templates import template_dict
from benchmark.dit_cases import definitions, select_cases
from benchmark.layout import original_directory
from benchmark.results import upsert_summary
from benchmark.sampling_profile import canonical_protocol
from DiT.detect_erase.outputs import concept_checkpoints, run_log, update_run
from DiT.detect_erase.prompts import load_concept_prompts


def summarize_gates(manifest, gate_log, nude_keys):
    """Score actual selected prompts; repeated generation replaces the earlier decision."""
    lines = gate_log.read_text(encoding="utf-8").splitlines() if gate_log.exists() else []
    decisions = {}
    for line in lines:
        decision = json.loads(line)
        decisions[(decision["sample_id"], int(decision["seed"]))] = decision
    targets = {concept_key(target): target for target in manifest["targets"]}
    fields = ("total_count", "recorded_count", "trigger_count", "overall_correct_count", "target_count",
              "target_correct_count", "target_other_count", "target_wrong_only_count", "target_no_trigger_count",
              "non_target_count", "non_target_false_count", "non_target_no_trigger_count")
    overall = Counter(dict.fromkeys(fields, 0))
    by_concept, records = {}, []
    by_erase = {key: Counter(selected_count=0, correct_count=0, wrong_target_count=0, non_target_false_count=0)
                for key in targets}
    for record in manifest["records"]:
        key = record["sample_id"], int(record["seed"])
        if manifest["task"] == "i2p":
            expected = "nudity" if key in nude_keys else None
            label = "nudity" if expected else "non_target"
        else:
            expected = concept_key(record["concept"]) if concept_key(record["concept"]) in targets else None
            label = record["concept"]
        if label not in by_concept:
            by_concept[label] = Counter(dict.fromkeys(fields, 0))
        counters = overall, by_concept[label]
        for counts in counters:
            counts["total_count"] += 1
        decision = decisions.get(key)
        row = dict(sample_id=key[0], seed=key[1], concept=record["concept"], prompt=record["prompt"],
                   expected_erase_prompt=targets.get(expected, ""), recorded=decision is not None)
        if decision is not None:
            selected = {concept_key(value) for value in decision["selected"]}
            correct = expected is not None and expected in selected
            other = bool(selected - {expected})
            for counts in counters:
                counts["recorded_count"] += 1
                counts["trigger_count"] += bool(selected)
                counts["overall_correct_count"] += selected == ({expected} if expected else set())
                if expected:
                    counts["target_count"] += 1
                    counts["target_correct_count"] += correct
                    counts["target_other_count"] += other
                    counts["target_wrong_only_count"] += bool(selected) and not correct
                    counts["target_no_trigger_count"] += not selected
                else:
                    counts["non_target_count"] += 1
                    counts["non_target_false_count"] += bool(selected)
                    counts["non_target_no_trigger_count"] += not selected
            for selected_key in selected:
                counts = by_erase[selected_key]
                counts["selected_count"] += 1
                counts["correct_count"] += selected_key == expected
                counts["wrong_target_count"] += expected is not None and selected_key != expected
                counts["non_target_false_count"] += expected is None
            row.update(selected_erase_prompts=json.dumps(decision["selected"], ensure_ascii=False),
                       correct_trigger=correct if expected else None, other_trigger=other if expected else None,
                       false_trigger=bool(selected) if expected is None else None, no_trigger=not selected)
        records.append(row)

    def rates(counts):
        result = dict(counts)
        result["missing_count"] = counts["total_count"] - counts["recorded_count"]
        for name, numerator, denominator in (
            ("coverage", "recorded_count", "total_count"),
            ("trigger_rate", "trigger_count", "recorded_count"),
            ("overall_accuracy", "overall_correct_count", "recorded_count"),
            ("target_correct_rate", "target_correct_count", "target_count"),
            ("target_other_rate", "target_other_count", "target_count"),
            ("target_wrong_only_rate", "target_wrong_only_count", "target_count"),
            ("target_no_trigger_rate", "target_no_trigger_count", "target_count"),
            ("non_target_false_rate", "non_target_false_count", "non_target_count"),
            ("non_target_no_trigger_rate", "non_target_no_trigger_count", "non_target_count"),
        ):
            result[name] = counts[numerator] / counts[denominator] if counts[denominator] else None
        return result

    concept_rows = [{"concept": label, **rates(counts)} for label, counts in by_concept.items()]
    expected_counts = Counter(concept_key(row["expected_erase_prompt"]) for row in records
                              if row["recorded"] and row["expected_erase_prompt"])
    erase_rows = []
    for key, counts in by_erase.items():
        row = {"erase_prompt": targets[key], **counts, "expected_count": expected_counts[key]}
        row["correct_rate"] = counts["correct_count"] / expected_counts[key] if expected_counts[key] else None
        other_targets = overall["target_count"] - expected_counts[key]
        row["wrong_target_rate"] = counts["wrong_target_count"] / other_targets if other_targets else None
        row["non_target_false_rate"] = (counts["non_target_false_count"] / overall["non_target_count"]
                                         if overall["non_target_count"] else None)
        erase_rows.append(row)
    return dict(summary=rates(overall), by_concept=concept_rows, by_erase_prompt=erase_rows), records


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", help="Cases/groups; default all, or existing cases for evaluate")
    parser.add_argument("--phase", choices=("all", "sample", "evaluate"), default="all")
    parser.add_argument("--run-dir", type=Path, required=True, help="Experiment root containing per-concept files and run.json")
    parser.add_argument("--similarity", choices=("cosine", "l2"))
    parser.add_argument("--tau", type=float, help="Cosine threshold; default: trained value (0.2)")
    parser.add_argument("--l2-threshold", type=float)
    parser.add_argument("--multi-concept", choices=("top1", "mean"))
    parser.add_argument("--i2p-prompt-count", type=int)
    parser.add_argument("--i2p-subset", choices=("sexual", "all"))
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
    if args.track and not args.dry_run and args.swanlab_mode == "local":
        from importlib.util import find_spec
        if find_spec("swanboard") is None:
            parser.error("Local SwanLab requires: pip install 'swanlab[dashboard]'; or use --swanlab-mode offline.")
    generation_fields = ("steps", "height", "width", "seeds", "i2p_prompt_count", "i2p_subset",
                         "tau", "similarity", "l2_threshold", "multi_concept", "ip_template_count",
                         "style_template_count", "object_template_count")
    generation_overrides = any(getattr(args, key) is not None for key in generation_fields)
    for key, default in dict(similarity="cosine", l2_threshold=0.2, multi_concept="top1",
                             ip_template_count=80, style_template_count=30, object_template_count=1).items():
        if getattr(args, key) is None:
            setattr(args, key, default)
    object_templates = ["a photo of {}", *template_dict["object"]]
    if args.batch_size <= 0 or args.metric_batch_size <= 0 or args.l2_threshold < 0:
        parser.error("Batch sizes must be positive; l2-threshold must be nonnegative.")
    if args.tau is not None and not -1 <= args.tau <= 1:
        parser.error("tau must lie in [-1, 1].")
    for group, maximum in (("ip", 80), ("style", 30), ("object", len(object_templates))):
        if not 1 <= getattr(args, f"{group}_template_count") <= maximum:
            parser.error(f"--{group}-template-count must be in [1, {maximum}].")
    if args.i2p_prompt_count is not None and args.i2p_prompt_count <= 0:
        parser.error("--i2p-prompt-count must be positive.")
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

    directory = args.run_dir.expanduser().resolve()
    info = read_json(directory / "run.json")
    selected = (list(info.get("manifests", {})) if args.phase == "evaluate" and args.cases is None else
                select_cases(args.cases or "all"))
    if not selected:
        parser.error("No benchmark cases have been prepared in this run.")
    manifests = {case: load_manifest(directory / case / "manifest.json") for case in selected
                 if (directory / case / "manifest.json").exists()}
    if manifests and generation_overrides:
        parser.error("Existing cases reuse their saved prompts/protocol; omit generation overrides or start a new run.")
    if args.phase == "evaluate" and set(selected) - set(manifests):
        parser.error("Evaluate requires completed manifests for all selected cases.")
    if args.device and args.phase != "evaluate" and any(
            args.device.split(":")[0] != manifest["config"]["sampling"]["device"].split(":")[0]
            for manifest in manifests.values()):
        parser.error("Resume sampling on the original execution backend, or create a new run.")
    new_cases = [case for case in selected if case not in manifests]
    cases = definitions()["cases"]
    if new_cases:
        targets = {target for case in new_cases for target in cases[case]["targets"]}
        checkpoints = concept_checkpoints(directory, sorted(targets))
        missing = [f"{item['target']}:{item['stage']}" for item in checkpoints if not Path(item["checkpoint"]).is_file()]
        if missing:
            parser.error(f"Missing concept checkpoints: {', '.join(missing)}")
        entries = load_concept_prompts(checkpoints)
        trained = info["config"]["erase"]
        detector = info["config"]["detect"]
        if (any(trained[key] != detector[key] for key in ("model_name", "base_model"))
                or any(trained["sampling"].get(key) != detector["sampling"].get(key)
                       for key in ("max_sequence_length", "dtype", "model_revision"))):
            parser.error("Detect and erase require the same model, text length, dtype and revision.")
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
            cfg = configuration(task=task, base=dict(case_id=case, model_name=trained["model_name"], result_layout="flat",
                                base_model=trained["base_model"], method="DetectErase", sampling=sampling,
                                dataset=dataset, cache_dir=str(args.cache_dir.resolve()),
                                evaluation={**evaluation, "vlm_enabled": group in {"ip", "style"} and not args.skip_vlm}))
            records, source = prompt_records(cfg)
            if group == "object":
                expanded = []
                for record in records:
                    for index, template in enumerate(object_templates[:args.object_template_count]):
                        expanded.append({**record, "prompt": template.format(record["concept"]), "template_index": index,
                                         "sample_id": f"object-{slug(record['concept']).lower()}-template-{index:03d}-seed-{record['seed']}"})
                records = expanded
                source["templates"] = object_templates[:args.object_template_count]
            if args.phase != "sample" and group in {"ip", "style"}:
                if min(Counter(record["concept"] for record in records).values()) < 2:
                    parser.error("FID needs >= 2 images per concept; increase templates/seeds or use --phase sample.")
            prepared.append((case, cfg, records, source))
        if args.dry_run:
            for case, _, records, _ in prepared:
                print(f"{case}: {len(records)} generated images", flush=True)
        else:
            for case, cfg, records, source in prepared:
                case_dir = directory / case
                case_dir.mkdir(exist_ok=True)
                cfg.update(output_dir=str(directory))
                cfg["adapter"] = dict(kind="plugin", factory="DiT.detect_erase.prompts:PromptModel",
                                      checkpoints=concept_checkpoints(directory, cfg["targets"]), source_files=["train.py"],
                                      similarity=args.similarity, tau=args.tau,
                                      l2_reduction="rms", l2_threshold=args.l2_threshold, multi_concept=args.multi_concept,
                                      gate_log=str(case_dir / "gates.jsonl"))
                manifests[case] = prepare_case(cfg, records, source, case_dir / "manifest.json")
            update_run(directory, trained["model_name"],
                       manifests={case: str(directory / case / "manifest.json") for case in new_cases})

    if args.dry_run:
        for case, manifest in manifests.items():
            print(f"{case}: {len(manifest['records'])} images, phase={args.phase}")
        return

    groups = list(dict.fromkeys(cases[case]["group"] for case in selected))
    with run_log(directory, "sample_" + "_".join(groups)):
        run_cases(args, directory, [manifests[case] for case in selected], evaluation)


def run_cases(args, directory, manifests, evaluation):
    """Sample/evaluate the selected cases and publish shared summaries."""

    tracker, nude_keys = None, set()
    if args.track:
        import swanlab
        if any(manifest["task"] == "i2p" for manifest in manifests):
            with args.real_nude_mapping.open(encoding="utf-8-sig", newline="") as source:
                nude_keys = {(row["sample_id"], int(row["seed"])) for row in csv.DictReader(source)
                             if row["is_real_nude"] == "1"}
        tracker = swanlab.init(project=args.swanlab_project,
                               name=args.swanlab_name or "sample-" + "+".join(m["case_id"] for m in manifests),
                               mode=args.swanlab_mode, log_dir=str(directory / "swanlog"),
                               config={"run_dir": str(directory), "cases": [m["case_id"] for m in manifests],
                                       "real_nude_mapping": str(args.real_nude_mapping.resolve()),
                                       "i2p_positive_records": len(nude_keys)})

    from benchmark.adapters import sample_manifest
    from benchmark.evaluate import evaluate
    for index, manifest in enumerate(manifests):
        path = directory / manifest["case_id"] / "manifest.json"
        config = manifest["config"]
        # Telemetry is outside adapter identity and never changes sampling or cache keys.
        config["tracking"] = {"enabled": args.track}
        config["evaluation"].update(evaluation)
        config["evaluation"]["vlm_enabled"] = config["task"] in {"instance", "style"} and not args.skip_vlm
        if args.device:
            config["sampling"]["device"] = args.device
        if args.phase != "evaluate":
            config["sampling"].update(batch_size=args.batch_size, local_files_only=args.offline)
            if args.device_map or args.cpu_offload:
                config["sampling"].update(device_map=args.device_map, cpu_offload=args.cpu_offload,
                                           max_memory=args.max_memory)
            sample_manifest(manifest, mode="image" if config["task"] == "i2p" else "both", manifest_path=path)
        if tracker:
            case = manifest["case_id"]
            case_dir = path.parent
            report, records = summarize_gates(manifest, Path(config["adapter"]["gate_log"]), nude_keys)
            write_json(case_dir / "trigger_summary.json", {"case_id": case, "task": config["task"],
                       "targets": manifest["targets"],
                       "label_source": str(args.real_nude_mapping.resolve()) if config["task"] == "i2p" else "benchmark concept",
                       **report})
            write_csv(case_dir / "trigger_records.csv", records)
            write_csv(case_dir / "trigger_by_concept.csv", report["by_concept"])
            write_csv(case_dir / "trigger_by_erase_prompt.csv", report["by_erase_prompt"])
            upsert_summary(directory, "trigger_summary", {"case_id": case, **report["summary"]})
            values = {f"sampling/{case}/{key}": value for key, value in report["summary"].items() if value is not None}
            for category in ("by_concept", "by_erase_prompt"):
                name = "concept" if category == "by_concept" else "erase_prompt"
                for row in report[category]:
                    values.update({f"sampling/{case}/{category}/{slug(row[name])}/{key}": value
                                   for key, value in row.items() if key != name and value is not None})
            tracker.log(values, step=index)
            print(f"[triggers:{case}] {report['summary']}", flush=True)
        if args.phase != "sample":
            evaluate(path, config=config)
    if tracker:
        tracker.finish()
    print(f"Run: {directory}", flush=True)


if __name__ == "__main__":
    main()
