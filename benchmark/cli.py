"""Unified benchmark command line."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import ROOT, TASKS, configuration, read_json


def parser():
    p = argparse.ArgumentParser(prog="python -m benchmark", description="统一概念擦除采样、评测与结果展示")
    sub = p.add_subparsers(dest="command", required=True)
    commands = {}
    for name in ("prepare", "sample", "evaluate", "run", "report", "plot"):
        q = sub.add_parser(name)
        commands[name] = q
        q.add_argument("--config", type=Path, help="JSON/YAML；路径相对于当前工作目录")
        q.add_argument("--task", choices=TASKS)
        from .model_profiles import MODEL_IDS
        q.add_argument("--model", choices=tuple(MODEL_IDS), help="模型配置名；默认sd14或JSON中的model_name")
        q.add_argument("--output-dir", type=Path)
        if name in {"report", "plot"}:
            continue
        q.add_argument("--manifest", type=Path, help="prepare输出清单；其他命令输入清单")
        q.add_argument("--method", help="结果CSV中唯一的方法名；同名成功结果覆盖旧行")
        q.add_argument("--targets", help="以逗号分隔；例如 Snoopy,Mickey")
        q.add_argument("--device", help="同时设置采样及指标设备，如 cpu 或 cuda:0")
        q.add_argument("--batch-size", type=int)
        q.add_argument("--cpu-threads", type=int, default=4)
        q.add_argument("--offline", action="store_true")
        q.add_argument("--dtype", choices=("float16", "float32", "bfloat16"))
        q.add_argument("--steps", type=int)
        q.add_argument("--height", type=int)
        q.add_argument("--width", type=int)
        q.add_argument("--guidance-scale", type=float)
        q.add_argument("--cpu-offload", action="store_true", help="DiT模型CPU offload；降低显存需求")
        q.add_argument("--cache-dir", type=Path)
        q.add_argument("--original-dir", type=Path, help="原模型图片目录，默认benchmark/original")
        if name in {"prepare", "run"}:
            q.add_argument("--import-casu", type=Path)
        if name in {"prepare", "sample", "run"}:
            q.add_argument("--adapter", choices=("original", "diffusers", "unet", "transformer", "lora", "casu2", "plugin"))
            q.add_argument("--checkpoint", help="UNet / LoRA / CASU检查点")
            q.add_argument("--model-path", help="完整Diffusers模型目录")
            q.add_argument("--factory", help="插件入口 module:function")
            q.add_argument("--limit-prompts", type=int)
            q.add_argument("--max-records", type=int, help="显式缩小评测集用于冒烟测试，改变协议指纹")
            q.add_argument("--seeds", help="逗号分隔的instance/style种子")
        if name in {"sample", "run"}:
            q.add_argument("--mode", choices=("both", "reference", "image"), default=None)
    q = sub.add_parser("dit", parents=[commands["run"]], add_help=False,
                       help="用同一原模型/擦除方法依次运行I2P检测和MS-COCO 1K保留评测")
    q.add_argument("--tasks", default="i2p,mscoco")
    q.add_argument("--phase", choices=("prepare", "sample", "evaluate", "run"), default="run")
    q.add_argument("--dry-run", action="store_true")
    q = sub.add_parser("download", help="独立准备指标权重；不生成实验分数")
    q.add_argument("--resource", choices=("q16", "clip", "fid", "nudenet"), required=True)
    q.add_argument("--destination", type=Path, default=ROOT / "weights/Q16/ViT-L-14/prompts.p")
    q.add_argument("--clip-model", default="ViT-L/14", help="Q16对应的OpenAI CLIP模型")
    q.add_argument("--include-backbone", action="store_true")
    return p


def get_config(args, manifest=None):
    task = args.task or (manifest or {}).get("task")
    if getattr(args, "import_casu", None) and not task:
        directory = args.import_casu if args.import_casu.is_dir() else args.import_casu.parent
        task = read_json(directory / "dataset.json")["task"]
    config = configuration(args.config, task, (manifest or {}).get("config"), model=getattr(args, "model", None))
    for field in ("output_dir", "cache_dir", "original_dir"):
        if getattr(args, field, None):
            config[field] = str(getattr(args, field).expanduser().resolve())
    if getattr(args, "method", None):
        config["method"] = args.method
    if getattr(args, "targets", None):
        config["targets"] = [x.strip() for x in args.targets.split(",") if x.strip()]
    for group in ("sampling", "evaluation"):
        if getattr(args, "device", None):
            config[group]["device"] = args.device
        if getattr(args, "batch_size", None) is not None:
            config[group]["batch_size"] = args.batch_size
        if getattr(args, "offline", False):
            config[group]["local_files_only"] = True
    for key in ("dtype", "steps", "height", "width", "guidance_scale"):
        value = getattr(args, key, None)
        if value is not None:
            config["sampling"][key] = value
    if getattr(args, "cpu_offload", False):
        config["sampling"]["cpu_offload"] = True
    for key in ("limit_prompts", "max_records"):
        if getattr(args, key, None) is not None:
            config["dataset"][key] = getattr(args, key)
    if getattr(args, "seeds", None):
        config["dataset"]["seeds"] = [int(x.strip()) for x in args.seeds.split(",")]
    for arg, key in (("adapter", "kind"), ("checkpoint", "checkpoint"), ("model_path", "model_path"), ("factory", "factory")):
        if getattr(args, arg, None):
            config["adapter"][key] = getattr(args, arg)
    return configuration(task=config["task"], base=config)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == "dit":
            from .dit_experiments import run
            run(args)
            return 0
        if args.command == "download":
            if args.resource == "q16":
                from .detectors import download_q16
                download_q16(args.destination, args.clip_model, args.include_backbone)
            elif args.resource == "clip":
                from .metrics import CLIPScorer
                CLIPScorer("openai/clip-vit-base-patch16", "cpu")
            elif args.resource == "fid":
                from .metrics import FIDExtractor
                FIDExtractor("cpu")
            else:
                from nudenet import NudeDetector
                NudeDetector()
            return 0
        from .data import import_casu, load_manifest, prepare
        manifest = load_manifest(args.manifest) if getattr(args, "manifest", None) and args.command != "prepare" else None
        config = get_config(args, manifest)
        if manifest and config["task"] != manifest["task"]:
            raise ValueError("Task disagrees with the image manifest.")
        if args.command in {"report", "plot"}:
            if args.command == "report":
                from .results import report
                report(config["output_dir"], config["task"])
            else:
                from .plotting import plot
                plot(config["output_dir"], config["task"])
            return 0
        if args.cpu_threads <= 0:
            raise ValueError("cpu-threads must be positive.")
        # Preparation is CPU-only and does not import model libraries.
        if args.command == "prepare":
            path = import_casu(args.import_casu, config, args.manifest) if args.import_casu else prepare(config, args.manifest)
            print(f"[manifest] {path}")
            return 0
        import torch
        torch.set_num_threads(args.cpu_threads)
        if args.command == "evaluate":
            if manifest is None:
                raise ValueError("evaluate requires --manifest; prepare/import images first.")
            from .evaluate import evaluate
            evaluate(args.manifest, config, args.method, config["targets"] if args.targets else None)
            return 0
        if manifest is None:
            path = import_casu(args.import_casu, config) if getattr(args, "import_casu", None) else prepare(config)
            manifest = load_manifest(path)
        else:
            path = args.manifest
            # Sampling parameters and adapter belong to the prepared plan; only execution overrides are safe.
            if any(getattr(args, key, None) for key in ("adapter", "checkpoint", "model_path", "factory", "limit_prompts", "max_records", "seeds", "targets", "method")):
                raise ValueError("With an existing manifest, change sampling/data/method via a new prepare command.")
            scientific = lambda s: {k: v for k, v in s.items() if k not in {"device", "batch_size", "local_files_only", "cpu_offload"}}
            if (scientific(config["sampling"]) != scientific(manifest["config"]["sampling"])
                    or config["adapter"] != manifest["config"]["adapter"]
                    or config["dataset"] != manifest["config"]["dataset"]):
                raise ValueError("Config changes a prepared sampling protocol. Prepare a new manifest first.")
            for group in ("sampling", "evaluation"):
                for key in ("device", "batch_size", "local_files_only", "cpu_offload"):
                    if key in config[group]:
                        manifest["config"][group][key] = config[group][key]
        if not getattr(args, "import_casu", None):
            from .adapters import sample_manifest
            sample_manifest(manifest, args.mode or ("image" if config["task"] == "i2p" else "both"))
        if args.command == "run":
            from .evaluate import evaluate
            evaluate(path, config)
        return 0
    except (OSError, ValueError, ImportError, RuntimeError, KeyError) as error:
        p.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
