import argparse
import json
from dataclasses import is_dataclass, fields
from typing import get_origin, List, get_args, Dict, Union

import subprocess
import datetime


def create_git_checkpoint(message: str = None, change_type: str = None, timestamp=True):
    """
    Creates a Git commit to checkpoint the current state of the repository.

    Args:
        message (str): Optional commit message. If not provided, generates one with timestamp.
        change_type (str): Optional type of change (e.g., 'feature', 'bugfix', 'experiment'). Adds an emoji to the message.
    """
    try:
        # Ensure we're inside a Git repo
        subprocess.run(['git', 'rev-parse', '--is-inside-work-tree'], check=True, stdout=subprocess.DEVNULL)

        # Add all tracked changes (excluding untracked files)
        subprocess.run(['git', 'add', '-u'], check=True)

        # Emoji mapping
        emoji_map = {
            "feature": "✨",
            "bugfix": "🐛",
            "experiment": "🧪",
            "refactor": "♻️",
            "docs": "📝",
            "style": "💄",
            "perf": "⚡",
            "test": "✅",
            "config": "🔧",
            "cleanup": "🧹",
            "remove": "🔥",
            "build": "🏗️",
            "version": "🔖",
            "deploy": "🚀",
            "ui": "🎨",
            "security": "🔒",
            "hotfix": "🚑",
            "dependencies": "📦",
            "logs": "📈",
            "revert": "⏪"
        }

        # Generate message if not provided
        if not message:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            emoji = emoji_map.get(change_type, "🧪") if change_type else ""
            label = change_type if change_type else "checkpoint"
            message = f"{emoji} Checkpoint: {label}"

        if timestamp:
            # Generate current time string
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            message += f" at {timestamp}"

        # Commit changes
        subprocess.run(['git', 'commit', '-m', message], check=True)
        print(f"✅ Git checkpoint created: {message}")

        # Print recent commit history
        print("📄 Recent local commits:")
        subprocess.run(['git', 'log', '--oneline', '--decorate', '--graph', '--all', '-n', '10'])

    except subprocess.CalledProcessError as e:
        print("⚠️ Failed to create Git checkpoint. Are you inside a Git repo and is everything set up?")
        print(e)
    except Exception as e:
        print("🚨 An unexpected error occurred.")
        print(e)


def add_args_from_dataclass(parser: argparse.ArgumentParser, cfg):

    if not is_dataclass(cfg):
        raise ValueError("cfg must be a dataclass instance")

    for f in fields(cfg):
        name = f.name
        default = getattr(cfg, name)
        ann = f.type
        arg_name = f"--{name}"
        origin = get_origin(ann)

        # Handle Optional[...] by unwrapping Union
        if origin is Union:
            args = [a for a in get_args(ann) if a is not type(None)]
            if len(args) == 1:
                ann = args[0]
                origin = get_origin(ann)

        # bool flags
        if isinstance(default, bool):
            # ----- Old variant using store_true/store_false -----
            # action = "store_false" if default else "store_true"
            # parser.add_argument(arg_name, action=action, help=f"{name} (default: {default})")
            # ----------------------------------------------------

            def str_to_bool(v):
                if isinstance(v, bool):
                    return v
                if v.lower() in ('yes', 'true', 't', 'y', '1'):
                    return True
                elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                    return False
                else:
                    raise argparse.ArgumentTypeError(f'Boolean value expected for {name}, got: {v}')
            
            parser.add_argument(arg_name, type=str_to_bool, default=default, 
                              help=f"{name} (default: {default})")
            continue

        # list[str], list[int], ...
        elif origin is list or origin is List:
            subtype = get_args(ann)[0] if get_args(ann) else str

            def split_comma_list(s):
                if not s:
                    return []
                try:
                    return [subtype(x.strip()) for x in s.split(",")]
                except ValueError as e:
                    raise argparse.ArgumentTypeError(
                        f"Invalid {name} list item: {e}"
                    )

            parser.add_argument(arg_name, type=split_comma_list, default=default,
                                help=f"{name} (comma-separated list of {subtype.__name__})")
            continue

        # dicts via JSON
        elif origin is dict or origin is Dict:
            parser.add_argument(arg_name, type=lambda s: json.loads(s), default=default, help=f"{name} (JSON string)")
            continue

        elif ann == List[str] or (origin is Union and str in get_args(ann)):
            parser.add_argument(arg_name, nargs="+", type=str, default=default)
            continue

        else:
            # fallback to str/int/float
            parser.add_argument(arg_name, type=type(default) if default is not None else ann, default=default,
                                help=f"{name} (default: {default})")
