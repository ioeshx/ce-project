"""Executable DiT case definitions, independent of method implementations."""
from __future__ import annotations

import json

from .common import ROOT, merge


def definitions():
    return json.loads((ROOT / "configs/dit_cases.json").read_text(encoding="utf-8"))


def select_cases(selection=None):
    spec = definitions()
    names = [s.strip() for s in (selection or spec["default_case"]).split(",") if s.strip()]
    result = []
    for name in names:
        selected = ([name] if name in spec["cases"] else
                    [key for key, value in spec["cases"].items() if name == "all" or value["group"] == name])
        if not selected:
            raise ValueError(f"Unknown DiT case/group {name!r}; choose all, i2p, ip, style, object or a case_id.")
        result.extend(key for key in selected if key not in result)
    if not result:
        raise ValueError("Choose at least one DiT case.")
    return result


def defaults(task, case_id=None):
    spec = definitions()
    case_id = case_id or {"instance": "ip_pikachu", "style": "style_van_gogh",
                          "object": "object_church"}.get(task, "i2p_nudity")
    if case_id not in spec["cases"] or task not in spec["cases"][case_id]["tasks"]:
        raise ValueError(f"DiT case {case_id!r} does not define task {task!r}.")
    case = spec["cases"][case_id]
    evaluation = json.loads((ROOT / "configs/dit_evaluation.json").read_text(encoding="utf-8"))
    return merge({"case_id": case_id, "dit_protocol": "v2", "targets": case["targets"],
                  "evaluation": evaluation}, case["tasks"][task])


def validate_anchor(config):
    spec = config["adapter"]
    if spec.get("requires_anchor") and not isinstance(config.get("erasure_anchor"), str):
        raise ValueError("This method requires an explicit erasure_anchor; no default anchor is supplied.")


def method_settings(path, case_id):
    from .common import read_json
    from .paths import config_paths
    value = config_paths(read_json(path))
    cases = value.pop("cases", {})
    return merge(value, cases.get(case_id, {}))
