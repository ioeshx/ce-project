"""Runnable prompt-replacement plugin example, not a published erasure baseline.

Replace generate() with your attention/embedding intervention while keeping the
record order, explicit targets/anchor, native sampling and close() interface.
"""
from __future__ import annotations

import copy
import re

from benchmark.dit_adapter import DiTAdapter


class PromptReplacement:
    def __init__(self, config):
        if not isinstance(config.get("erasure_anchor"), str):
            raise ValueError("This example needs an explicit erasure_anchor; no default is supplied.")
        self.config = copy.deepcopy(config)
        native = copy.deepcopy(config)
        native["adapter"] = {"kind": "original"}
        self.backend = DiTAdapter(native)
        self.pipe = self.backend.pipe  # Exposes all GPU devices to the existing sampling timer.
        self.runtime_info = self.backend.runtime_info

    def generate(self, records, sampling_config):
        targets = sorted(self.config["targets"], key=len, reverse=True)
        pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(t) for t in targets) + r")(?!\w)", re.IGNORECASE)
        rewritten = [dict(row, prompt=pattern.sub(lambda match: self.config["erasure_anchor"], row["prompt"]))
                     for row in records]
        # The caller's records keep the original prompt for CS and auditing.
        return self.backend.generate(rewritten, sampling_config)

    def close(self):
        self.backend.close()
        self.pipe = None


def create(config):
    return PromptReplacement(config)
