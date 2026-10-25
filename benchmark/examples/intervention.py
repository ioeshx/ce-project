"""Runnable inference-intervention plugin example: configurable negative conditioning.

This is an integration example, NOT a published erasure baseline. Replace generate
with your method's embedding edits/hooks/additional sampling, retaining row order.
"""
from benchmark.adapters import DiffusersAdapter


class NegativeConditioning(DiffusersAdapter):
    def generate(self, records, sampling_config):
        opts = dict(sampling_config)
        opts["negative_prompt"] = self.config["adapter"].get("intervention_prompt", "Snoopy")
        return super().generate(records, opts)


def create(config):
    return NegativeConditioning(config)
