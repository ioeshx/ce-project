import json
from dataclasses import dataclass, asdict


@dataclass
class BaseConfig:

    def __str__(self):
        return json.dumps(asdict(self), indent=2)