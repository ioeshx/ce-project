"""One experiment directory shared by training and benchmark processes."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
import sys

from benchmark.common import locked, read_json, write_json
from benchmark.layout import timestamp, unique_directory

HERE = Path(__file__).resolve().parent


def run_directory(model, run_dir=None, output_dir=HERE / "results"):
    directory = (Path(run_dir).expanduser().resolve() if run_dir else
                 unique_directory(Path(output_dir).expanduser().resolve(), f"{model}-{timestamp()}"))
    directory.mkdir(parents=True, exist_ok=True)
    update_run(directory, model)
    return directory


def update_run(directory, model, **changes):
    """Merge independent stages/cases without losing another GPU's updates."""
    with locked(directory / ".run.lock"):
        path = directory / "run.json"
        info = read_json(path) if path.exists() else {
            "schema": "detect-erase-run-v1", "model_name": model, "training": {}, "manifests": {}}
        if info["model_name"] != model:
            raise ValueError(f"Run belongs to {info['model_name']}, not {model}.")
        for key, value in changes.items():
            if isinstance(value, dict):
                info[key] = {**info.get(key, {}), **value}
            else:
                info[key] = value
        write_json(path, info, portable=True)


class Tee:
    """Keep console output while recording it in the experiment's logs/."""

    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, value):
        self.file.write(value)
        return self.stream.write(value)

    def flush(self):
        self.file.flush()
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextmanager
def run_log(directory, name):
    logs = directory / "logs"
    logs.mkdir(exist_ok=True)
    with (logs / f"{name}.log").open("a", encoding="utf-8", buffering=1) as file:
        with redirect_stdout(Tee(sys.stdout, file)), redirect_stderr(Tee(sys.stderr, file)):
            yield
