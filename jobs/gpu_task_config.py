"""Write an AI Runtime task's hyperparameters.yaml from deploy-time bundle variables.

Runs as the `gpu_task_config` artifact build step in databricks.yml on every
`bundle deploy`. The runtime exposes a hyperparameters.yaml that sits beside an
`ai_runtime_task`'s command script as HYPERPARAMETERS_PATH, and
`jobs/train_multinode/ray_bootstrap.sh` exports its keys as environment variables.
The file is JSON, which is valid YAML.

Usage: python3 jobs/gpu_task_config.py <output_path> KEY=VALUE [KEY=VALUE ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def parse_pairs(pairs: list[str]) -> dict[str, str]:
    """Parse KEY=VALUE arguments into a dict.

    Args:
        pairs: Arguments of the form KEY=VALUE; the value may contain '='.

    Returns:
        The parsed mapping, in argument order.

    Raises:
        ValueError: If an argument has no '=' or an empty key.
    """
    config: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"expected KEY=VALUE, got {pair!r}")
        config[key] = value
    return config


def main(argv: list[str]) -> None:
    """Write the KEY=VALUE pairs in argv[1:] as JSON to argv[0]."""
    if len(argv) < 2:
        raise SystemExit(__doc__)
    Path(argv[0]).write_text(json.dumps(parse_pairs(argv[1:]), indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
