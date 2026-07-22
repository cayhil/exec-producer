"""CLI entry shim. --data-dir must take effect BEFORE exec_producer.config
loads (show config is resolved into module globals at import time), so this
module — which imports nothing heavy — scans argv, sets the env var, and
only then imports the real CLI. A broken data contract surfaces here as a
clean message instead of an import traceback."""

import os
import sys

from .errors import ConfigError


def apply_data_dir_arg(argv):
    """Pre-parse --data-dir out of argv into EXEC_PRODUCER_DATA_DIR.
    argparse still declares the flag (for --help); by parse time it has
    already done its job."""
    for i, arg in enumerate(argv):
        if arg == "--data-dir" and i + 1 < len(argv):
            os.environ["EXEC_PRODUCER_DATA_DIR"] = argv[i + 1]
        elif arg.startswith("--data-dir="):
            os.environ["EXEC_PRODUCER_DATA_DIR"] = arg.split("=", 1)[1]


def main():
    apply_data_dir_arg(sys.argv[1:])
    try:
        from .cli import main as cli_main
    except ConfigError as e:
        print(f"❌ {e}")
        sys.exit(1)
    cli_main()
