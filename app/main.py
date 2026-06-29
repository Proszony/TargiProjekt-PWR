from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from core.distributed_worker import run_worker_process
from ui.main_window import run_application


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fair Monitor application entrypoint")
    parser.add_argument("--mode", choices=["server", "worker"], default="server")
    parser.add_argument("--camera-id", default="")
    parser.add_argument("--server-host", default="")
    parser.add_argument("--server-port", type=int, default=0)
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        type=str.upper,
        help="Terminal log level for worker mode.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    if args.mode == "worker":
        if not args.camera_id:
            parser.error("--camera-id is required in worker mode")
        return run_worker_process(
            project_root=project_root,
            camera_id=args.camera_id,
            server_host=args.server_host or None,
            server_port=args.server_port or None,
            log_level=args.log_level,
        )
    return run_application(project_root)


if __name__ == "__main__":
    sys.exit(main())
