# scripts/dummy_run.py
import argparse
import logging
import os
import sys

# Добавим project_root в sys.path для удобного импорта src.*
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config
from src.utils.logging_utils import setup_logger
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Dummy run for infra test.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config.",
    )
    # Можно добавить простые override'ы (например, seed), если захочется:
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed from config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    cfg_overrides = {}
    if args.seed is not None:
        cfg_overrides["seed"] = args.seed

    cfg = load_config(args.config, overrides=cfg_overrides)
    cfg_dict = cfg.to_dict()

    # Setup seed
    seed = cfg_dict.get("seed", 42)
    set_seed(seed)
    # Setup logger
    log_cfg = cfg_dict.get("logging", {})
    log_level_str = log_cfg.get("level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_dir = log_cfg.get("log_dir", None)

    logger = setup_logger(name="mdm_project", log_level=log_level, log_dir=log_dir)

    logger.info("===== Dummy run started =====")
    logger.info(f"Using config: {args.config}")
    logger.info(f"Effective seed: {seed}")
    logger.info(f"Config dict: {cfg_dict}")
    logger.info("Everything seems to work. Infrastructure is ready for the next steps.")
    logger.info("===== Dummy run finished =====")


if __name__ == "__main__":
    main()
