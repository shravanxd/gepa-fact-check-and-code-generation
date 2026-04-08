#!/usr/bin/env python3

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def check_env_setup():
    openai_key = os.getenv("OPENAI_API_KEY")
    wandb_key = os.getenv("WANDB_API_KEY")
    
    checks = {
        "OPENAI_API_KEY configured": bool(openai_key),
        "WANDB_API_KEY configured": bool(wandb_key),
        "reflector_invoker.py exists": os.path.exists("reflector_invoker.py"),
        "generate_reflection_logs.py exists": os.path.exists("generate_reflection_logs.py"),
        ".env file exists": os.path.exists(".env")
    }
    
    print("\nEnvironment Check:")
    print("-" * 60)
    for check_name, result in checks.items():
        status = "PASS" if result else "FAIL"
        print(f"{check_name:<40} [{status}]")
    print("-" * 60)
    
    all_pass = all(checks.values())
    
    if not all_pass:
        print("\nMissing Configuration:")
        if not openai_key:
            print("  1. Add to .env: OPENAI_API_KEY=sk-proj-...")
        if not wandb_key:
            print("  2. Optional: Add to .env: WANDB_API_KEY=... (get from https://wandb.ai/authorize)")
        print("\n  2. Install packages: pip install openai python-dotenv wandb")
        sys.exit(1)
    
    print("\nSetup Ready. Run:")
    print("  python generate_reflection_logs.py")
    print("\nThen check results:")
    print("  ls -la reflector_logs/")
    print("  cat reflector_logs/*.jsonl | head")


if __name__ == "__main__":
    check_env_setup()
