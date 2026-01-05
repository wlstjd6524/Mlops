"""
Entrypoint for daily experiment runs.

- Runs one full training cycle and exits (batch job style).
- W&B is enabled only when env is configured.
- No secrets are printed.
"""

from src.main import run_once

def main():
    run_once()

if __name__ == "__main__":
    main()
