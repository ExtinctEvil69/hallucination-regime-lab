"""Release gate: run all offline unit/oracle/contract tests (no network, no model)."""
import subprocess
import sys

r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=__file__.rsplit("/", 1)[0] or ".")
sys.exit(r.returncode)
