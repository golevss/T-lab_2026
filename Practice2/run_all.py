import pathlib
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parent
for args in [
    ["-m", "unittest", "-v", "test_protocol.py"],
    ["prepare.py"],
    ["validate.py"],
    ["experiment.py"],
    ["validate.py"],
    ["report.py"],
]:
    subprocess.run([sys.executable, *args], cwd=root, check=True)

if (root / "active_model.json").exists():
    subprocess.run(
        [sys.executable, "-m", "unittest", "-v", "test_upgrades.py"], cwd=root, check=True
    )
    subprocess.run([sys.executable, "train_active.py"], cwd=root, check=True)
