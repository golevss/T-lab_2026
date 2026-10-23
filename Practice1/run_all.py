import pathlib
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parent
commands = [
    ["-m", "unittest", "-v", "test_protocol.py"],
    ["prepare.py", "--users", "10000", "--seed", "42"],
    ["validate.py"],
    ["experiment.py"],
    ["verify_results.py"],
    ["make_report.py"],
]
for args in commands:
    print("Running:", sys.executable, *args, flush=True)
    subprocess.run([sys.executable, *args], cwd=root, check=True)

if (root / "active_model.json").exists():
    subprocess.run(
        [sys.executable, "-m", "unittest", "-v", "test_upgrades.py"], cwd=root, check=True
    )
    subprocess.run([sys.executable, "train_active.py"], cwd=root, check=True)
