"""Check that a machine is ready to run the experiments.

Usage: python scripts/check_setup.py [--data-dir DIR]
"""
import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = REPO_ROOT / 'requirements.txt'

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=Path, default=REPO_ROOT / 'data' / 'extracted',
                    help='where scripts/setup_data.sh extracted the data')
args = parser.parse_args()

ok = True


def report(passed, msg):
    global ok
    ok &= passed
    print(('[ok]   ' if passed else '[FAIL] ') + msg)


print(f'[info] Python {sys.version.split()[0]}')

# Installed package versions match requirements.txt
for line in REQUIREMENTS.read_text().splitlines():
    line = line.split('#')[0].strip()
    if not line:
        continue
    name, version = line.split('==')
    try:
        installed = importlib.import_module(name).__version__
        report(installed == version, f'{name} {installed} (expected {version})')
    except ImportError:
        report(False, f'{name} not installed')

# Gurobi licence: a model bigger than the pip trial limit (2000 variables) must solve
try:
    import gurobipy as gp
    with gp.Env(params={'OutputFlag': 0}) as env, gp.Model(env=env) as m:
        x = m.addVars(2500)
        m.addConstr(x.sum() >= 1)
        m.setObjective(x.sum())
        m.optimize()
    report(True, 'Gurobi licence works for full-size models')
except Exception as e:
    report(False, f'Gurobi licence: {e}')

# Data (only needed for the real-data experiments)
csvs = list(args.data_dir.rglob('*.csv')) if args.data_dir.is_dir() else []
report(bool(csvs), f'{len(csvs)} CSV files under {args.data_dir}' if csvs
       else f'no data under {args.data_dir} (run scripts/setup_data.sh)')

sys.exit(0 if ok else 1)
