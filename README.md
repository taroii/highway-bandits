# highway-bandits

Code for **"Learning a Lipschitz Reward Surface with Endogenous Partitions"**

`online-tolls/` is copied from [StanfordASL/online-tolls](https://github.com/StanfordASL/online-tolls) at commit `fb161ab` (MIT licence, see `online-tolls/LICENSE`), the code for their AISTATS 2023 paper, and modified here to suit newer Python / packages and better fit our setting. Its `requirements.txt` is their original conda export; use the top-level `requirements.txt` instead.



## Requirements

Create a fresh environment and install the Python dependencies:

```bash
conda create -n env_name python=3.11
conda activate env_name
pip install -r requirements.txt
```

Install PyTorch separately to match your hardware. On CPU only:

```bash
pip install torch torchvision
```

On CUDA (example — substitute your CUDA version):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

See https://pytorch.org/get-started/locally/ for the correct command for your system.

### Gurobi licence

The simulators solve LPs and QPs with Gurobi. The licence bundled with `pip install gurobipy` is size-limited and too small for Sioux Falls, so each machine needs a full licence:

- **Academic named-user licence:** request a new key for each machine at https://portal.gurobi.com, then run `grbgetkey <key>` from the university network (or VPN). `grbgetkey` isn't included in the pip package; get it with `conda install -c gurobi gurobi`, or download Gurobi's standalone licence tools for Linux. The licence is written to `~/gurobi.lic` by default.
- **Headless/remote servers:** if `grbgetkey` can't validate from the server, use an Academic WLS licence instead: download `gurobi.lic` from the portal and copy it to `~/gurobi.lic` (or set `GRB_LICENSE_FILE`).

### Data

Our work uses proprietary LA Metro ExpressLanes data, which we do not have permission to share. For our internal purposes, here are instructions to run our code using that data on a new machine (data file name concealed for privacy): 

Copy `data.zip` onto the machine (e.g. with `scp`), then (needs `unzip`):

```bash
scripts/setup_data.sh path/to/data.zip [expected_sha256]
```

This extracts everything, including zips nested inside the archive, to `data/extracted/`. If you pass the archive's SHA-256, it's checked first. Keep the hash with your own records, not in the repo. The simulator's daily demand profile is derived from this data on the first run and cached in `data/derived/` (also not in git). Without the data, pass a profile CSV with `--set profile=path/to/profile.csv` (columns `slot`, `demand`; 288 five-minute slots).

### Check the setup

```bash
python scripts/check_setup.py
```

It checks the Python and package versions, that Gurobi can solve a full-size model, and that the data is in place.

Gurobi results can differ slightly across Gurobi versions or thread counts when an LP has several optimal solutions, so keep `gurobipy` at the pinned version when comparing runs across machines.

## Running experiments

One command trains and evaluates every method and writes all results as CSVs:

```bash
python experiments/run.py --run-name <name>
```

- Results go to `Results/<name>/`: `config.json` and `run_info.json` (settings, git commit, package versions), `oracle/` (reward surface and calibration), `tuning/` (one-time hyperparameter tuning of the baselines on held-out seeds), one subfolder per method with a CSV per `(T, seed)` job on the managed-lane simulator, `n_sweep/` (our methods at fixed T for each number of intervals N), `synthetic/` (synthetic reward surfaces with known zooming dimension, no traffic simulator), and `sioux_falls/` (the AISTATS 2023 Sioux Falls experiment, run unchanged).
- Rerunning the same command resumes: finished jobs are skipped. A run's settings are fixed when it starts; changing them needs a new `--run-name`.
- Jobs run in parallel. Defaults are set for a shared machine: half the cores (`--workers N` to change), lowered CPU priority (`--nice`, default 10), one thread per process, and on Linux the run stops itself if available RAM drops below 15% (`--min-free-memory`). If the run stops for any reason, its workers and subprocess are stopped too. Finished jobs are kept, so rerunning the same command resumes. Two runs can't use the same `--run-name` at once. Change any setting with `--set KEY=JSON`, e.g. `--set seeds=10 'Ts=[100,1000]'`. `--skip-sioux-falls` leaves out the reproduction.
- `--quick` uses tiny settings for an end-to-end check and skips the Sioux Falls reproduction. Run it on a new machine first.

For long runs on a remote server, detach from the session:

```bash
nohup python experiments/run.py --run-name <name> > <name>.log 2>&1 &
```

Figures and tables, from the CSVs only (works on a partial run):

```bash
python experiments/report.py Results/<name>
```

This writes `Results/<name>/figures/` (regret vs. T, final price schedules, reward surface, regret vs. N, synthetic rates) and `Results/<name>/tables/` (CSV and LaTeX), and prints the tables.