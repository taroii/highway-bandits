"""Managed-lane experiments: train and evaluate tolling methods on a time-of-day simulator.

One command runs the whole pipeline and writes CSVs to Results/<run-name>/:
  1. demand profile (derived once from data/extracted, cached in data/derived/)
  2. oracle: expected reward surface over (time slot, price) by Monte Carlo, and calibration
  3. every (method, T, seed) job, one CSV per job in Results/<run-name>/<method>/
  4. a reproduction of the AISTATS 2023 Sioux Falls experiment in Results/<run-name>/sioux_falls/

Re-running with the same --run-name skips everything that already finished.
Figures and tables: python experiments/report.py Results/<run-name>

Usage: python experiments/managed_lane.py [--run-name NAME] [--workers N] [--quick]
                                          [--skip-sioux-falls] [--set KEY=JSON ...]
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TOLLS_DIR = REPO_ROOT / 'online-tolls'
BPR_DIR = TOLLS_DIR / 'bpr-approximation'
sys.path.insert(0, str(BPR_DIR))
import gurobipy as gp  # noqa: E402
from network import Network  # noqa: E402  (bpr-approximation version)
from user_equilibrium_with_tolls import UserEquilibriumWithTolls  # noqa: E402

# Defaults; override any entry with --set KEY=JSON. The rationale for each is in the project decisions log.
CONFIG = {
    # experiment grid
    'Ts': [100, 300, 1000, 3000, 10000],
    'seeds': 5,
    'methods': ['oracle', 'no_toll', 'gradient_descent', 'constant_update', 'pooled_ucb', 'pooled_zooming'],
    # simulator
    'network': 'ManagedLane',        # folder in online-tolls/bpr-approximation/Locations
    'el_edge': 0,                    # tolled express-lane edge
    'gp_edge': 2,                    # general-purpose edge
    'slots': 96,                     # time-of-day slots per day; must divide 288 (5-minute data)
    'peak_demand': 8.5,              # mean demand at the busiest slot, in network flow units
    'vehicles_per_flow_unit': 28800,  # vehicles per day carried by one unit of flow (Poisson scale)
    'vot_groups': 10,
    'vot_seed': 5,
    'max_price': 60.0,
    # reward: g = (EL + GP flow) / peak_demand - density_penalty * (EL density above rho_crit) / rho_crit
    'density_penalty': 1.0,
    'rho_crit': None,                # None: EL density at the BPR kink (1.1 * capacity)
    # oracle and calibration
    'price_grid': 61,
    'mc_draws': 20,
    # their baselines
    'gd_eta0': 6.0,                  # gradient descent step size is gd_eta0 / sqrt(T)
    'const_step': 0.6,
    # demand profile CSV (columns slot, demand); None: derive from data/extracted
    'profile': None,
}
QUICK = {'Ts': [20, 50], 'seeds': 2, 'slots': 24, 'price_grid': 13, 'mc_draws': 4}
ORACLE_SEED = 10 ** 6  # Monte Carlo draws use seeds ORACLE_SEED + i, disjoint from run seeds


@contextmanager
def working_dir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


# ----------------------------------------------------------------------------- demand profile

def derive_demand_profile(data_dir, out_path):
    """Weekday demand shape over 288 five-minute slots, normalised to mean 1.

    For each loop-detector CSV (columns data_timestamp, sensor_id, volume): the median over valid
    weekday sensor-days of volume / that sensor-day's mean volume. The result is the average over files.
    """
    files = []
    for p in sorted(data_dir.rglob('*.csv')):
        if '__MACOSX' in p.parts or p.name.startswith('._'):
            continue
        with p.open(errors='ignore') as f:
            if {'data_timestamp', 'sensor_id', 'volume'} <= set(f.readline().strip().split(',')):
                files.append(p)
    if not files:
        raise SystemExit(f'No loop-detector CSVs under {data_dir}; run scripts/setup_data.sh or pass --set profile=...')

    shapes = []
    for path in files:
        print(f'[profile] {path.name}')
        df = pd.read_csv(path, usecols=['data_timestamp', 'sensor_id', 'volume'], na_values='NULL')
        ts = df.pop('data_timestamp')
        df['day'] = pd.to_datetime(ts.str.slice(0, 10))
        df['slot'] = ts.str.slice(11, 13).astype(int) * 12 + ts.str.slice(14, 16).astype(int) // 5
        df = df[df['day'].dt.dayofweek < 5].dropna(subset=['volume'])

        # Valid sensor-day: reports all 288 slots, nonzero mean, <= 10% zero readings 06:00-21:00
        df['daytime'] = df['slot'].between(6 * 12, 21 * 12 - 1)
        df['daytime_zero'] = df['daytime'] & df['volume'].eq(0)
        s = df.groupby(['sensor_id', 'day']).agg(n=('volume', 'size'), mean=('volume', 'mean'),
                                                 zeros=('daytime_zero', 'sum'), daytime=('daytime', 'sum'))
        valid = s[(s['n'] == 288) & (s['mean'] > 0) & (s['zeros'] <= 0.1 * s['daytime'])]
        if valid.empty:
            continue
        df = df.join(valid['mean'], on=['sensor_id', 'day'], how='inner')
        shapes.append((df['volume'] / df['mean']).groupby(df['slot']).median().reindex(range(288)))

    profile = pd.concat(shapes, axis=1).mean(axis=1).interpolate().bfill().ffill()
    profile /= profile.mean()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'slot': range(288), 'demand': profile.values}).to_csv(out_path, index=False)
    print(f'[profile] wrote {out_path} from {len(shapes)} files')


def load_profile(cfg):
    path = Path(cfg['profile']) if cfg['profile'] else REPO_ROOT / 'data' / 'derived' / 'demand_profile.csv'
    if not path.exists():
        derive_demand_profile(REPO_ROOT / 'data' / 'extracted', path)
    demand = pd.read_csv(path)['demand'].to_numpy()
    if len(demand) % cfg['slots']:
        raise SystemExit(f"slots={cfg['slots']} must divide the profile length {len(demand)}")
    return demand.reshape(cfg['slots'], -1).mean(axis=1), path


# ----------------------------------------------------------------------------- simulator

class Users:
    """Stand-in for their Users class: one OD pair split into equal-size value-of-time groups."""

    def __init__(self, orig, dest, vot_means):
        self.num_users = len(vot_means)
        self.data = {u: {'orig': orig, 'dest': dest, 'vol': 0.0, 'vot': v} for u, v in enumerate(vot_means)}


class ManagedLaneEnv:
    """Each day: for every time slot, draw Poisson demand and VOTs, then solve their piecewise-BPR user
    equilibrium given the express-lane price. Draws depend only on (seed, day, slot), so every method
    sees the same demand and VOTs (common random numbers)."""

    def __init__(self, cfg, profile):
        self.cfg = cfg
        with working_dir(BPR_DIR):
            self.network = Network(cfg['network'])
            od = pd.read_csv(f"Locations/{cfg['network']}/od.csv").iloc[0]
        # Plain floats, so their objective never multiplies numpy scalars by Gurobi variables
        self.network.edge_latency = [float(v) for v in self.network.edge_latency]
        # Group mean VOTs as in their main experiment: 5 + 95 * U(0, 1)
        self.vot_means = 5 + 95 * np.random.default_rng(cfg['vot_seed']).random(cfg['vot_groups'])
        self.users = Users(int(od['origin']), int(od['destination']), self.vot_means.tolist())
        self.ue = UserEquilibriumWithTolls(self.network, self.users, np.zeros(self.network.NumEdges),
                                           piecewise_bpr=True)

        self.slots = cfg['slots']
        self.mean_flow = cfg['peak_demand'] * profile / profile.max()
        self.vehicles_per_slot = cfg['vehicles_per_flow_unit'] / self.slots  # per unit flow
        self.t0 = np.array(self.network.edge_latency)
        self.capacity = np.array(self.network.capacity)
        self.el, self.gp = cfg['el_edge'], cfg['gp_edge']
        self.el_length = float(self.network.raw_edges['length'][self.el])
        self.el_target = 1.1 * self.capacity[self.el]
        self.rho_crit = cfg['rho_crit'] or self.el_target * 1.025 * self.t0[self.el] / self.el_length

    def draw(self, seed, day, slot):
        rng = np.random.default_rng([seed, day, slot])
        groups = self.users.num_users
        counts = rng.poisson(self.mean_flow[slot] * self.vehicles_per_slot / groups, size=groups)
        vots = self.vot_means * (0.8 + 0.4 * rng.random(groups))  # their per-round redraw
        return counts / self.vehicles_per_slot, vots

    def solve(self, vols, vots, price):
        for u in range(self.users.num_users):
            self.users.data[u]['vot'] = float(vots[u])
            self.ue.origin_demand[u].RHS = float(vols[u])
            self.ue.dest_demand[u].RHS = float(vols[u])
        tolls = [0.0] * self.network.NumEdges
        tolls[self.el] = float(price)
        self.ue.set_obj(self.users, tolls)
        x, f = self.ue.solve()

        latency = 1.025 * self.t0 + 0.6 * np.maximum(0, f - 1.1 * self.capacity)  # their piecewise BPR
        density = f[self.el] * latency[self.el] / self.el_length
        excess = max(0.0, density - self.rho_crit) / self.rho_crit
        throughput = f[self.el] + f[self.gp]
        reward = throughput / self.cfg['peak_demand'] - self.cfg['density_penalty'] * excess
        return {'reward': min(max(reward, 0.0), 1.0), 'throughput': throughput, 'el_flow': f[self.el],
                'el_excess_density': excess, 'revenue': price * f[self.el],
                'vot_travel_time': vots @ (x.T @ latency)}

    def simulate_day(self, seed, day, prices):
        out = [self.solve(*self.draw(seed, day, k), prices[k]) for k in range(self.slots)]
        return {key: np.array([o[key] for o in out]) for key in out[0]}


# ----------------------------------------------------------------------------- methods
# Each method posts one price per slot (schedule) and learns only from sensor-derived quantities:
# per-slot express-lane flow and per-slot reward Y_k = g_k / slots.

class FixedSchedule:
    def __init__(self, prices):
        self.prices = np.asarray(prices, dtype=float)

    def schedule(self):
        return self.prices

    def update(self, obs):
        pass


class GradientDescent:
    """Theirs (bpr-approximation, 1.1 * capacity target), with one toll per time slot."""

    def __init__(self, slots, T, max_price, eta0, target):
        self.tolls, self.eta, self.max_price, self.target = np.zeros(slots), eta0 / np.sqrt(T), max_price, target

    def schedule(self):
        return self.tolls

    def update(self, obs):
        self.tolls = np.clip(self.tolls + self.eta * (obs['el_flow'] - self.target), 0, self.max_price)


class ConstantUpdate(GradientDescent):
    """Theirs: fixed-size step in the direction of the capacity violation."""

    def __init__(self, slots, max_price, step, target):
        super().__init__(slots, 1, max_price, step, target)

    def update(self, obs):
        self.tolls = np.clip(self.tolls + self.eta * np.sign(obs['el_flow'] - self.target), 0, self.max_price)


def group_layout(slots, T):
    """Pooled-UCB tuning (Corollary 1): J = N/m groups and K grid prices, both ~ T^(1/4); J must divide N."""
    divisors = [d for d in range(1, slots + 1) if slots % d == 0]
    groups = min(divisors, key=lambda d: abs(np.log(d) - 0.25 * np.log(T)))
    return groups, slots // groups, max(1, round(T ** 0.25))


class PooledUCB:
    """Ours (Algorithm 1), with the uniform partition N = slots and delta = 1/T."""

    def __init__(self, slots, T, max_price, sigma):
        self.groups, self.m, self.K = group_layout(slots, T)
        self.grid = (np.arange(1, self.K + 1) - 0.5) * max_price / self.K
        self.width = sigma * np.sqrt(slots / self.m)
        self.iota = np.log(2 * self.groups * self.K * T * T)
        self.slots, self.t = slots, 0
        self.n = np.zeros((self.groups, self.K))
        self.mean = np.zeros((self.groups, self.K))

    def schedule(self):
        if self.t < self.K:
            self.choice = np.full(self.groups, self.t)
        else:
            self.choice = np.argmax(self.mean + self.width * np.sqrt(2 * self.iota / self.n), axis=1)
        return np.repeat(self.grid[self.choice], self.m)

    def update(self, obs):
        z = obs['Y'].reshape(self.groups, self.m).sum(axis=1) * self.slots / self.m
        idx = (np.arange(self.groups), self.choice)
        self.n[idx] += 1
        self.mean[idx] += (z - self.mean[idx]) / self.n[idx]
        self.t += 1


class PooledZooming:
    """Ours (Algorithm 2), as standard zooming on pooled observations: an arm covers prices within
    rad / L_p, and each round one arm is activated at the midpoint of the widest uncovered gap."""

    def __init__(self, slots, T, max_price, sigma, lip):
        self.groups, self.m, _ = group_layout(slots, T)
        self.slots, self.max_price, self.lip = slots, max_price, max(lip, 1e-9)
        self.width = sigma * np.sqrt(slots / self.m)
        self.iota = np.log(2 * self.groups * T * T)  # delta = 1/T, at most T arms per group
        self.arms = [[[max_price / 2, 0, 0.0]] for _ in range(self.groups)]  # [price, pulls, mean]

    def radius(self, n):
        return np.inf if n == 0 else self.width * np.sqrt(2 * self.iota / n)

    def schedule(self):
        self.choice = [max(range(len(a)), key=lambda i: a[i][2] + 2 * self.radius(a[i][1])) for a in self.arms]
        return np.repeat([a[i][0] for a, i in zip(self.arms, self.choice)], self.m)

    def update(self, obs):
        z = obs['Y'].reshape(self.groups, self.m).sum(axis=1) * self.slots / self.m
        for arms, i, zj in zip(self.arms, self.choice, z):
            arms[i][1] += 1
            arms[i][2] += (zj - arms[i][2]) / arms[i][1]
            balls = sorted((p - self.radius(n) / self.lip, p + self.radius(n) / self.lip) for p, n, _ in arms)
            gaps, reach = [], 0.0
            for lo, hi in balls:
                if lo > reach:
                    gaps.append((reach, lo))
                reach = max(reach, hi)
            if reach < self.max_price:
                gaps.append((reach, self.max_price))
            if gaps:
                lo, hi = max(gaps, key=lambda g: g[1] - g[0])
                arms.append([(lo + hi) / 2, 0, 0.0])


def make_method(name, T, cfg, calib):
    slots, max_price, target = cfg['slots'], cfg['max_price'], calib['el_target']
    if name == 'oracle':
        return FixedSchedule(calib['oracle_prices'])
    if name == 'no_toll':
        return FixedSchedule(np.zeros(slots))
    if name == 'gradient_descent':
        return GradientDescent(slots, T, max_price, cfg['gd_eta0'], target)
    if name == 'constant_update':
        return ConstantUpdate(slots, max_price, cfg['const_step'], target)
    if name == 'pooled_ucb':
        return PooledUCB(slots, T, max_price, calib['sigma'])
    if name == 'pooled_zooming':
        return PooledZooming(slots, T, max_price, calib['sigma'], calib['lipschitz_price'])
    raise ValueError(f'unknown method {name}')


# ----------------------------------------------------------------------------- jobs

ENV = None


def init_worker(cfg, profile):
    global ENV
    gp.setParam('OutputFlag', 0)
    gp.setParam('Threads', 1)
    ENV = ManagedLaneEnv(cfg, profile)


def oracle_slot(args):
    slot, prices, draws = args
    rewards = np.empty((draws, len(prices)))
    for i in range(draws):
        vols, vots = ENV.draw(ORACLE_SEED + i, 0, slot)
        rewards[i] = [ENV.solve(vols, vots, p)['reward'] for p in prices]
    return rewards.mean(axis=0), rewards.var(axis=0, ddof=1)


def write_csv(df, path):
    tmp = path.with_suffix('.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)  # atomic: a CSV only exists once its job finished


def run_job(args):
    method, T, seed, cfg, calib, out_dir = args
    start = time.time()
    algo = make_method(method, T, cfg, calib)
    rows = []
    for day in range(T):
        prices = np.array(algo.schedule(), dtype=float)
        res = ENV.simulate_day(seed, day, prices)
        algo.update({'el_flow': res['el_flow'], 'Y': res['reward'] / cfg['slots']})
        rows.append({'day': day, **{k: v.mean() for k, v in res.items()}, 'mean_price': prices.mean()})
    out_dir = Path(out_dir)
    write_csv(pd.DataFrame({'slot': range(cfg['slots']), 'price': prices}),
              out_dir / f'T{T}_seed{seed}_final_prices.csv')
    write_csv(pd.DataFrame(rows), out_dir / f'T{T}_seed{seed}.csv')
    return method, T, seed, time.time() - start


def start_sioux_falls(run_dir):
    """Their main experiment, unchanged, run in its own folder (their code writes to ./Results)."""
    sf = run_dir / 'sioux_falls'
    if (sf / 'DONE').exists():
        return None
    sf.mkdir(parents=True, exist_ok=True)
    if not (sf / 'Locations').exists():
        (sf / 'Locations').symlink_to(TOLLS_DIR / 'Locations')
    (sf / 'gurobi.env').write_text('Threads 1\n')
    log = open(sf / 'log.txt', 'w')
    print(f'[sioux_falls] started, log: {sf / "log.txt"}')
    return subprocess.Popen(
        [sys.executable, '-u', '-c', 'from online_toll_eperiments import OnlineTollEperiments; OnlineTollEperiments()'],
        cwd=sf, env={**os.environ, 'PYTHONPATH': str(TOLLS_DIR)}, stdout=log, stderr=subprocess.STDOUT)


def finish_sioux_falls(proc, run_dir):
    if proc is None:
        return
    sf = run_dir / 'sioux_falls'
    code = proc.wait()
    (sf / 'Locations').unlink()
    (sf / 'gurobi.env').unlink()
    # Results get pushed: drop Gurobi's licence banner (user name, licence ID) from the log
    log = sf / 'log.txt'
    log.write_text(''.join(line for line in log.open() if not line.startswith(('Set parameter', 'Academic license'))))
    if code == 0:
        (sf / 'DONE').touch()
    print(f'[sioux_falls] finished with exit code {code}')


def run_info(cfg, profile_path):
    def git(*cmd):
        try:
            return subprocess.run(['git', *cmd], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    import matplotlib
    import scipy
    return {
        'started': datetime.now().isoformat(timespec='seconds'),
        'git_commit': git('rev-parse', 'HEAD'),
        'git_dirty': bool(git('status', '--porcelain')),
        'python': platform.python_version(),
        'platform': platform.platform(),
        'cpu_count': os.cpu_count(),
        'packages': {m.__name__: m.__version__ for m in (np, pd, scipy, matplotlib, gp)},
        'profile_sha256': hashlib.sha256(profile_path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-name', default=datetime.now().strftime('%Y-%m-%d_%H%M%S'))
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--quick', action='store_true', help='tiny settings for checking the pipeline end to end')
    parser.add_argument('--skip-sioux-falls', action='store_true')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=JSON', help='override a CONFIG entry')
    args = parser.parse_args()

    cfg = dict(CONFIG, **(QUICK if args.quick else {}))
    for item in args.set:
        key, value = item.split('=', 1)
        if key not in cfg:
            raise SystemExit(f'unknown config key {key}')
        try:
            cfg[key] = json.loads(value)
        except json.JSONDecodeError:
            cfg[key] = value

    run_dir = REPO_ROOT / 'Results' / args.run_name
    cfg_path = run_dir / 'config.json'
    if cfg_path.exists():
        saved = json.loads(cfg_path.read_text())
        if saved != cfg:
            raise SystemExit(f'{cfg_path} differs from this invocation; use a new --run-name')
        print(f'Resuming {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=True)
    profile, profile_path = load_profile(cfg)
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps(cfg, indent=2))
        (run_dir / 'run_info.json').write_text(json.dumps(run_info(cfg, profile_path), indent=2))
    print(f'Run folder: {run_dir}\nResume with: python experiments/managed_lane.py --run-name {args.run_name}'
          + (' --quick' if args.quick else '') + ''.join(f' --set {s}' for s in args.set))

    sioux_falls = None if args.skip_sioux_falls else start_sioux_falls(run_dir)

    with Pool(args.workers, initializer=init_worker, initargs=(cfg, profile)) as pool:
        # Oracle: Monte Carlo reward surface over (slot, price); calibrates sigma and L_p for our methods
        oracle_dir = run_dir / 'oracle'
        oracle_dir.mkdir(exist_ok=True)
        calib_path = oracle_dir / 'calibration.json'
        if not calib_path.exists():
            prices = np.linspace(0, cfg['max_price'], cfg['price_grid'])
            print(f"[oracle] {cfg['slots']} slots x {len(prices)} prices x {cfg['mc_draws']} draws")
            results = pool.map(oracle_slot, [(k, prices, cfg['mc_draws']) for k in range(cfg['slots'])])
            mean = np.array([r[0] for r in results])
            var = np.array([r[1] for r in results])
            write_csv(pd.DataFrame({'slot': np.repeat(np.arange(cfg['slots']), len(prices)),
                                    'price': np.tile(prices, cfg['slots']),
                                    'mean_reward': mean.ravel(), 'var_reward': var.ravel()}),
                      oracle_dir / 'surface.csv')
            env = ManagedLaneEnv(cfg, profile)
            calib = {
                'oracle_prices': prices[mean.argmax(axis=1)].tolist(),
                # White noise: Var(Y_k) = sigma^2 * Delta with Y_k = g_k * Delta, Delta = 1 / slots
                'sigma': float(np.sqrt(var.max() / cfg['slots'])),
                'lipschitz_price': float(np.abs(np.diff(mean, axis=1)).max() / (prices[1] - prices[0])),
                'el_target': float(env.el_target),
                'rho_crit': float(env.rho_crit),
            }
            calib_path.write_text(json.dumps(calib, indent=2))
        calib = json.loads(calib_path.read_text())

        jobs = [(m, T, s) for m in cfg['methods'] for T in cfg['Ts'] for s in range(cfg['seeds'])
                if m != 'oracle' or T == max(cfg['Ts'])]  # oracle never learns; one run per seed covers every T
        todo = [(m, T, s, cfg, calib, str(run_dir / m)) for m, T, s in sorted(jobs, key=lambda j: -j[1])
                if not (run_dir / m / f'T{T}_seed{s}.csv').exists()]
        for m in cfg['methods']:
            (run_dir / m).mkdir(exist_ok=True)
        print(f'[jobs] {len(todo)} of {len(jobs)} to run')
        for i, (m, T, s, secs) in enumerate(pool.imap_unordered(run_job, todo), 1):
            print(f'[jobs] {i}/{len(todo)} done: {m} T={T} seed={s} ({secs / 60:.1f} min)', flush=True)

    finish_sioux_falls(sioux_falls, run_dir)
    print(f'All done. Figures and tables: python experiments/report.py {run_dir.relative_to(REPO_ROOT)}')


if __name__ == '__main__':
    main()
