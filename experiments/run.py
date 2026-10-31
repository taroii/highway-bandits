"""All experiments for the paper. One command runs the whole pipeline and writes CSVs to Results/<run-name>/:
  1. demand profile (derived once from data/extracted, cached in data/derived/)
  2. oracle: expected reward surface over (time slot, price) by Monte Carlo, and calibration
  3. one-time tuning of baseline hyperparameters on held-out seeds, in tuning/
  4. managed-lane simulator: every (method, T, seed) job, one CSV per job in <method>/
  5. N sweep: our methods at fixed T for each number of intervals N, in n_sweep/<method>/
  6. synthetic surfaces with known zooming dimension (no traffic simulator), in synthetic/<shape>/<method>/
  7. a reproduction of the AISTATS 2023 Sioux Falls experiment in sioux_falls/

Re-running with the same --run-name skips everything that already finished.
Figures and tables: python experiments/report.py Results/<run-name>

Usage: python experiments/run.py [--run-name NAME] [--workers N] [--quick]
                                 [--skip-sioux-falls] [--set KEY=JSON ...]
"""
import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

# Shared machine: keep every process single-threaded (numerical libraries default to one thread per core).
# Must be set before numpy is imported.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
TOLLS_DIR = REPO_ROOT / 'online-tolls'
BPR_DIR = TOLLS_DIR / 'bpr-approximation'
sys.path.insert(0, str(BPR_DIR))
# Use a licence kept in the repo root if there is one; the Sioux Falls subprocess inherits this
if (REPO_ROOT / 'gurobi.lic').is_file():
    os.environ.setdefault('GRB_LICENSE_FILE', str(REPO_ROOT / 'gurobi.lic'))
import gurobipy as gp  # noqa: E402
from network import Network  # noqa: E402  (bpr-approximation version)
from user_equilibrium_with_tolls import UserEquilibriumWithTolls  # noqa: E402

# Defaults; override any entry with --set KEY=JSON. The rationale for each is in the project decisions log.
CONFIG = {
    # experiment grid
    'Ts': [100, 300, 1000, 3000, 10000],
    'seeds': 5,
    'methods': ['oracle', 'no_toll', 'gradient_descent', 'constant_update', 'uniform_ucb', 'interval_zooming'],
    # simulator
    'network': 'ManagedLane',        # folder in online-tolls/bpr-approximation/Locations
    'el_edge': 0,                    # tolled express-lane edge
    'gp_edge': 2,                    # general-purpose edge
    'slots': 96,                     # time-of-day slots per day; must divide 288 (5-minute data)
    'peak_demand': 8.5,              # mean demand at the busiest slot, in network flow units
    'vehicles_per_flow_unit': 28800,  # vehicles per day carried by one unit of flow (Poisson scale)
    'vot_groups': 10,
    'vot_seed': 5,
    'hours_per_length': 0.25,        # one network length unit is 15 minutes, so VOT ($/hour) * time is in dollars
    'max_price': 15.0,
    # reward: g = (EL + GP flow) / peak_demand - density_penalty * (EL density above rho_crit) / rho_crit
    'density_penalty': 1.0,
    'rho_crit': None,                # None: EL density at the BPR kink (1.1 * capacity)
    # oracle and calibration
    'price_grid': 61,
    'mc_draws': 1000,                # enough that the per-slot argmax is the benchmark, not Monte Carlo noise
    # ours: number of intervals N. None: tuned to T as in the paper (Uniform-UCB N = ceil(T^(1/4)),
    # Interval-Zooming N = ceil(T^(1/(zooming_dz + 3))); zooming_dz = 1 is the instance-agnostic T^(1/4))
    'intervals': None,
    'zooming_dz': 1.0,
    # their baselines
    'gd_eta0': 1.5,                  # gradient descent step size is gd_eta0 / sqrt(T); replaced if tuned below
    'const_step': 0.15,
    # One-time tuning of baseline hyperparameters, {method: {config key: grid}}: each value is run at
    # T = tune_T on tune_seeds held-out seeds, and the one with the highest mean reward is used for every job
    'tune': {'gradient_descent': {'gd_eta0': [0.15, 0.5, 1.5, 5.0, 15.0, 50.0, 150.0]}},
    'tune_T': 1000,
    'tune_seeds': 3,
    # N sweep on the managed-lane simulator: our methods at each T in n_sweep_Ts for every N in n_sweep_Ns
    # (None: every divisor of slots), on the first n_sweep_seeds evaluation seeds
    'n_sweep_methods': ['uniform_ucb', 'interval_zooming'],
    'n_sweep_Ts': [1000, 10000],
    'n_sweep_Ns': None,
    'n_sweep_seeds': 3,
    # Synthetic surfaces with known zooming dimension (see SYNTHETIC_SHAPES), Gaussian white noise
    'synthetic_shapes': ['sharp', 'quadratic', 'plateau'],
    'synthetic_methods': ['uniform_ucb', 'interval_zooming', 'interval_zooming_agnostic'],
    'synthetic_Ts': [1000, 3000, 10000, 30000, 100000],
    'synthetic_seeds': 5,
    'synthetic_sigma': 0.1,
    'synthetic_grid': 2000,          # time points per day for the integrals
    # demand profile CSV (columns slot, demand); None: derive from data/extracted
    'profile': None,
}
QUICK = {'Ts': [20, 50], 'seeds': 2, 'slots': 24, 'price_grid': 13, 'mc_draws': 4,
         'tune': {'gradient_descent': {'gd_eta0': [0.5, 5.0]}}, 'tune_T': 20, 'tune_seeds': 1,
         'n_sweep_Ts': [20], 'n_sweep_Ns': [1, 4, 24], 'n_sweep_seeds': 1,
         'synthetic_Ts': [100, 300], 'synthetic_seeds': 1, 'synthetic_grid': 240}
ORACLE_SEED = 10 ** 6  # Monte Carlo draws use seeds ORACLE_SEED + i, disjoint from run seeds
TUNE_SEED = 2 * 10 ** 6  # tuning runs use seeds TUNE_SEED + i, disjoint from run and oracle seeds


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
        # Read in chunks and keep only compact columns, so memory stays far below the raw file size
        parts = []
        for chunk in pd.read_csv(path, usecols=['data_timestamp', 'sensor_id', 'volume'], na_values='NULL',
                                 chunksize=1_000_000):
            ts = chunk['data_timestamp']
            part = pd.DataFrame({
                'sensor_id': chunk['sensor_id'].astype('int64'),
                'day': pd.to_datetime(ts.str.slice(0, 10)),
                'slot': (ts.str.slice(11, 13).astype(int) * 12 + ts.str.slice(14, 16).astype(int) // 5).astype('int16'),
                'volume': chunk['volume'].astype('float32')})
            parts.append(part[part['day'].dt.dayofweek < 5].dropna(subset=['volume']))
        df = pd.concat(parts, ignore_index=True)
        del parts

        # Valid sensor-day: exactly one reading for each of the 288 slots, nonzero mean,
        # <= 10% zero readings 06:00-21:00
        df['daytime'] = df['slot'].between(6 * 12, 21 * 12 - 1)
        df['daytime_zero'] = df['daytime'] & df['volume'].eq(0)
        s = df.groupby(['sensor_id', 'day']).agg(n=('volume', 'size'), slots=('slot', 'nunique'),
                                                 mean=('volume', 'mean'), zeros=('daytime_zero', 'sum'),
                                                 daytime=('daytime', 'sum'))
        valid = s[(s['n'] == 288) & (s['slots'] == 288) & (s['mean'] > 0) & (s['zeros'] <= 0.1 * s['daytime'])]
        if valid.empty:
            continue
        df = df.join(valid['mean'], on=['sensor_id', 'day'], how='inner')
        shapes.append((df['volume'] / df['mean']).groupby(df['slot']).median().reindex(range(288)))

    if not shapes:
        raise SystemExit(f'No valid sensor-days in the loop-detector CSVs under {data_dir}')
    profile = pd.concat(shapes, axis=1).mean(axis=1).interpolate().bfill().ffill()
    profile /= profile.mean()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(pd.DataFrame({'slot': range(288), 'demand': profile.values}), out_path)
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
        # Group mean VOTs as in their main experiment, $5-100 per hour: 5 + 95 * U(0, 1), per network time unit
        self.vot_means = cfg['hours_per_length'] * (5 + 95 * np.random.default_rng(cfg['vot_seed']).random(cfg['vot_groups']))
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


def num_intervals(slots, T, exponent):
    """N = ceil(T^exponent), rounded up to a divisor of the simulator's slots so that every interval is a
    whole number of slots. The slots are a finer measurement partition whose observations are pooled per
    interval, which the paper shows changes nothing (remark rem:pool). With 96 slots, N stays in
    [T^exponent, 2 T^exponent] as the tuned rates require."""
    target = min(int(np.ceil(T ** exponent)), slots)
    return min(d for d in range(target, slots + 1) if slots % d == 0)


def normalised_observations(Y, N):
    """Z_{t,k} = N Y_{t,k}, with Y_{t,k} the sum of the slot observations inside interval k."""
    return N * Y.reshape(N, -1).sum(axis=1)


class UniformUCB:
    """Ours: Uniform-UCB (Algorithm alg:pooled) on N intervals with K grid prices, delta = 1/T.
    Corollary cor:tuned sets N = K = ceil(T^(1/4))."""

    def __init__(self, slots, N, K, T, max_price, sigma):
        self.N, self.K = N, K
        self.grid = (np.arange(1, K + 1) - 0.5) * max_price / K
        self.width = sigma * np.sqrt(N)
        self.iota = np.log(2 * N * K * T * T)  # log(2NKT / delta)
        self.slots_per_interval, self.t = slots // N, 0
        self.n = np.zeros((N, K))
        self.mean = np.zeros((N, K))

    def schedule(self):
        if self.t < self.K:  # initial sweep
            self.choice = np.full(self.N, self.t)
        else:
            self.choice = np.argmax(self.mean + self.width * np.sqrt(2 * self.iota / self.n), axis=1)
        return np.repeat(self.grid[self.choice], self.slots_per_interval)

    def update(self, obs):
        z = normalised_observations(obs['Y'], self.N)
        idx = (np.arange(self.N), self.choice)
        self.n[idx] += 1
        self.mean[idx] += (z - self.mean[idx]) / self.n[idx]
        self.t += 1


class IntervalZooming:
    """Ours: Interval-Zooming (Algorithm alg:zooming) on N intervals, delta = 1/T. Corollary cor:zoom sets
    N = ceil(T^(1/(d_z + 3))); d_z = 1 gives the instance-agnostic N = ceil(T^(1/4))."""

    def __init__(self, slots, N, T, max_price, sigma, lip):
        self.N, self.slots_per_interval = N, slots // N
        self.max_price, self.lip = max_price, max(lip, 1e-12)
        self.iota = np.log(2 * N * T ** 3)  # bar-iota = log(2 N T^2 / delta)
        self.sigma_prime = sigma * np.sqrt(N)
        self.active = [[] for _ in range(N)]  # per interval: [price, plays, mean of Z]

    def radius(self, n):
        return np.inf if n == 0 else self.sigma_prime * np.sqrt(2 * self.iota / n)

    def uncovered(self, active):
        """A price outside every ball {q : L_p |p - q| <= r}: the midpoint of the widest uncovered gap."""
        gaps, reach = [], 0.0
        for lo, hi in sorted((p - self.radius(n) / self.lip, p + self.radius(n) / self.lip) for p, n, _ in active):
            if lo > reach:
                gaps.append((reach, lo))
            reach = max(reach, hi)
        if reach < self.max_price:
            gaps.append((reach, self.max_price))
        return None if not gaps else sum(max(gaps, key=lambda g: g[1] - g[0])) / 2

    def schedule(self):
        self.choice = []
        for active in self.active:
            q = self.uncovered(active)
            if q is not None:  # activation: at most one new price per interval per round
                active.append([q, 0, 0.0])
            self.choice.append(max(range(len(active)), key=lambda i: active[i][2] + 2 * self.radius(active[i][1])))
        return np.repeat([a[i][0] for a, i in zip(self.active, self.choice)], self.slots_per_interval)

    def update(self, obs):
        for active, i, z in zip(self.active, self.choice, normalised_observations(obs['Y'], self.N)):
            active[i][1] += 1
            active[i][2] += (z - active[i][2]) / active[i][1]


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
    if name == 'uniform_ucb':
        N = cfg['intervals'] or num_intervals(slots, T, 1 / 4)
        return UniformUCB(slots, N, int(np.ceil(T ** 0.25)), T, max_price, calib['sigma'])
    if name == 'interval_zooming':
        N = cfg['intervals'] or num_intervals(slots, T, 1 / (cfg['zooming_dz'] + 3))
        return IntervalZooming(slots, N, T, max_price, calib['sigma'], calib['lipschitz_price'])
    raise ValueError(f'unknown method {name}')


# ----------------------------------------------------------------------------- synthetic surfaces
# g(theta, p) = h(theta) - 0.5 * phi(|p - p*(theta)|) on [0, 1] x [0, 1], with a moving peak p*(theta).
# The gap profile phi sets the zooming dimension d_z of the interval averages mu_i (Definition def:zoom).

SYNTHETIC_SHAPES = {  # name: (phi, d_z, L_p)
    'sharp': (lambda x: x, 0.0, 0.5),                               # linear gap
    'quadratic': (lambda x: x ** 2, 0.5, 0.8),                      # quadratic gap (|x| <= 0.8)
    'plateau': (lambda x: np.maximum(0.0, x - 0.25), 1.0, 0.5),     # flat top of half-width 0.25
}


def synthetic_N(method, T, dz):
    exponent = 1 / (dz + 3) if method == 'interval_zooming' else 1 / 4
    return int(np.ceil(T ** exponent))


def run_synthetic_job(args):
    shape, method, T, seed, cfg, out_path = args
    start = time.time()
    phi, dz, lip = SYNTHETIC_SHAPES[shape]
    sigma, grid = cfg['synthetic_sigma'], cfg['synthetic_grid']
    theta = (np.arange(grid) + 0.5) / grid
    height = 0.7 + 0.2 * np.sin(2 * np.pi * theta)
    peak = 0.5 + 0.3 * np.sin(2 * np.pi * theta + 1)
    N = synthetic_N(method, T, dz)
    if method == 'uniform_ucb':
        algo = UniformUCB(N, N, int(np.ceil(T ** 0.25)), T, 1.0, sigma)
    else:  # interval_zooming (N tuned to the true d_z) or interval_zooming_agnostic (N = ceil(T^(1/4)))
        algo = IntervalZooming(N, N, T, 1.0, sigma, lip)
    piece = (theta * N).astype(int)  # interval of each time point
    points = np.bincount(piece, minlength=N)  # time points per interval (needs grid >= N)
    interval_mean = lambda values: np.bincount(piece, values, minlength=N) / points  # noqa: E731
    # Every interval has width 1/N, so integrals and day rewards weight intervals equally even when N does
    # not divide the grid. g*(theta) = h(theta), attained at p*(theta).
    r_star = interval_mean(height).mean()
    rng = np.random.default_rng([seed, list(SYNTHETIC_SHAPES).index(shape), T])
    checkpoints = {int(d) for d in np.geomspace(1, T, 60)} | {T}  # log-spaced days, always including T
    cumulative, rows = 0.0, []
    for day in range(1, T + 1):
        mu = interval_mean(height - 0.5 * phi(np.abs(np.asarray(algo.schedule())[piece] - peak)))
        # Y_i = integral of g over I_i + N(0, sigma^2 Delta_i), Delta_i = 1/N
        algo.update({'Y': mu / N + rng.normal(0, sigma / np.sqrt(N), N)})
        cumulative += r_star - mu.mean()
        if day in checkpoints:
            rows.append({'day': day, 'regret': cumulative})
    write_csv(pd.DataFrame(rows).assign(N=N, d_z=dz), Path(out_path))
    return f'synthetic {shape} {method} T={T} seed={seed}', time.time() - start


# ----------------------------------------------------------------------------- jobs

ENV = None


def die_with_parent():
    """Linux: have the kernel kill this process when its parent dies (crash, watchdog exit, kill -9), so no
    worker or subprocess is left running on the shared machine."""
    try:  # ctypes is imported at module level: this also runs in a forked child before exec
        ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, signal.SIGKILL)  # 1 = PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass


def init_worker(cfg, profile):
    global ENV
    die_with_parent()
    gp.setParam('OutputFlag', 0)
    gp.setParam('Threads', 1)
    ENV = ManagedLaneEnv(cfg, profile)


def oracle_slot(args):
    slot, prices, draws = args
    rewards = np.empty((draws, len(prices)))
    for i in range(draws):
        vols, vots = ENV.draw(ORACLE_SEED + i, 0, slot)
        rewards[i] = [ENV.solve(vols, vots, p)['reward'] for p in prices]
    # Bracket max_p E[g(p)]: the max of a sample mean overestimates it in expectation, and the value on one half
    # of the price chosen on the other half underestimates it
    half_a, half_b = rewards[:draws // 2].mean(axis=0), rewards[draws // 2:].mean(axis=0)
    upper = (half_a.max() + half_b.max()) / 2
    lower = (half_b[half_a.argmax()] + half_a[half_b.argmax()]) / 2
    return rewards.mean(axis=0), rewards.var(axis=0, ddof=1), lower, upper


def write_csv(df, path):
    tmp = path.with_suffix('.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)  # atomic: a CSV only exists once its job finished


def write_json(obj, path):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def run_job(args):
    method, T, seed, cfg, calib, out_path = args
    start = time.time()
    algo = make_method(method, T, cfg, calib)
    rows = []
    for day in range(T):
        prices = np.array(algo.schedule(), dtype=float)
        res = ENV.simulate_day(seed, day, prices)
        algo.update({'el_flow': res['el_flow'], 'Y': res['reward'] / cfg['slots']})
        rows.append({'day': day, **{k: v.mean() for k, v in res.items()}, 'mean_price': prices.mean()})
    out_path = Path(out_path)
    write_csv(pd.DataFrame({'slot': range(cfg['slots']), 'price': prices}),
              out_path.with_name(out_path.stem + '_final_prices.csv'))
    write_csv(pd.DataFrame(rows).assign(N=getattr(algo, 'N', cfg['slots'])), out_path)
    return f'{method} T={T} seed={seed} N={getattr(algo, "N", "-")}', time.time() - start


def call(job):
    fn, args = job
    return fn(args)


def run_all(pool, jobs, stage):
    todo = [(fn, args) for fn, args in jobs if not Path(args[-1]).exists()]
    print(f'[{stage}] {len(todo)} of {len(jobs)} jobs to run')
    for i, (label, secs) in enumerate(pool.imap_unordered(call, todo), 1):
        print(f'[{stage}] {i}/{len(todo)} done: {label} ({secs / 60:.1f} min)', flush=True)


LICENCE_LINE = re.compile(r'licen[cs]e|set parameter|registered to', re.IGNORECASE)


def start_sioux_falls(run_dir):
    """Their main experiment, unchanged, run in its own folder (their code writes to ./Results).
    Their per-T logs are appended to, so an unfinished earlier attempt is cleared and it starts clean."""
    sf = run_dir / 'sioux_falls'
    if (sf / 'DONE').exists():
        return None
    shutil.rmtree(sf, ignore_errors=True)
    sf.mkdir(parents=True)
    (sf / 'Locations').symlink_to(TOLLS_DIR / 'Locations')
    (sf / 'gurobi.env').write_text('Threads 1\n')
    print(f'[sioux_falls] started, log: {sf / "log.txt"}')
    with open(sf / 'log.txt', 'w') as log:
        return subprocess.Popen(
            [sys.executable, '-u', '-c', 'from online_toll_eperiments import OnlineTollEperiments; OnlineTollEperiments()'],
            cwd=sf, env={**os.environ, 'PYTHONPATH': str(TOLLS_DIR)}, stdout=log, stderr=subprocess.STDOUT,
            preexec_fn=die_with_parent)


def cleanup_sioux_falls(run_dir):
    """Remove the machine-specific helper files and the Gurobi licence banner (user name, licence ID) from
    the Sioux Falls folder, which gets pushed with Results."""
    sf = run_dir / 'sioux_falls'
    for name in ('Locations', 'gurobi.env'):
        if (sf / name).is_symlink() or (sf / name).exists():
            (sf / name).unlink()
    log = sf / 'log.txt'
    if log.exists():
        with open(log, errors='replace') as f:
            kept = [line for line in f if not LICENCE_LINE.search(line)]
        log.write_text(''.join(kept))


def finish_sioux_falls(proc, run_dir):
    code = proc.wait()
    cleanup_sioux_falls(run_dir)
    if code == 0:
        (run_dir / 'sioux_falls' / 'DONE').touch()
    print(f'[sioux_falls] finished with exit code {code}')


def stop_children(children, run_dir):
    """Kill the pool workers and the Sioux Falls subprocess (if running) and tidy the Sioux Falls folder."""
    if children.get('pool') is not None:
        for worker in list(children['pool']._pool):
            worker.kill()
    proc = children.get('sioux_falls')
    if proc is not None:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        cleanup_sioux_falls(run_dir)


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
        'git_dirty': bool(git('status', '--porcelain', '--', '.', ':(exclude)Results')),
        'python': platform.python_version(),
        'platform': platform.platform(),
        'cpu_count': os.cpu_count(),
        'packages': {m.__name__: m.__version__ for m in (np, pd, scipy, matplotlib, gp)},
        'profile_sha256': hashlib.sha256(profile_path.read_bytes()).hexdigest(),
    }


def available_memory_fraction():
    """MemAvailable / MemTotal from /proc/meminfo (Linux); None where that isn't available. This is the whole
    machine's memory; cgroup or container limits are not taken into account."""
    try:
        with open('/proc/meminfo') as f:
            info = dict(line.split(':', 1) for line in f)
        return int(info['MemAvailable'].split()[0]) / int(info['MemTotal'].split()[0])
    except (OSError, KeyError, ValueError):
        return None


def start_memory_watchdog(min_free, children, run_dir):
    """Stop the whole run if the machine's available memory drops below min_free (a fraction of RAM).
    Finished jobs are already saved, so rerunning the same command resumes where it stopped."""
    def watch():
        while True:
            free = available_memory_fraction()
            if free is not None and free < min_free:
                print(f'[memory] only {free:.0%} of RAM available (limit {min_free:.0%}): stopping. '
                      'Rerun the same command to resume.', flush=True)
                stop_children(children, run_dir)
                os._exit(3)
            time.sleep(10)
    if available_memory_fraction() is None:
        print('[memory] /proc/meminfo not available; memory watchdog disabled')
        return
    threading.Thread(target=watch, daemon=True).start()


def acquire_lock(run_dir):
    """Refuse to run two invocations on the same run folder at once. The lock is released when the process
    exits, however it exits."""
    import fcntl
    lock = open(run_dir / '.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f'{run_dir} is in use by another run')
    return lock


MANAGED_LANE_METHODS = {'oracle', 'no_toll', 'gradient_descent', 'constant_update', 'uniform_ucb', 'interval_zooming'}
SIMULATOR_KEYS = {'network', 'el_edge', 'gp_edge', 'slots', 'peak_demand', 'vehicles_per_flow_unit', 'vot_groups',
                  'vot_seed', 'hours_per_length', 'max_price', 'density_penalty', 'rho_crit', 'profile'}


def check_config(cfg):
    def fail(msg):
        raise SystemExit(f'Config error: {msg}')
    unknown = set(cfg['methods']) - MANAGED_LANE_METHODS
    if unknown:
        fail(f'unknown methods {unknown}')
    if 'oracle' not in cfg['methods']:
        fail("'oracle' must be in methods: regret for the simulator, tuning and N sweep is measured against it")
    if set(cfg['n_sweep_methods']) - {'uniform_ucb', 'interval_zooming'}:
        fail('n_sweep_methods must be uniform_ucb and/or interval_zooming')
    if set(cfg['synthetic_methods']) - {'uniform_ucb', 'interval_zooming', 'interval_zooming_agnostic'}:
        fail('synthetic_methods must be from uniform_ucb, interval_zooming, interval_zooming_agnostic')
    if set(cfg['synthetic_shapes']) - set(SYNTHETIC_SHAPES):
        fail(f'synthetic_shapes must be from {list(SYNTHETIC_SHAPES)}')
    if cfg['mc_draws'] < 2 or cfg['price_grid'] < 2:
        fail('mc_draws and price_grid must be at least 2 (variance and slope estimates)')
    divisors = [d for d in range(1, cfg['slots'] + 1) if cfg['slots'] % d == 0]
    for N in ([cfg['intervals']] if cfg['intervals'] else []) + (cfg['n_sweep_Ns'] or []):
        if N not in divisors:
            fail(f"N={N} must divide slots={cfg['slots']}")
    # N-sweep regret uses the oracle runs, which cover the evaluation seeds up to max(Ts)
    if cfg['n_sweep_seeds'] > cfg['seeds'] or max(cfg['n_sweep_Ts'], default=0) > max(cfg['Ts']):
        fail('n_sweep_seeds must be <= seeds and n_sweep_Ts <= max(Ts)')
    synthetic_Ns = [synthetic_N(m, T, SYNTHETIC_SHAPES[shape][1]) for shape in cfg['synthetic_shapes']
                    for m in cfg['synthetic_methods'] for T in cfg['synthetic_Ts']]
    if max(synthetic_Ns, default=0) > cfg['synthetic_grid']:
        fail('synthetic_grid must be at least the largest synthetic N')
    for method, grids in cfg['tune'].items():
        if method not in cfg['methods'] or len(grids) != 1:
            fail(f'tune: {method} must be in methods, with exactly one key to tune')
        # Workers build the simulator once from the run's settings, so only method settings can be tuned
        if set(grids) & SIMULATOR_KEYS:
            fail(f'tune: {set(grids) & SIMULATOR_KEYS} are simulator settings and cannot be tuned')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-name', default=datetime.now().strftime('%Y-%m-%d_%H%M%S'))
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help='parallel worker processes (default: half the cores, for shared machines)')
    parser.add_argument('--nice', type=int, default=10,
                        help='CPU priority increment for this run (default 10: other users take precedence)')
    parser.add_argument('--min-free-memory', type=float, default=0.15,
                        help='stop the run if available RAM falls below this fraction (Linux; default 0.15)')
    parser.add_argument('--quick', action='store_true',
                        help='tiny settings for checking the pipeline end to end (skips the Sioux Falls reproduction)')
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
    check_config(cfg)
    if args.nice:
        os.nice(args.nice)  # inherited by the workers and the Sioux Falls subprocess
    free = available_memory_fraction()
    if free is not None and free < args.min_free_memory:
        raise SystemExit(f'Only {free:.0%} of RAM available (limit {args.min_free_memory:.0%}); not starting')

    run_dir = REPO_ROOT / 'Results' / args.run_name
    cfg_path = run_dir / 'config.json'
    if cfg_path.exists():
        if json.loads(cfg_path.read_text()) != cfg:
            raise SystemExit(f'{cfg_path} differs from this invocation; use a new --run-name')
        print(f'Resuming {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(run_dir)  # noqa: F841  (held until the process exits)
    children = {'pool': None, 'sioux_falls': None}
    start_memory_watchdog(args.min_free_memory, children, run_dir)
    try:
        profile, profile_path = load_profile(cfg)
        if not cfg_path.exists():
            write_json(cfg, cfg_path)
            write_json(run_info(cfg, profile_path), run_dir / 'run_info.json')
        print(f'Run folder: {run_dir}\nResume with: '
              + shlex.join(['python', 'experiments/run.py', '--run-name', args.run_name]
                           + (['--quick'] if args.quick else []) + (['--set', *args.set] if args.set else [])))

        # Fail fast in this process (licence, network files) instead of in every worker: a Pool whose
        # initializer fails respawns workers forever. Then free Gurobi before forking.
        gp.setParam('OutputFlag', 0)
        env = ManagedLaneEnv(cfg, profile)
        el_target, rho_crit = float(env.el_target), float(env.rho_crit)
        env.ue.model.dispose()
        del env
        gp.disposeDefaultEnv()

        if args.skip_sioux_falls or args.quick:
            print('[sioux_falls] skipped')
        else:
            children['sioux_falls'] = start_sioux_falls(run_dir)

        with Pool(args.workers, initializer=init_worker, initargs=(cfg, profile)) as pool:
            children['pool'] = pool
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
                write_json({
                    'oracle_prices': prices[mean.argmax(axis=1)].tolist(),
                    # The optimal mean daily reward on the price grid lies between these, up to Monte Carlo error
                    'oracle_value_lower': float(np.mean([r[2] for r in results])),
                    'oracle_value_upper': float(np.mean([r[3] for r in results])),
                    # White noise: Var(Y_k) = sigma^2 * Delta with Y_k = g_k * Delta, Delta = 1 / slots
                    'sigma': float(np.sqrt(var.max() / cfg['slots'])),
                    'lipschitz_price': float(np.abs(np.diff(mean, axis=1)).max() / (prices[1] - prices[0])),
                    'el_target': el_target,
                    'rho_crit': rho_crit,
                }, calib_path)
            calib = json.loads(calib_path.read_text())

            # One-time tuning of baseline hyperparameters on held-out seeds
            job_cfg = dict(cfg)
            for method, grids in cfg['tune'].items():
                (key, grid), = grids.items()
                tune_dir = run_dir / 'tuning' / method
                chosen_path = tune_dir / 'chosen.json'
                if not chosen_path.exists():
                    tune_dir.mkdir(parents=True, exist_ok=True)
                    path = lambda v, s: tune_dir / f'{key}={v}_seed{s}.csv'  # noqa: E731
                    run_all(pool, [(run_job, (method, cfg['tune_T'], TUNE_SEED + s, dict(cfg, **{key: v}), calib,
                                              str(path(v, s)))) for v in grid for s in range(cfg['tune_seeds'])],
                            f'tuning {method}')
                    scores = {v: float(np.mean([pd.read_csv(path(v, s))['reward'].mean()
                                                for s in range(cfg['tune_seeds'])])) for v in grid}
                    best = max(grid, key=scores.get)
                    if best in (min(grid), max(grid)):
                        print(f'[tuning] warning: best {key}={best} is at the edge of the grid')
                    write_json({'key': key, 'value': best, 'mean_daily_reward': scores}, chosen_path)
                chosen = json.loads(chosen_path.read_text())
                job_cfg[chosen['key']] = chosen['value']
                print(f"[tuning] {method}: using {chosen['key']}={chosen['value']}")

            # Managed lane, N sweep and synthetic jobs share the pool; longest (largest T) first
            jobs = []
            for m in cfg['methods']:
                (run_dir / m).mkdir(exist_ok=True)
                for T in (cfg['Ts'] if m != 'oracle' else [max(cfg['Ts'])]):  # oracle never learns: one run covers every T
                    jobs += [(run_job, (m, T, s, job_cfg, calib, str(run_dir / m / f'T{T}_seed{s}.csv')))
                             for s in range(cfg['seeds'])]
            divisors = [d for d in range(1, cfg['slots'] + 1) if cfg['slots'] % d == 0]
            for m in cfg['n_sweep_methods']:
                (run_dir / 'n_sweep' / m).mkdir(parents=True, exist_ok=True)
                jobs += [(run_job, (m, T, s, dict(job_cfg, intervals=N), calib,
                                    str(run_dir / 'n_sweep' / m / f'N{N}_T{T}_seed{s}.csv')))
                         for T in cfg['n_sweep_Ts'] for N in (cfg['n_sweep_Ns'] or divisors)
                         for s in range(cfg['n_sweep_seeds'])]
            for shape in cfg['synthetic_shapes']:
                for m in cfg['synthetic_methods']:
                    if m == 'interval_zooming_agnostic' and SYNTHETIC_SHAPES[shape][1] == 1.0:
                        continue  # same N as interval_zooming when d_z = 1
                    (run_dir / 'synthetic' / shape / m).mkdir(parents=True, exist_ok=True)
                    jobs += [(run_synthetic_job, (shape, m, T, s, cfg,
                                                  str(run_dir / 'synthetic' / shape / m / f'T{T}_seed{s}.csv')))
                             for T in cfg['synthetic_Ts'] for s in range(cfg['synthetic_seeds'])]
            run_all(pool, sorted(jobs, key=lambda j: -j[1][2 if j[0] is run_synthetic_job else 1]), 'jobs')
            children['pool'] = None

        if children['sioux_falls'] is not None:
            finish_sioux_falls(children['sioux_falls'], run_dir)
            children['sioux_falls'] = None
    finally:
        # On any error or interrupt: leave nothing running and nothing machine-specific in Results
        stop_children(children, run_dir)
    print(f'All done. Figures and tables: python experiments/report.py {run_dir.relative_to(REPO_ROOT)}')


if __name__ == '__main__':
    main()
