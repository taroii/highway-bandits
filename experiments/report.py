"""Figures and tables from a finished (or partial) run of experiments/run.py.

Writes to <run>/figures/ and <run>/tables/ and prints the tables. Only reads CSVs; runs no simulations.

Usage: python experiments/report.py Results/<run-name>
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS = {'oracle': 'Oracle', 'no_toll': 'No toll', 'gradient_descent': 'Gradient descent [AISTATS]',
          'constant_update': 'Constant update [AISTATS]', 'uniform_ucb': 'Uniform-UCB (ours)',
          'interval_zooming': 'Interval-Zooming (ours)',
          'interval_zooming_no_floor': 'Interval-Zooming, no noise floor (diagnostic)',
          'interval_zooming_agnostic': 'Interval-Zooming, $N = T^{1/4}$ (ours)'}
METRICS = ['reward', 'throughput', 'el_excess_density', 'revenue', 'vot_travel_time']


def load(run_dir, cfg):
    """Per-day results of every finished job, with the oracle's reward on the same days and seed."""
    oracle = {s: pd.read_csv(p) for s in range(cfg['seeds'])
              if (p := run_dir / 'oracle' / f"T{max(cfg['Ts'])}_seed{s}.csv").exists()}
    frames = []
    for method in cfg['methods']:
        for T in cfg['Ts']:
            for s in oracle:
                path = run_dir / method / f'T{T}_seed{s}.csv'
                if method == 'oracle' or not path.exists():
                    continue
                df = pd.read_csv(path)
                df['oracle_reward'] = oracle[s]['reward'].to_numpy()[:T]
                frames.append(df.assign(method=method, T=T, seed=s))
        if method == 'oracle':
            for s, od in oracle.items():
                for T in cfg['Ts']:
                    head = od.iloc[:T]
                    frames.append(head.assign(method='oracle', T=T, seed=s, oracle_reward=head['reward']))
    if not frames:
        return None
    days = pd.concat(frames, ignore_index=True)
    days['gap'] = days['oracle_reward'] - days['reward']
    return days


def summarise(days):
    per_job = days.groupby(['method', 'T', 'seed']).agg(
        regret=('gap', 'sum'), oracle_total=('oracle_reward', 'sum'), **{m: (m, 'mean') for m in METRICS})
    per_job['normalised_regret'] = per_job['regret'] / per_job['oracle_total']
    g = per_job.groupby(['method', 'T'])
    out = g.mean().drop(columns='oracle_total')
    out['regret_se'] = g['regret'].sem()
    out['seeds'] = g.size()
    out = out.reset_index()
    # Regret against a true optimum cannot be negative beyond noise: if it is, the oracle isn't optimal
    beaten = out[out['regret'] < -2 * out['regret_se']]
    if not beaten.empty:
        print('WARNING: oracle beaten, so its regret benchmark is invalid:\n',
              beaten[['method', 'T', 'regret', 'regret_se']].to_string(index=False))
    return out


def to_latex(df, path):
    cols = list(df.columns)
    lines = ['\\begin{tabular}{' + 'l' * len(cols) + '}', ' & '.join(map(str, cols)).replace('_', '\\_') + ' \\\\', '\\hline']
    for _, row in df.iterrows():
        lines.append(' & '.join(f'{v:.4g}' if isinstance(v, float) else str(v).replace('_', '\\_') for v in row)
                     + ' \\\\')
    path.write_text('\n'.join(lines + ['\\end{tabular}', '']))


def plot_regret(summary, path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for method, df in summary[summary['method'] != 'oracle'].groupby('method', sort=False):
        df = df[df['regret'] > 0]  # log axes: non-positive regret (beating the oracle) is not drawn
        ax.errorbar(df['T'], df['regret'], yerr=df['regret_se'], marker='o', capsize=3, label=LABELS.get(method, method))
    ours = summary[(summary['method'] == 'uniform_ucb') & (summary['regret'] > 0)]
    if not ours.empty:
        T0, R0 = ours['T'].iloc[0], ours['regret'].iloc[0]
        Ts = np.array(sorted(summary['T'].unique()))
        ax.plot(Ts, R0 * (Ts / T0) ** 0.75, 'k--', lw=1, label='$T^{3/4}$ reference')
    ax.set(xscale='log', yscale='log', xlabel='Days $T$', ylabel='Cumulative regret')
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_schedules(run_dir, cfg, path):
    T = max(cfg['Ts'])
    hours = np.arange(cfg['slots']) * 24 / cfg['slots']
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in cfg['methods']:
        p = run_dir / method / f'T{T}_seed0_final_prices.csv'
        if p.exists():
            style = {'color': 'k', 'lw': 2} if method == 'oracle' else {}
            ax.step(hours, pd.read_csv(p)['price'], where='post', label=LABELS.get(method, method), **style)
    ax.set(xlabel='Time of day (h)', ylabel='Express-lane price (\\$)', title=f'Final schedules, T={T}, seed 0')
    ax.legend(frameon=False, fontsize=8)
    # Demand profile is derived from the proprietary data, so it is only drawn if present locally
    profile = Path(cfg['profile']) if cfg['profile'] else REPO_ROOT / 'data' / 'derived' / 'demand_profile.csv'
    if profile.exists():
        demand = pd.read_csv(profile)['demand'].to_numpy().reshape(cfg['slots'], -1).mean(axis=1)
        ax2 = ax.twinx()
        ax2.fill_between(hours, demand, step='post', color='gray', alpha=0.15)
        ax2.set_ylabel('Relative demand')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_surface(run_dir, cfg, path):
    surface = pd.read_csv(run_dir / 'oracle' / 'surface.csv')
    grid = surface.pivot(index='price', columns='slot', values='mean_reward')
    calib = json.loads((run_dir / 'oracle' / 'calibration.json').read_text())
    hours = np.arange(cfg['slots']) * 24 / cfg['slots']
    fig, ax = plt.subplots(figsize=(7, 4))
    mesh = ax.pcolormesh(hours, grid.index, grid.to_numpy(), shading='nearest')
    ax.step(hours, calib['oracle_prices'], where='post', color='w', lw=1.5, label='Oracle price')
    fig.colorbar(mesh, label='Expected reward $g(\\theta, p)$')
    ax.set(xlabel='Time of day (h)', ylabel='Express-lane price (\\$)')
    ax.legend(frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def n_sweep(run_dir, cfg):
    """Regret of our methods at each fixed N, against the oracle on the same seeds and days."""
    rows = []
    for path in sorted((run_dir / 'n_sweep').glob('*/*.csv')):
        match = re.fullmatch(r'N(\d+)_T(\d+)_seed(\d+)', path.stem)
        oracle = match and run_dir / 'oracle' / f"T{max(cfg['Ts'])}_seed{match[3]}.csv"
        if not match or not oracle.exists():
            continue
        N, T, s = (int(x) for x in match.groups())
        gap = pd.read_csv(oracle)['reward'].to_numpy()[:T] - pd.read_csv(path)['reward'].to_numpy()
        rows.append({'method': path.parent.name, 'T': T, 'N': N, 'seed': s, 'regret': gap.sum()})
    if not rows:
        return None
    out = pd.DataFrame(rows).groupby(['method', 'T', 'N'])['regret'].agg(['mean', 'sem', 'size']).reset_index()
    return out.rename(columns={'mean': 'regret', 'sem': 'regret_se', 'size': 'seeds'})


def plot_n_sweep(sweep, run_dir, path):
    Ts = sorted(sweep['T'].unique())
    fig, axes = plt.subplots(1, len(Ts), figsize=(6 * len(Ts), 4.5), squeeze=False)
    for ax, T in zip(axes[0], Ts):
        for i, (method, df) in enumerate(sweep[sweep['T'] == T].groupby('method')):
            color = f'C{i}'
            ax.errorbar(df['N'], df['regret'], yerr=df['regret_se'], marker='o', capsize=3, color=color,
                        label=LABELS.get(method, method))
            tuned = run_dir / method / f'T{T}_seed0.csv'  # the main run's N, tuned to T as in the paper
            if tuned.exists():
                ax.axvline(pd.read_csv(tuned)['N'].iloc[0], color=color, ls='--', lw=1)
        ax.set(xscale='log', xlabel='Number of intervals $N$', ylabel='Cumulative regret',
               title=f'T={T} (dashed: tuned $N$)')
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def theory_slope(method, dz):
    """Exponent of T in the paper's tuned bounds (Corollaries cor:tuned and cor:zoom)."""
    return (dz + 2) / (dz + 3) if method == 'interval_zooming' else 3 / 4


def synthetic(run_dir):
    """Final regret of each synthetic job, and the fitted log-log slope per (shape, method)."""
    rows = []
    for path in sorted((run_dir / 'synthetic').glob('*/*/T*_seed*.csv')):
        df = pd.read_csv(path)
        T, s = (int(x) for x in re.fullmatch(r'T(\d+)_seed(\d+)', path.stem).groups())
        rows.append({'shape': path.parent.parent.name, 'method': path.parent.name, 'T': T, 'seed': s,
                     'N': df['N'].iloc[0], 'd_z': df['d_z'].iloc[0], 'regret': df['regret'].iloc[-1]})
    if not rows:
        return None, None
    g = pd.DataFrame(rows).groupby(['shape', 'method', 'T'])
    summary = g.agg(N=('N', 'first'), d_z=('d_z', 'first'), regret=('regret', 'mean'),
                    regret_se=('regret', 'sem'), seeds=('regret', 'size')).reset_index()
    slopes = []
    for (shape, method), df in summary.groupby(['shape', 'method']):
        df = df[df['regret'] > 0]
        if len(df) >= 2:
            slopes.append({'shape': shape, 'method': method, 'd_z': df['d_z'].iloc[0],
                           'fitted_slope': np.polyfit(np.log(df['T']), np.log(df['regret']), 1)[0],
                           'theory_slope': theory_slope(method, df['d_z'].iloc[0])})
    return summary, pd.DataFrame(slopes)


def plot_synthetic(summary, slopes, path):
    shapes = list(summary['shape'].unique())
    fig, axes = plt.subplots(1, len(shapes), figsize=(5 * len(shapes), 4.5), squeeze=False)
    for ax, shape in zip(axes[0], shapes):
        for method, df in summary[summary['shape'] == shape].groupby('method'):
            fit = slopes[(slopes['shape'] == shape) & (slopes['method'] == method)]
            label = LABELS.get(method, method) + (f" (slope {fit['fitted_slope'].iloc[0]:.2f}, "
                                                  f"theory {fit['theory_slope'].iloc[0]:.2f})" if len(fit) else '')
            ax.errorbar(df['T'], df['regret'], yerr=df['regret_se'], marker='o', capsize=3, label=label)
        ax.set(xscale='log', yscale='log', xlabel='Rounds $T$', ylabel='Cumulative regret',
               title=f"{shape} ($d_z$ = {summary[summary['shape'] == shape]['d_z'].iloc[0]:g})")
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save(table, name, tab_dir, index=True):
    table.to_csv(tab_dir / f'{name}.csv', index=index)
    to_latex(table.reset_index() if index else table, tab_dir / f'{name}.tex')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir
    cfg = json.loads((run_dir / 'config.json').read_text())
    fig_dir, tab_dir = run_dir / 'figures', run_dir / 'tables'
    fig_dir.mkdir(exist_ok=True)
    tab_dir.mkdir(exist_ok=True)
    pd.set_option('display.width', 200)

    # Managed-lane simulator
    days = load(run_dir, cfg)
    if days is not None:
        summary = summarise(days)
        summary.to_csv(tab_dir / 'summary.csv', index=False)
        regret = summary.pivot(index='method', columns='T', values='normalised_regret')
        final = summary[summary['T'] == summary['T'].max()].set_index('method')[METRICS + ['regret', 'seeds']]
        save(regret, 'normalised_regret', tab_dir)
        save(final, 'final_metrics', tab_dir)
        print('Normalised regret (cumulative regret / cumulative oracle reward), mean over seeds:\n', regret, '\n')
        print(f"Per-day averages at T={summary['T'].max()}:\n", final, '\n')
        plot_regret(summary, fig_dir / 'regret_vs_T.pdf')
        plot_schedules(run_dir, cfg, fig_dir / 'schedules.pdf')
        plot_surface(run_dir, cfg, fig_dir / 'reward_surface.pdf')

    # N sweep
    sweep = n_sweep(run_dir, cfg)
    if sweep is not None:
        save(sweep, 'n_sweep', tab_dir, index=False)
        print('N sweep (cumulative regret, mean over seeds):\n',
              sweep.pivot_table(index=['method', 'T'], columns='N', values='regret'), '\n')
        plot_n_sweep(sweep, run_dir, fig_dir / 'n_sweep.pdf')

    # Synthetic surfaces
    syn, slopes = synthetic(run_dir)
    if syn is not None:
        save(syn, 'synthetic_regret', tab_dir, index=False)
        if not slopes.empty:
            save(slopes, 'synthetic_slopes', tab_dir, index=False)
            print('Synthetic surfaces, log-log slope of regret vs T (fitted vs. the paper\'s bound):\n', slopes, '\n')
            plot_synthetic(syn, slopes, fig_dir / 'synthetic_rates.pdf')

    calib_path = run_dir / 'oracle' / 'calibration.json'
    if calib_path.exists():
        calib = json.loads(calib_path.read_text())
        print(f"Optimal mean daily reward is in [{calib.get('oracle_value_lower')}, {calib.get('oracle_value_upper')}]; "
              'the oracle row of the summary should be inside it\n')

    for chosen in sorted((run_dir / 'tuning').glob('*/chosen.json')):
        c = json.loads(chosen.read_text())
        print(f"Tuning, {chosen.parent.name}: {c['key']}={c['value']} "
              f"(mean daily reward per value: {c['mean_daily_reward']})\n")

    sf = run_dir / 'sioux_falls' / 'Results' / 'comparison.csv'
    if sf.exists():
        print('Sioux Falls reproduction (their metrics; their plot is sioux_falls/Results/comparison.png):\n',
              pd.read_csv(sf, index_col=0).set_index('T'))
    print(f'Wrote {fig_dir} and {tab_dir}')


if __name__ == '__main__':
    main()
