"""Figures and tables from a finished (or partial) run of experiments/managed_lane.py.

Writes to <run>/figures/ and <run>/tables/ and prints the tables. Only reads CSVs; runs no simulations.

Usage: python experiments/report.py Results/<run-name>
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS = {'oracle': 'Oracle', 'no_toll': 'No toll', 'gradient_descent': 'Gradient descent [AISTATS]',
          'constant_update': 'Constant update [AISTATS]', 'pooled_ucb': 'Pooled-UCB (ours)',
          'pooled_zooming': 'Pooled-Zooming (ours)'}
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
        raise SystemExit(f'No finished jobs in {run_dir}')
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
    return out.reset_index()


def to_latex(df, path):
    cols = list(df.columns)
    lines = ['\\begin{tabular}{' + 'l' * len(cols) + '}', ' & '.join(cols).replace('_', '\\_') + ' \\\\', '\\hline']
    for _, row in df.iterrows():
        lines.append(' & '.join(f'{v:.4g}' if isinstance(v, float) else str(v).replace('_', '\\_') for v in row)
                     + ' \\\\')
    path.write_text('\n'.join(lines + ['\\end{tabular}', '']))


def plot_regret(summary, path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for method, df in summary[summary['method'] != 'oracle'].groupby('method', sort=False):
        df = df[df['regret'] > 0]  # log axes: non-positive regret (beating the oracle) is not drawn
        ax.errorbar(df['T'], df['regret'], yerr=df['regret_se'], marker='o', capsize=3, label=LABELS.get(method, method))
    ours = summary[(summary['method'] == 'pooled_ucb') & (summary['regret'] > 0)]
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
    ax.set(xlabel='Time of day (h)', ylabel='Express-lane price', title=f'Final schedules, T={T}, seed 0')
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
    ax.set(xlabel='Time of day (h)', ylabel='Express-lane price')
    ax.legend(frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir
    cfg = json.loads((run_dir / 'config.json').read_text())
    fig_dir, tab_dir = run_dir / 'figures', run_dir / 'tables'
    fig_dir.mkdir(exist_ok=True)
    tab_dir.mkdir(exist_ok=True)

    summary = summarise(load(run_dir, cfg))
    summary.to_csv(tab_dir / 'summary.csv', index=False)

    regret = summary.pivot(index='method', columns='T', values='normalised_regret')
    final = summary[summary['T'] == summary['T'].max()].set_index('method')[METRICS + ['regret', 'seeds']]
    for name, table in [('normalised_regret', regret), ('final_metrics', final)]:
        table.to_csv(tab_dir / f'{name}.csv')
        to_latex(table.reset_index(), tab_dir / f'{name}.tex')
    pd.set_option('display.width', 200)
    print('Normalised regret (cumulative regret / cumulative oracle reward), mean over seeds:\n', regret, '\n')
    print(f"Per-day averages at T={summary['T'].max()}:\n", final, '\n')

    plot_regret(summary, fig_dir / 'regret_vs_T.pdf')
    plot_schedules(run_dir, cfg, fig_dir / 'schedules.pdf')
    plot_surface(run_dir, cfg, fig_dir / 'reward_surface.pdf')

    sf = run_dir / 'sioux_falls' / 'Results' / 'comparison.csv'
    if sf.exists():
        print('Sioux Falls reproduction (their metrics; their plot is sioux_falls/Results/comparison.png):\n',
              pd.read_csv(sf, index_col=0).set_index('T'))
    print(f'Wrote {fig_dir} and {tab_dir}')


if __name__ == '__main__':
    main()
