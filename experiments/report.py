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
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS = {'oracle': 'Oracle', 'no_toll': 'No toll', 'gradient_descent': 'Gradient descent',
          'constant_update': 'Constant update', 'uniform_ucb': 'Uniform-UCB', 'interval_zooming': 'Interval-Zooming'}
# Each method keeps its colour in every figure; ORDER is also the order in tables and legends
COLORS = {'uniform_ucb': '#2a78d6', 'interval_zooming': '#eb6834', 'gradient_descent': '#1baf7a',
          'constant_update': '#eda100', 'no_toll': '#e87ba4'}
ORDER = ['uniform_ucb', 'interval_zooming', 'gradient_descent', 'constant_update', 'no_toll']
ORACLE_COLOR, GRID_COLOR = '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 10, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True})
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


def regret_table(summary, path):
    """Paper table: cumulative regret (standard error) per method and T, best method per T in bold."""
    Ts = sorted(summary['T'].unique())
    table = summary.set_index(['method', 'T'])
    lines = ['\\begin{tabular}{l' + 'r' * len(Ts) + '}', '\\toprule',
             'Method & ' + ' & '.join(f'$T={T:,}$'.replace(',', '{,}') for T in Ts) + ' \\\\', '\\midrule']
    best = summary[summary['method'].isin(ORDER)].groupby('T')['regret'].min()
    for method in ORDER:
        if method not in set(summary['method']):
            continue
        cells = []
        for T in Ts:
            regret, se = table.loc[(method, T), ['regret', 'regret_se']]
            cell = f'{regret:,.1f} ({se:.2f})'.replace(',', '{,}')
            cells.append(f'\\textbf{{{cell}}}' if regret == best[T] else cell)
        lines.append(LABELS[method] + ' & ' + ' & '.join(cells) + ' \\\\')
    path.write_text('\n'.join(lines + ['\\bottomrule', '\\end{tabular}', '']))


def to_latex(df, path):
    cols = list(df.columns)
    lines = ['\\begin{tabular}{' + 'l' * len(cols) + '}', ' & '.join(map(str, cols)).replace('_', '\\_') + ' \\\\', '\\hline']
    for _, row in df.iterrows():
        lines.append(' & '.join(f'{v:.4g}' if isinstance(v, float) else str(v).replace('_', '\\_') for v in row)
                     + ' \\\\')
    path.write_text('\n'.join(lines + ['\\end{tabular}', '']))


def plot_regret(summary, path):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for method in ORDER:
        df = summary[(summary['method'] == method) & (summary['regret'] > 0)]
        if not df.empty:
            ax.errorbar(df['T'], df['regret'], yerr=df['regret_se'], marker='o', ms=4.5, capsize=2, lw=1.8,
                        color=COLORS[method], label=LABELS[method])
    Ts = sorted(summary['T'].unique())
    ax.set(xscale='log', yscale='log', xlabel='Horizon $T$ (days)', ylabel='Cumulative regret')
    ax.set_xticks(Ts, [f'{T:,}' for T in Ts], minor=False)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:,g}'))
    ax.grid(color=GRID_COLOR, lw=0.6)
    ax.legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1), fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_schedules(run_dir, cfg, path):
    T = max(cfg['Ts'])
    hours = np.arange(cfg['slots'] + 1) * 24 / cfg['slots']
    prices = {m: pd.read_csv(p)['price'].to_numpy() for m in ['oracle', *ORDER]
              if (p := run_dir / m / f'T{T}_seed0_final_prices.csv').exists()}
    learners = [m for m in ORDER if m in prices and m != 'no_toll']
    rows = int(np.ceil(len(learners) / 2))
    # Demand profile is derived from the proprietary data, so it is only drawn if present locally
    profile = Path(cfg['profile']) if cfg['profile'] else REPO_ROOT / 'data' / 'derived' / 'demand_profile.csv'
    strip = int(profile.exists())
    fig = plt.figure(figsize=(6.5, 1 + 1.75 * rows + 1.1 * strip))
    grid = fig.add_gridspec(rows + strip, 2, height_ratios=[0.55] * strip + [1] * rows)
    ticks, ticklabels = [0, 6, 12, 18, 24], ['00:00', '06:00', '12:00', '18:00', '24:00']
    if strip:
        ax = fig.add_subplot(grid[0, :])
        demand = pd.read_csv(profile)['demand'].to_numpy().reshape(cfg['slots'], -1).mean(axis=1)
        ax.fill_between(hours, np.append(demand, demand[-1]) / demand.max(), step='post', color='#c3c2b7', lw=0)
        ax.set(xlim=(0, 24), ylim=(0, 1.05), ylabel='Relative\ndemand')
        ax.set_xticks(ticks, ticklabels)
    axes = []
    for i, method in enumerate(learners):
        ax = fig.add_subplot(grid[strip + i // 2, i % 2], sharey=axes[0] if axes else None)
        axes.append(ax)
        ax.step(hours, np.append(prices['oracle'], prices['oracle'][-1]), where='post', color=ORACLE_COLOR, lw=1,
                label=LABELS['oracle'])
        ax.step(hours, np.append(prices[method], prices[method][-1]), where='post', color=COLORS[method], lw=1.8,
                label=LABELS[method])
        ax.set(xlim=(0, 24), ylim=(0, cfg['max_price']))
        ax.set_xticks(ticks, ticklabels if i >= len(learners) - 2 else [])
        ax.set_ylabel('Price (\\$)' if i % 2 == 0 else '')
        ax.grid(axis='y', color=GRID_COLOR, lw=0.6)
        ax.legend(frameon=False, ncol=2, loc='lower left', bbox_to_anchor=(0, 0.98), fontsize=9, borderaxespad=0,
                  borderpad=0, handlelength=1.4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_surface(run_dir, cfg, path):
    surface = pd.read_csv(run_dir / 'oracle' / 'surface.csv')
    grid = surface.pivot(index='price', columns='slot', values='mean_reward')
    calib = json.loads((run_dir / 'oracle' / 'calibration.json').read_text())
    hours = np.arange(cfg['slots']) * 24 / cfg['slots']
    fig, ax = plt.subplots(figsize=(7, 4))
    mesh = ax.pcolormesh(hours, grid.index, grid.to_numpy(), shading='nearest', rasterized=True)
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
    fig, axes = plt.subplots(1, len(Ts), figsize=(3.25 * len(Ts), 3), squeeze=False)
    for ax, T in zip(axes[0], Ts):
        for method, df in sweep[sweep['T'] == T].groupby('method'):
            ax.errorbar(df['N'], df['regret'], yerr=df['regret_se'], marker='o', ms=4, capsize=2, lw=1.5,
                        color=COLORS[method], label=LABELS[method])
            tuned = run_dir / method / f'T{T}_seed0.csv'  # the main run's N, tuned to T as in the paper
            if tuned.exists():
                ax.axvline(pd.read_csv(tuned)['N'].iloc[0], color=COLORS[method], ls='--', lw=1)
        ax.set(xscale='log', yscale='log', xlabel='Number of intervals $N$', ylabel='Cumulative regret',
               title=f'$T = {T:,}$'.replace(',', '{,}'))
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:,g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
        ax.grid(color=GRID_COLOR, lw=0.6)
        ax.legend(frameon=False, fontsize=9)
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
        regret_table(summary, tab_dir / 'regret.tex')
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
