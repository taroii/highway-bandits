# highway-bandits

Code for **"Learning a Lipschitz Reward Surface with Endogenous Partitions"**
(draft in `paper/main.tex`).

We model dynamic toll pricing on a managed lane as a Lipschitz semi-bandit over a
2-D reward surface `g(θ, p)` on `[0,1] × [0,M]` (time of day × price). Each round the
learner picks a partition of the day plus a price per interval, and sees one noisy
signal per interval whose variance scales with the interval width (white-noise model,
so total information per round is fixed at `σ⁻²` regardless of partition resolution).

Main results (theory only so far):
- **Pooled-UCB** — uniform partition, pools `m` neighboring intervals, UCB over a
  `K`-point price grid. Finite-sample bound splits into partition bias + price
  discretization + statistical error; tuned, it gives `Õ(T^{3/4})`, **independent of `N`**.
- **Pooled-Zooming** — adaptive price discretization; instance-dependent
  `Õ(T^{(d_z+2)/(d_z+3)})` in the zooming dimension `d_z`.
- Matching `Ω(T^{3/4})` lower bound is still a conjecture (Appendix B sketch).

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

## Status / TODO

No code yet. Planned, from the collaborator thread:

1. **Data** — `data/26-1763.zip` is not a zip; it's an S3 "Request has expired"
   error page (expired presigned link, 2026-08-29). Needs re-downloading from the
   Google Drive link Prof. Kim sent. The LA Metro ExpressLanes data stays within the
   core project team.
2. **Empirical regret check** — simulate a known `g`, run Pooled-UCB over increasing
   `T`, log-log plot regret vs `T` against the predicted `3/4` slope; also break out
   the three bound components separately.
3. **Test the white-noise assumption** — aggregate ExpressLanes observations at
   different interval lengths and see how residual variance scales with width; also
   check serial correlation (may matter more than sub-Gaussianity).
4. **Reward design** — Metro's objective is throughput / keeping density below
   critical, with revenue secondary. Pick a `g` (e.g. one-sided penalty above critical
   density, optionally weighted with revenue) that is operationally meaningful and
   still Lipschitz.
5. **Toll-update interval** — Metro uses 5 minutes; the theory says the right
   resolution is `T^{-1/4}` in normalized time (data-dependent, not intrinsic). Use
   the data to ask whether 5 min is right, and whether it should vary across the day.
6. **Extensions** — density as context, `g(θ, ρ, p)` (`ρ` is not fully exogenous, so
   this drifts past contextual bandits); adaptive pooling in *time*, not just price;
   lower-bound proof.

## System Specifications of Reported Runs

For all of the results we report in our paper, we use a dedicated Linux server with
the following specifications:

| **Component** | **Details** |
|--------------|-------------|
| **Machine** | Lenovo ThinkStation P3 Tower Gen 2 |
| **CPU** | Intel Core Ultra 9 285K (24 cores, up to 7.3 GHz) |
| **GPU** | NVIDIA RTX 4000 Ada Generation (Lovelace) |
| **RAM** | 48 GB |
| **Storage** | 1.86 TB NVMe SSD + 2×10.9 TB HDD (≈ 23.7 TB total) |
| **OS** | Ubuntu 24.04.2 LTS (Noble Numbat) |
| **Driver** | NVIDIA 580.126.09 |
