# Running the training on the cluster (Orfeo)

This guide describes how to run the 5-stage curriculum on the Orfeo GPU partition
(Slurm, 1x V100 per job) and how to bring the results back.

A single run (60,000 episodes) took about 60 minutes on one V100 in the previous
experiment; the job time limit is 2 hours. Training on a CPU-only machine is not
practical: a network update takes ~130 ms per step on 8 CPU cores, i.e. roughly a day
per run.

## What to run

We compare two algorithms over several seeds, so that results are not a single lucky run:

| Algorithm | `ALGO` value | Seeds |
|---|---|---|
| Standard DQN | `dqn` | 0, 1, 2 |
| Double DQN | `ddqn` | 0, 1, 2 |

That is 6 jobs, one GPU each. Each job writes to its own folder,
`models/<ALGO>_seed<SEED>/`, so runs never overwrite each other.

## 0. One-time prerequisites

- An SSH alias `orfeo` that works from your machine (`ssh orfeo` logs in). The sync
  scripts use `REMOTE_HOST` (default `orfeo`) and `REMOTE_DIR`
  (default `/orfeo/scratch/dssc/kj_mersimoski/RL_project`); override them with
  environment variables if yours differ.
- `uv` installed on the cluster (`uv --version`). The job runs `uv sync`, which needs
  network access to download the PyTorch CUDA 12.6 wheel (configured in `pyproject.toml`).
- Access to the `dssc` account and the `GPU` partition (already set in the script).

## 1. Push the code to the cluster

From the repository root on your machine (Git Bash / WSL / Linux / macOS):

```bash
git status --short            # make sure the working tree is what you want to run
bash scripts/sync_to_hpc.sh   # rsync of the repo (ignores models/, *.pt, runs/, ...)
```

The sync includes `src/data/inner_win_table.npy`, the solved table used by the teacher.
If it is ever missing on the cluster, the job rebuilds it automatically
(`uv run python src/inner_solver.py --build`, ~2 minutes), because without it the teacher
silently falls back to a strategy that does not always win.

## 2. (Optional) Quick sanity check on the cluster

```bash
ssh orfeo
cd /orfeo/scratch/dssc/kj_mersimoski/RL_project
uv sync
uv run pytest -q tests/test_dqn.py tests/test_curriculum.py tests/test_env.py
```

Do this on a compute node (`srun`) if the login node restricts heavy jobs.

## 3. Submit the jobs

The Slurm output directory must exist **before** submitting, otherwise the job fails
without leaving a log:

```bash
ssh orfeo
cd /orfeo/scratch/dssc/kj_mersimoski/RL_project
mkdir -p slurm_logs

# Double DQN
for SEED in 0 1 2; do ALGO=ddqn SEED=$SEED sbatch run_curriculum.sh; done

# Standard DQN
for SEED in 0 1 2; do ALGO=dqn SEED=$SEED sbatch run_curriculum.sh; done
```

`ALGO` and `SEED` are read from the environment (defaults: `ddqn`, `0`). To run a single
job: `ALGO=ddqn SEED=0 sbatch run_curriculum.sh`.

All jobs use the same hyperparameters (batch 256, gamma 0.90, buffer 100k trimmed to 20k
between stages, target update every 1000 steps, evaluation every 500 episodes on 100
games). Only the algorithm and the seed differ, which is what makes the comparison fair.

## 4. Monitor

```bash
squeue -u $USER                                  # job states
tail -f slurm_logs/order_chaos_v100_<JOBID>.out  # live log of one job
tail -f runs/<JOBID>/job.log                     # same log, plus GPU telemetry files in runs/<JOBID>/
```

Healthy logs show, per stage, lines such as
`[stage1_random] Ep 500/8000 | ... Eval Win Rate: ...` and
`==> New stage best model saved ...`. The curriculum stages are: random Chaos (8k episodes),
then heuristic Chaos with defence probability 0.35 (10k), 0.70 (12k), 0.85 (12k) and 1.00 (18k).
At the end each job prints a holdout table (100 games, greedy policy) for the agent and for
the theoretical teacher.

## 5. Bring the results back

When the jobs are finished (`squeue` is empty):

```bash
# from the repository root on your machine
bash scripts/sync_from_hpc.sh               # logs, CSV/JSON metrics and *_best.pt / *_final.pt
# or, metrics only (no weights):
bash scripts/sync_from_hpc.sh --logs-only
```

Results are in `models/<ALGO>_seed<SEED>/`:

| File | Content |
|---|---|
| `final_holdout_benchmark.json` | Final 100-game holdout: agent vs random Chaos, agent vs full-defence Chaos, and the teacher on both |
| `order_dqn_<stage>_metrics.csv` / `_metrics.json` / `_history.json` | Evaluation curve per stage (win rate, moves, reward, loss) |
| `order_dqn_<stage>_best.pt` / `_final.pt` | Checkpoints per stage. The next stage starts from the previous stage's **best** checkpoint |

A quick look at the headline numbers:

```bash
grep -H -A2 '"win_rate"' models/*/final_holdout_benchmark.json | grep -E 'json|win_rate'
```

## 6. What to compare

- **Headline metric:** `dqn_vs_heuristic_100.win_rate` in `final_holdout_benchmark.json`
  (100 games against the full-defence Chaos). Report mean and standard deviation over the
  three seeds for each algorithm.
- **Reference points:** the teacher should win 100% against both Chaos types. The earlier
  Standard DQN run reached 83% against the full-defence Chaos, but it used a different teacher
  reward (the teacher has since been fixed), so treat it as indicative only.
- **Learning curves:** per-stage `*_metrics.csv`, one line per seed.
- With 100 holdout games the standard error of a single win rate is about 4 percentage
  points; differences smaller than that between algorithms are not meaningful.

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Job disappears immediately, no log | `slurm_logs/` did not exist at submit time: `mkdir -p slurm_logs` and resubmit |
| `uv: command not found` in the log | Load/install `uv` on the cluster, or add it to `PATH` in `run_curriculum.sh` |
| `torch.cuda.is_available()` is `False` / CUDA errors | The wheel was not the CUDA build: check `uv run python -c "import torch; print(torch.__version__)"` ends with `+cu126`, and that the job got a GPU (`nvidia-smi` in the log) |
| Job killed at the 2 h limit | Raise `#SBATCH --time` in `run_curriculum.sh`. A run took ~60 min before; larger evaluation sets add a few minutes |
| `ALGO must be 'ddqn' or 'dqn'` | Typo in the `ALGO` variable |
| Warning about a missing `inner_win_table.npy` | Expected the first time if the file was not synced; the job rebuilds it. Re-sync to avoid the 2 min cost |

## 8. Optional: timing benchmark

`run_time_test.sh` profiles environment, action selection and network update speed on the
GPU (15-minute job): `sbatch run_time_test.sh`. The result is written to
`time_test_results.json` and brought back by `sync_from_hpc.sh`.
