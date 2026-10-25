# CIFAR-10 Poisoning Grid — 5 Models x 3 Rates

Single dataset (CIFAR-10 — the standard benchmark across this
literature), one attack type (backdoor) at three poisoning rates
(0.1%, 0.5%, 1%), across five architectures. Code lives in GitHub;
Colab clones it fresh each session and downloads CIFAR-10 to LOCAL
disk (never Google Drive — that was the slow-I/O problem before).

## One-time setup (your machine, VS Code)

```bash
git init
git add .
git commit -m "initial project setup"
git remote add origin <your-empty-github-repo-url>
git push -u origin main
```

## Generate the 15 configs

```bash
python src/generate_configs.py
git add configs/experiments/
git commit -m "generate experiment configs"
git push
```

This writes `configs/experiments/cifar10_<model>_backdoor_<rate>.yaml`
for every (model, rate) pair — 5 models x 3 rates = 15 files.

## Running in Colab (yours or a friend's)

```python
!git clone <your-repo-url>
%cd <repo-folder-name>
!pip install -r requirements.txt

# smoke test first — always
!python src/run_experiment.py \
    --config configs/experiments/cifar10_resnet18_backdoor_0p01.yaml \
    --epochs 2 --data-root /content/data
```

Note `--data-root /content/data` — this is LOCAL Colab disk, not Drive.
CIFAR-10 downloads in under a minute here since it's 5 large files, not
thousands of small ones.

If the smoke test looks sane (train_acc climbing, ASR eventually well
above 10%), drop `--epochs 2` and run the full config:

```python
!python src/run_experiment.py \
    --config configs/experiments/cifar10_resnet18_backdoor_0p01.yaml \
    --data-root /content/data
```

## Resuming after a disconnect

Checkpoints are written locally too (`./checkpoints`, not Drive), which
means they do NOT survive a Colab disconnect — local disk is wiped
between sessions. This is the tradeoff for avoiding slow Drive I/O:
- A run that finishes in one session: no problem, nothing to resume.
- A run interrupted mid-way: it restarts from epoch 1 next session,
  since the checkpoint was local and is now gone.

Because of this, prefer `--max-minutes` set comfortably under a Colab
session length (170 is already in the generated configs) so a run is
likely to actually finish in one sitting. If a specific combo keeps
timing out, cut its `epochs` down for that YAML file instead.

## Collecting results from everyone

Every run writes exactly one `results/<run_id>_result.json`. After a
run finishes, commit and push just that file back:

```python
!git add results/
!git commit -m "add result: cifar10_resnet18_backdoor_0p01"
!git push
```

(Colab needs your GitHub credentials/token for push — a personal
access token used as the password when prompted is the simplest path.)

Once everyone's pushed, pull and aggregate:

```bash
git pull
python src/aggregate_results.py --results-dir ./results --out grid_summary.csv
```

One CSV, every model x rate combination's clean accuracy, attack
success rate, and detector AUROC/precision — sortable by any column.

## What each file does

- `src/registry_datasets.py` — CIFAR-10 only (trimmed from an earlier
  multi-dataset version — same interface, nothing else changed).
- `src/registry_models.py` — ResNet-10/18/50, VGG-19, DenseNet-161,
  each with small-image stem adaptation.
- `src/registry_poison.py` — label_flip / backdoor / clean_label attack
  generators (this grid only uses `backdoor`; point a config at
  `attack: label_flip` etc. to use the others).
- `src/generate_configs.py` — writes the 5x3 config grid; re-run any
  time to regenerate (edit RATES/epochs at the top first).
- `src/run_experiment.py` — the one script every run uses; config- and
  CLI-driven (CLI flags always win over the config file).
- `src/aggregate_results.py` — merges every `results/*_result.json`
  into one CSV.
