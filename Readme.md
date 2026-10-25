# SSL Data Poisoning

**Data Poisoning Attacks Using Self-Supervised Representation Learning** — a benchmark of backdoor poisoning attacks across datasets and architectures, with embedding-based poison detection.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red) ![License](https://img.shields.io/badge/license-MIT-green)

## Overview
Training data can be poisoned so that a model behaves normally on clean inputs but misclassifies any input carrying a hidden trigger. This project measures how vulnerable common architectures are to such **backdoor** attacks and how well poisoned samples can be found afterwards by analysing feature embeddings.

> **Status:** attack grid and baseline detectors are implemented. Detectors currently use a frozen ImageNet-pretrained encoder; the self-supervised encoder comparison is [planned / see `<link>`]. *(Edit this line to match your final work.)*

## Highlights
- 5 datasets × 5 architectures × 3 poisoning rates = **75 experiments**, each fully described by a YAML config.
- One entry point (`src/run_experiment.py`) for every experiment; CLI flags override config values.
- Resumable training with atomic checkpoints, mixed precision, and a per-run result JSON.
- Baseline detectors: kNN distance, Isolation Forest, k-means, kNN label agreement.
- Designed to run on free Colab/Kaggle GPUs; results are collected into one CSV.

## Experiment grid
| Axis | Values |
|---|---|
| Datasets | CIFAR-10, MNIST, GTSRB, Tiny-ImageNet-200, ImageNet subset (Imagenette) |
| Models | ResNet-10, ResNet-18, ResNet-50, VGG-19-BN, DenseNet-161 |
| Attack | Backdoor: 3×3 white patch (bottom-right), target class 0 |
| Poison rates | 0.1%, 0.5%, 1% of the training set |
| Metrics | Clean test accuracy, Attack Success Rate (ASR), detector AUROC / precision@k |

Training: SGD (lr 0.1, momentum 0.9, Nesterov, weight decay 5e-4), cosine schedule, batch 128, 30–60 epochs depending on model, seed 42.

## Repository structure
```
├── src/
│   ├── run_experiment.py      # train + evaluate one config
│   ├── registry_datasets.py   # dataset loaders / transforms
│   ├── registry_models.py     # architectures (+ small-image adaptations)
│   ├── registry_poison.py     # backdoor / label-flip / clean-label attacks
│   ├── embeddings.py          # frozen-encoder embedding extraction
│   ├── detectors.py           # poison detectors
│   ├── evaluate.py            # AUROC, precision@k, ...
│   ├── generate_configs.py    # writes the config grid
│   └── aggregate_results.py   # merges results into one CSV
├── configs/experiments/       # one YAML per experiment
├── results/                   # one JSON per finished run + grid_summary.csv
├── notebooks/                 # Colab runner
└── docs/                      # report, figures
```

## Quick start
```bash
git clone https://github.com/shivamm-verma/ssl-data-poisoning.git
cd ssl-data-poisoning
pip install -r requirements.txt

# smoke test (2 epochs)
python src/run_experiment.py --config configs/experiments/cifar10_resnet18_backdoor_1pct.yaml --epochs 2 --data-root ./data

# full run
python src/run_experiment.py --config configs/experiments/cifar10_resnet18_backdoor_1pct.yaml --data-root ./data

# aggregate all finished runs
python src/aggregate_results.py --results-dir results --out results/grid_summary.csv
```
CIFAR-10, MNIST and GTSRB download automatically. Tiny-ImageNet and Imagenette must be available under `--data-root` (see `registry_datasets.py`). Use `--run-detectors` to also run the detector suite after training.

**Colab/Kaggle:** clone the repo, install requirements, point `--data-root` at local disk (not Drive), and push each `results/*_result.json` as it finishes. Checkpoints are local and do not survive a disconnect.

## Results (preliminary)
Mean Attack Success Rate across architectures (from `results/grid_summary.csv`):

| Dataset | 0.1% | 0.5% | 1% |
|---|---|---|---|
| MNIST | 0.94 | 1.00 | 1.00 |
| CIFAR-10 | 0.01 | 0.89 | 0.92 |
| GTSRB | 0.00 | 0.34 | 0.43 |
| ImageNet subset | 0.21 | 0.21 | 0.21 |

- Higher poison rates give much higher ASR on most datasets, with a sharp threshold on CIFAR-10 between 0.1% and 0.5%.
- Label-agnostic detectors (kNN, Isolation Forest, k-means) stay near chance (AUROC ≈ 0.48–0.50), while the label-aware **kNN label agreement** detector reaches AUROC ≈ 0.98 (15 runs with detectors enabled).
- Tiny-ImageNet runs are still being completed. Update this table from the final CSV.

## Limitations
- Tiny-ImageNet evaluation uses the training split as the test set (no held-out labels in the loader).
- Imagenette stands in for full ImageNet-1k.
- Some runs use per-model batch sizes/schedules chosen to fit free-tier GPUs.
- Single seed per configuration; no confidence intervals yet.

## Authors
Shivam Verma · Nikhil Kumar · Shivam Singh · Simerjeet Singh — *[Maharaja Surajmal Institute of Technology, Department of Computer Science and Engineering, 2027]*

## License
MIT — see `LICENSE`.

## Citation
```
@misc{ssl-data-poisoning,
  title  = {Data Poisoning Attacks Using Self-Supervised Representation Learning},
  author = {Verma, Shivam and Kumar, Nikhil and Singh, Simerjeet},
  year   = {2026},
  url    = {https://github.com/shivamm-verma/ssl-data-poisoning}
}
```