# Qsaria V1 with Chemprop and TabICL

This follow-up evaluates model families already available in Qsaria V1. It reuses the eligible test rows from the official AqSolDB, BBB_Martins, ABL1 and FXR partitions. It reports MAE for AqSolDB, positive-class AUROC for BBB_Martins, and RMSE for MoleculeACE, including RMSE on its annotated activity-cliff subsets. [TDC benchmark](README.md) · [MoleculeACE protocol](qsaria-v1-moleculeace.md)

Chemprop v2 takes standardized molecular graphs as input. It fits each official training partition with a directed message-passing network, 18 epochs, depth 3, a 64-unit hidden dimension and no hyperparameter search. Both the data ordering and PyTorch initialization use the recorded seed. The report compares this fixed configuration with the original LightGBM and Random Forest results on the full training partitions. [Chemprop v2](https://doi.org/10.1021/acs.jcim.5c02332)

TabICL v2 uses the official pretrained classifier and regressor. We compare it with an 80-tree LightGBM fit on the same at-most-800 training rows and the same binary Morgan fingerprints of radius 2 and 128 bits. The context sample is fixed at seed 42 and stratified for BBB_Martins. This comparison holds the training data and representation fixed within each endpoint. Its context size, feature dimension and single TabICL ensemble member bound CPU memory and runtime. The two model families use different computation budgets. [TabICL v2](https://arxiv.org/abs/2602.11139) · [Official checkpoints](https://huggingface.co/jingang/TabICL) · [Fixed settings](qsaria-v1-backends.toml)

Both evaluations use model seeds 42, 1701 and 2026. The JSON reports record the versioned source hashes, configuration, package versions, scoring cohort sizes, identity overlaps, run durations and prediction hashes. Test targets are used for scoring. The two reports describe fixed configurations at these resource budgets. Generalization to other hyperparameters or test partitions requires additional evaluation. [Chemprop results](results/qsaria-v1/chemprop.json) · [TabICL and matched LightGBM results](results/qsaria-v1/tabicl.json)

## Replay

Prepare the locked environment and the datasets using the main guide and the MoleculeACE download instructions. Run from the Qsaria repository root. [TDC data and environment](README.md) · [MoleculeACE data](qsaria-v1-moleculeace.md)

```bash
qsaria_publication="$PWD/.files/publication"
qsaria_weights="$qsaria_publication/checkpoints/tabicl"
mkdir -p "$qsaria_weights" "$qsaria_publication"/{mpl,tmp,xdg-cache}
for checkpoint in tabicl-regressor-v2-20260212.ckpt tabicl-classifier-v2-20260212.ckpt; do
  curl --fail --location --silent --show-error \
    "https://huggingface.co/jingang/TabICL/resolve/d700cf0/$checkpoint" \
    --output "$qsaria_weights/$checkpoint" || exit
done

export USE_S3=false OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR="$qsaria_publication/mpl"
export TMPDIR="$qsaria_publication/tmp" XDG_CACHE_HOME="$qsaria_publication/xdg-cache"
export HF_HUB_OFFLINE=1
if [ "$(uname -s)" = Darwin ]; then
  export DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/torch/lib${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
fi

for backend in chemprop tabicl; do
  .venv/bin/python -m scripts.benchmark_qsaria_v1_backends \
    --backend "$backend" \
    --archive "$qsaria_publication/datasets/tdc-admet-4426004.zip" \
    --moleculeace "$qsaria_publication/datasets/moleculeace-v3" \
    --checkpoints "$qsaria_weights" \
    --output "$qsaria_publication/replay-$backend" || exit
done
```

Choose fresh output directories. The runner verifies the archive, CSV and checkpoint digests before fitting any model. It writes the row IDs used for training and the scored predictions alongside the aggregate report for each backend.
