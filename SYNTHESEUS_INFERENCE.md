# graphretro self-contained inference

This directory layout adds a syntheseus-compatible inference pipeline on top
of the upstream graphretro repo, so it can be run end-to-end (and submitted
as a slurm array job) without depending on the `multiguide` package.

## Layout

```
graphretro/
├── (upstream model code, untouched)
├── syntheseus_inference/         # Python package added by this work
│   ├── __init__.py
│   ├── wrapper.py                # syntheseus BackwardReactionModel subclass
│   ├── dataset.py                # slim get_batch + parse_batch_to_reaction_data
│   └── output.py                 # get_rows_per_batch + save_df
├── scripts/
│   └── evaluate_in_batch.py      # Hydra entrypoint
├── configs/
│   ├── config.yaml
│   ├── single_step_model/graphretro.yaml
│   └── single_step_evaluation/uspto190.yaml
├── slurm/
│   ├── slurm_utils.py            # copied verbatim from multiguide
│   └── submit_inference.py       # array-job submitter
├── data/uspto190/processed/test.csv  # copied from retrosynthesis-dataset
├── models/                       # download from Drive (see below)
└── install_env.sh                # one-shot conda env build
```

## Setup

```
bash install_env.sh           # default: torch 1.7.0 + cudatoolkit 10.1
# bash install_env.sh --cpu   # CPU-only
conda activate graphretro
```

**Pretrained checkpoints already ship with this fork** under `models/`:

- `models/SingleEdit_10-02-2021--08-44-37/checkpoints/epoch_156.pt`  (un-typed edit predictor)
- `models/LGIndEmbed_18-02-2021--12-23-26/checkpoints/step_101951.pt`  (un-typed leaving-group selector)
- `models/SingleEdit_14-02-2021--19-26-20/checkpoints/step_144228.pt`  (typed edit predictor)
- `models/LGIndEmbedClassifier_18-04-2021--11-59-29/checkpoints/step_110701.pt`  (typed LG selector)

The default config uses the un-typed pair. For reaction-class-aware
inference, point `single_step_model.{edits_exp,edits_step,lg_exp,lg_step}`
in `configs/single_step_model/graphretro.yaml` at the typed pair.

If you need to re-fetch them, the upstream Drive is at
<https://drive.google.com/drive/folders/1u4N6jIsjfA0XxqtKRGAd5N3T-tL8wdKO>.

## Run inference locally

```
conda activate graphretro
python scripts/evaluate_in_batch.py \
    single_step_evaluation.start_idx=0 \
    single_step_evaluation.end_idx=2 \
    general.experiment_group=smoke \
    general.experiment_name=local_smoke
```

Output lands in `experiments/smoke/local_smoke/sampled_start0_end2.csv`.

## Submit a slurm array job (Puhti / Mahti)

Edit the parameters at the top of `slurm/submit_inference.py`
(`start_array_job`, `end_array_job`, `targets_per_job`), then:

```
python slurm/submit_inference.py --platform puhti
```

## Output schema

Same as RetroPrime — matches multiguide's existing single-step format so the
failures-analysis tooling reads it unchanged. See `RetroPrime/SYNTHESEUS_INFERENCE.md`.

## Architecture notes

- The wrapper (`syntheseus_inference/wrapper.py`) imports the model
  in-process via `seq_graph_retro.search.BeamSearch`. No subprocess
  bridging — graphretro provides a real Python API. Construction mirrors
  `scripts/eval/single_edit_lg.py:104-127`.
- `SEQ_GRAPH_RETRO` and `WANDB_MODE=offline` are baked into the conda env's
  activate hook by `install_env.sh`, so slurm jobs don't try to phone home.
- Python is bumped from the upstream environment.yml's 3.7.3 to **3.8** (the
  syntheseus minimum). torch 1.7.0 + cudatoolkit 10.1 has Py 3.8 builds on
  conda-forge, so no pip workaround is needed here.
