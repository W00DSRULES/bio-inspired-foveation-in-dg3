# Bio-inspired foveation in DeepGaze III

Bachelor's thesis, Imer Itez, Goethe University Frankfurt. The thesis is [`thesis-tex/main.pdf`](thesis-tex/main.pdf); its LaTeX source is in [`thesis-tex/`](thesis-tex/).

[DeepGaze III](https://github.com/matthias-k/DeepGaze) (Kümmerer, Bethge & Wallis, 2022) predicts where a person looks next in an image. It sees the whole image at one resolution. A human eye does not: only a small region around the current gaze is sharp, and resolution falls off with distance from it. This thesis puts that constraint into the model's input and measures what it changes.

**Mechanism.** At every fixation the input image is re-foveated around the current gaze point with the Geisler–Perry falloff (a Gaussian-pyramid blend that is sharp at gaze and blurred, but never removed, in the periphery). The read-out heads of DeepGaze III are then fine-tuned on that input; the DenseNet backbone stays frozen.

**Design.** Seven arms on MIT1003 (Judd et al., 2009), image-stratified 10-fold cross-validation, every arm trained and scored on the same folds:

| arm | input |
|---|---|
| sharp control | the unmodified image |
| gaze-contingent @ 40, 20, 10 | foveated around the current fixation; the number is the foveal cutoff in cycles per degree (40 is the human value, lower is a stronger blur) |
| fixed-centre @ 40, 20, 10 | foveated around the image centre, so the blur is present but does not follow the gaze |

Every metric is per fixation on the held-out folds: log-likelihood, information gain over the centerbias (IG, bits), NSS and AUC. The primary contrast is sharp control vs gaze-contingent @ 40, reported as a fold-paired difference with two standard errors. The scoring protocol is the one of the DeepGaze III paper (the central starting fixation is kept as history and every free fixation is scored, 104,171 fixations).

**Result.** On the test folds the gaze-contingent @ 40 arm scores $1.5781$ bits IG against $1.5742$ for the sharp control, a fold-paired difference of $+0.0039 \pm 0.0032$ bits per fixation. Stronger blur costs: $-0.0059 \pm 0.0053$ at 20 and $-0.0365 \pm 0.0083$ at 10 cycles per degree. The full table is [`results/foveation_mit1003_initial/results_tables.tex`](results/foveation_mit1003_initial/results_tables.tex), generated from [`results/foveation_mit1003_initial/test/table.json`](results/foveation_mit1003_initial/test/table.json); the pretrained read-out before any fine-tuning, scored the same way over the whole dataset, is [`results/foveation_mit1003_initial/pretrained_epoch0/baseline.json`](results/foveation_mit1003_initial/pretrained_epoch0/baseline.json). The thesis reads and interprets these numbers.

## Setup

Python 3.11 and [`uv`](https://docs.astral.sh/uv/). NumPy must stay below 2.0 (`pysaliency` links against the 1.x C API); the versions are pinned in `uv.lock`.

```bash
uv sync                                        # .venv/ with every pinned dependency
.venv/bin/python scripts/fetch_mit1003.py --with-initial   # MIT1003 into data/ (needs MATLAB or Octave for the extraction step)
.venv/bin/python scripts/smoke_test.py         # one forward pass
```

The centerbias and the pretrained weights download on first use. `--with-initial` builds the paper-protocol variant of the dataset (`data/mit1003/MIT1003_initial_fix_consistent/`); training and evaluation select it with `--dataset-variant initial`.

Everything runs on CUDA, Apple MPS or CPU (`src/tez_deepgaze/device.py`). Forward-only demos on a few images are fine on a laptop; training and full-dataset evaluation ran on an NVIDIA cluster through the SLURM scripts in [`scripts/slurm/`](scripts/slurm/).

## Layout

```
src/tez_deepgaze/     library: foveation, forward-pass primitives, evaluator, training loop, CV split
scripts/              entry points: data fetch, training and evaluation drivers, every figure generator
scripts/slurm/        SLURM templates for training, evaluation and the test suite (README inside)
tests/                pytest suite; `-m "not heavy"` is the laptop-safe subset
notebooks/            interactive companions and figure reproductions (README inside)
results/              committed artefacts: tables (JSON + Markdown), figures, the CV split, the centerbias
thesis-tex/           LaTeX source of the thesis and the built PDF; figures are read from results/
```

The library modules that matter most: `foveate_input.py` (the Geisler–Perry foveation), `instrument.py` (log-density forward pass, scanpath sampling, checkpoint bundles), `evaluate.py` (the distribution-level evaluator every table comes from), `foveated_train.py` (read-out fine-tuning), `cv_split.py` (the 10-fold image split in `results/cv_splits/`).

## Reproducing the tables and figures

Every committed figure and table has a generator in `scripts/`. These need the dataset in `data/` but no model:

```bash
.venv/bin/python scripts/make_results_figures.py     # results_tables.tex, fold_paired, training_curves, … from the committed JSON
.venv/bin/python scripts/make_protocol_figures.py    # method figures: pyramid, strength grid, saccade coverage, stratified panels
.venv/bin/python scripts/entropy_panel.py            # per-image entropy panel
```

These also run the pretrained model, laptop-sized:

```bash
.venv/bin/python scripts/make_background_figures.py  # background chapter figures
.venv/bin/python scripts/demo_consensus_panel.py     # consensus panels
.venv/bin/python scripts/demo_priority_evolution.py --stim-idx 91
.venv/bin/python scripts/demo_foveated_scanpath.py   # gaze-contingent foveation following a scanpath
```

Training the seven arms, scoring them on the held-out folds, the stratified analysis and the 1003-image diagnostic are cluster jobs; `scripts/slurm/README.md` lists them in order. [`notebooks/README.md`](notebooks/README.md) maps each committed figure to what regenerates it.

## Tests

```bash
uv run ruff check src tests scripts
uv run pytest -m "not heavy"      # laptop: no dataset, no weights
sbatch scripts/slurm/pytest.sbatch   # cluster: the full suite
```

## References

- Kümmerer, M., Bethge, M., & Wallis, T. S. A. (2022). DeepGaze III: Modeling free-viewing human scanpaths with deep learning. *Journal of Vision*, 22(5):7.
- Judd, T., Ehinger, K., Durand, F., & Torralba, A. (2009). Learning to predict where humans look. *ICCV*.
- Geisler, W. S., & Perry, J. S. (1998). A real-time foveated multiresolution system for low-bandwidth video communication. *SPIE Human Vision and Electronic Imaging*.
