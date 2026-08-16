# Notebooks

Interactive companions to the README. Each notebook is **standalone** — open it,
edit the `PARAMETERS` cell at the top, re-run all cells, see results inline.

There are two kinds, and they differ on purpose.

**01–03 — exploration.** Meant to be read top-to-bottom and modified. They
reimplement the plotting inline rather than importing it, so you never have to
jump into another file to follow what is happening.

| # | Notebook | What it does |
|---|---|---|
| 01 | [`01_scanpath_explorer.ipynb`](01_scanpath_explorer.ipynb) | Pick one MIT1003 image. See DG3's priority map, one human's actual scanpath, and an AI scanpath sampled from DG3 — all side-by-side. Tweak `STIM_IDX`, `SUBJECT_IDX`, `N_FIX`, `SEED`. |
| 02 | [`02_consensus_panel.ipynb`](02_consensus_panel.ipynb) | 3-panel composite: every human scanpath, the ≥75 % / ≥90 % consensus regions (colourblind-safe), and DG3's probability map. Tweak `RADIUS`. Its last cell is a reproduction cell rather than an exploration one — it rebuilds the six stacks ch04 includes. |
| 03 | [`03_priority_map_evolution.ipynb`](03_priority_map_evolution.ipynb) | Sample one AI scanpath step-by-step and snapshot the priority map at each chosen fixation count, so you can watch DG3's "where to look next" expectation move as the history grows. |

**04–07 — figure reproduction.** Thin drivers that call the same code the
committed figure came from, via a `run_script` helper. They deliberately do *not*
reimplement the plotting: a reproduction notebook that carries its own copy of
the drawing code can silently disagree with what is in `results/` and in the
thesis.

| # | Notebook | Reproduces | Cost |
|---|---|---|---|
| 04 | [`04_foveation_figures.ipynb`](04_foveation_figures.ipynb) | every entry in `make_protocol_figures.FIGURES` — four of them in ch03, two in ch04, one in the appendix, the rest supporting material. It reads the script's own `FIGURE_DIR`, so a figure that writes outside `foveation_mit1003/figs/` is still found | no model; ~30 s on GPU, minutes on CPU |
| 05 | [`05_gaze_contingent_demo.ipynb`](05_gaze_contingent_demo.ipynb) | `foveation_mit1003/gaze_contingent_demo_stim0091.png` and `next_pixel_check_stim0091.png` — README figures, not thesis figures | DG3 forward-only; ~5 min |
| 06 | [`06_intro_scanpath.ipynb`](06_intro_scanpath.ipynb) | `foveation_mit1003/intro_scanpath_stim0091.png` (ch01) | no model; seconds |
| 07 | [`07_per_image_diagnostic.ipynb`](07_per_image_diagnostic.ipynb) | §1 redraws `per_image_diagnostic_initial_1003/scatter.png` — the figure ch04 includes — from its committed JSON. §2 reruns the pipeline on a laptop-sized sample under the same protocol, into a gitignored scratch directory | §1 no model, seconds · §2 DG3 over 50 stimuli, **30 min+** |

## Every committed figure, and what regenerates it

The **in** column is the chapter that `\includegraphics` it. A dash means no
chapter includes the figure.

| figure | in | reproduce with |
|---|---|---|
| `background/*.png` | ch02 | `scripts/make_background_figures.py` — `dg3_composition` is the one that needs the pretrained weights: two forward passes on one stimulus, laptop-sized |
| `foveation_mit1003/figs/` — `gp_pyramid`, `gp_strength_grid`, `foveation_strength`, `saccade_coverage` | ch03 | notebook 04 · `scripts/make_protocol_figures.py` |
| `foveation_mit1003_initial/figs/` — `stratified_amplitude`, `stratified_fixation_index` | ch04 | notebook 04 · `scripts/make_protocol_figures.py` — these read the primary run, so they land in the initial tree |
| `foveation_mit1003_initial/figs/training_curves.png` | ch03 | `scripts/make_results_figures.py` — the one figure in that script that reads the dataset, for the epoch-0 fixation weights |
| `foveation_mit1003_initial/figs/fold_paired.png` | ch04 | `scripts/make_results_figures.py`, which also writes `results_tables.tex`, `per_image_dispersion.json` and `grouping_intervals.json` |
| `foveation_mit1003/gaze_contingent_demo_stim0091.png` | — | notebook 05 · `scripts/demo_foveated_scanpath.py`. A README figure |
| `foveation_mit1003/next_pixel_check_stim0091.png` | — | notebook 05 · `scripts/next_pixel_check.py`. A README figure |
| `foveation_mit1003/intro_scanpath_stim0091.png` | ch01 | notebook 06 · `scripts/demo_intro_scanpath.py` |
| `per_image_diagnostic_initial_1003/scatter.png` | ch04 | notebook 07 §1 · `scripts/per_image_diagnostic.py --replot --out results/per_image_diagnostic_initial_1003` redraws it from the committed JSON on a laptop; computing it (`--n-stim 1003 --dataset-variant initial`) is **cluster only** |
| `consensus_panels/consensus_stim*.png` — the six ch04 stacks (128 704 528 750 639 489) | ch04 | notebook 02, last cell · `scripts/demo_consensus_panel.py --stim-indices 128 704 528 750 639 489 --metrics-from results/per_image_diagnostic_initial_1003/diagnostic.json --titles-first-only --page-frac 0.5`, which puts each stimulus's IG/NSS/AUC in its title and the A/B/C headings on the first only |
| `consensus_panels/consensus_method.png` | appendix | notebook 04 · `scripts/make_protocol_figures.py --only consensus_method` — it writes here rather than into `figs/` |
| `foveation_mit1003_initial/figs/entropy_panel.png` | ch03 | `scripts/entropy_panel.py` — reads the committed `diagnostic.json` and the fixation data; no model, no GPU |
| `foveation_mit1003_initial/stratified/index_profile.json` (the fixation-index paragraph of ch04 §Where the cost falls) | ch04 | `scripts/index_profile.py` — aggregates the gitignored per-fixation dumps, as `collect_training_curves.py` does for the checkpoints |
| `demos/priority_evolution/priority_evolution_stim0091.png` | ch02 | `scripts/demo_priority_evolution.py --stim-idx 91` (ranks 1 3 4) — feeds subject 0's recorded scanpath, nothing sampled (notebook 03 is the sampling playground and does not draw the thesis figure) |

### Thesis figures with a generator but no notebook

The notebooks cover most of what a reader would want to re-run, not all of it.
These are included by a chapter and have a committed generator, and running that
script is the only route:

- `background/*.png` (ch02) — `make_background_figures.py`
- `demos/priority_evolution/priority_evolution_stim0091.png` (ch02) — `demo_priority_evolution.py`
- `entropy_panel.png` (ch03) — `entropy_panel.py`
- `training_curves.png` (ch03) and `fold_paired.png` (ch04) — `make_results_figures.py`

None of them is expensive; `make_results_figures.py` reads only committed JSON.
They are listed so the gap is visible rather than assumed away.

The cluster-only entries read per-(arm, fold) trained checkpoints under
`results/foveation_mit1003/ckpts/`, which are gitignored and live on Goethe-NHR.
No laptop notebook can produce them; `scripts/slurm/README.md` has the job
sequence.

`make_results_figures.py` reads only committed result JSONs, so it needs neither
the model nor the corpus. The one input it cannot derive is `training_curves.json`,
aggregated from the 840 per-epoch `metrics.json` files in the checkpoint tree by
`scripts/collect_training_curves.py`; that aggregate is committed so the figure
survives the tree it came from.

**Sampling is stochastic.** Notebooks 03 and 05 sample scanpaths, and float
jitter on MPS is enough to change a draw at a fixed seed. A rerun reproduces the
figure's *structure*, not its exact path — which is why each figure writes a JSON
sidecar recording the draw it actually made.

## Setup

```bash
uv sync                                # installs pinned deps into .venv/, jupyter included
.venv/bin/jupyter lab notebooks/
```

Alternatively open the `.ipynb` files in **VSCode** — its built-in notebook
support uses the project `.venv` automatically and does not require `jupyter`
to be installed.

The first cell of each notebook locates the repository root by walking up until
it finds `pyproject.toml`, then adds `src/` to `sys.path`. So you can launch
Jupyter from any directory and the imports still resolve.

## Conventions

- **Outputs are stripped before commit.** Researchers should re-run the cells
  locally to populate results; the committed `.ipynb` files contain only code
  and markdown, so git diffs stay readable. To strip outputs manually:
  ```bash
  jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
  ```
  Or install [`nbstripout`](https://github.com/kynan/nbstripout) once and add
  it as a git filter:
  ```bash
  uv pip install nbstripout
  nbstripout --install
  ```
- **PARAMETERS cells.** Each notebook has a clearly-marked parameter cell.
  Edit values there; the rest of the notebook is the analysis logic. Add
  cells freely — these notebooks are meant to be modified.

## Relationship to the scripts under `scripts/`

The scripts under `scripts/` remain the production path: they are what the
cluster runs and what wrote every figure committed to `results/`.

Notebooks 01–03 sit alongside them for exploration, sharing the library
(`tez_deepgaze.instrument`, `tez_deepgaze.human_scanpaths`,
`tez_deepgaze.consensus_panel`, …) but carrying their own plotting so they read
top-to-bottom.

Notebooks 04–07 sit *on top of* the scripts and add no logic of their own. Their
job is to make the claim "every committed figure is regenerable from committed
code" checkable by someone who has just cloned the repo — so they are the place
to look when a figure and its caption seem to disagree. They do not cover every
figure; the section above names the ones that only a script produces.
