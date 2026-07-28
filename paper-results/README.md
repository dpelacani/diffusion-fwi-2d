# paper-results

Generates every figure and table in the paper from `Paper_plots.ipynb`.

## Running the notebook

Install the repo's regular dependencies (`pip install -e .` from the repo root,
or `pip install -r requirements.txt`) and run all cells with that environment's
Jupyter kernel. `stride` is not required.
No cluster/scratch access or external data is needed either: everything the
notebook reads lives under `data/` and `precomputed/` in this directory.

## Where the data comes from

- **`data/`**: small frozen subsets needed for Figures 1, 3, and 4:
  ground-truth velocity maps, one acquisition file, transducer-ring geometry, and a handful of diffusion-sweep outputs.
- **`precomputed/`**: the values for Tables 2, 3, B2, B3 and Figure 5.
  For Table 3/B2 and Figure 5 values are computed from FWI run data
  (`compute_fwi_results.py`); for Table 2/B3 these are the published values from the paper itself as placeholders. Change by passing
  `precomputed_json=None` / `use_precomputed=False` once data is hosted.
- **`results/`**: where a run of `compute_fwi_results.py` writes output; not 
  needed to reproduce the paper's plots.

## Layout

- `figures/`: one script per figure (`fig1_setup.py`, `fig3_samples.py`,
  `fig4_sweep.py`, `fig5_convergence.py`), imported directly by the notebook.
- `compute_fwi_results.py`, `latex_tables.py`: aggregate FWI run metrics and
  render the LaTeX tables.
- `style.py`: shared matplotlib styling.