# ThePerfectPotStudy

This project investigates human aesthetic preference through vase shapes. Data was collected from an evolution experiment where participants iteratively traverse a vase morphospace, making a series of binary choices to select shapes they find most beautiful. The results are analysed through an evolutionary biology and psychology lens using advanced computational frameworks.

---

## Repository Structure

### `file_processing/`
Processes raw game session data into analysis-ready formats. The main notebook (`game_data_processing.ipynb`) ingests session logs, cleans and validates them, and outputs structured parquet files used by downstream analyses.

### `pca_space/`
Defines the vase shape space. PCA is fit on vase outline coordinates to produce a low-dimensional shape embedding. Real museum vases are projected into this space and used to define the landscape participants explore during the study. The resulting PC scores are the primary representation used throughout the analysis.

- `pca_func_new.py` — reusable PCA functions
- `potPCA_new.ipynb` — fits the PCA model
- `pca_stratification.ipynb` — stratification analysis across the shape space
- `real_pot_pc_projection.ipynb` — backprojects real museum vases into the game PCA space (deprecated)

### `feature_extraction/`
Extracts geometric and morphological features from vase outlines (curvature, symmetry, proportions, entropy, etc.). These features are used in downstream modelling of aesthetic preference.

- `extraction_run.ipynb` — main extraction pipeline
- `function_modules/` — modular feature extraction code (kinematics, proportions, entropy, SRV elastic distance, etc.)
- `hubner_lines_0326/` — standardised body line profiles taken from Hübner et al. (2023), used as reference outlines
- `outdir/` — extracted feature matrices (large files excluded from version control)

### `flow_mapping/`
Maps participant selection trajectories through the UMAP-projected shape space and estimates an *aesthetic gravity field* using a Koopman operator framework. The gravity field describes the latent attractor structure of human aesthetic preference.

- `umap_projection.ipynb` / `umap_projection_game.ipynb` — projects vases into 2D UMAP space
- `trajectory_mapping.ipynb` — builds parent→child selection trajectories, estimates the Koopman operator, computes the aesthetic gravity field, and evaluates selection entropy
- `umap_projections/` — saved UMAP embeddings for game and real vases
- `gravity_projections/` — saved per-vase aesthetic gravity scores
- `figures/` — output figures

---

## Setup

```bash
pip install -r requirements.txt
```

Large data files (parquet/csv outputs over 50 MB) are excluded from version control via `.gitignore`. 

This file was generated using Claude Sonner 4.6 and edited by the author.