# Genetic Diversity and Climate Change Resilience for California Coastal Species

A meta-learning framework using Conditional Neural Processes (CNP) to predict genetic diversity and assess climate change resilience across 31 California coastal species.

## Project structure

```
Genomics/
├── 01_prepare_data.py              # Compile genetic diversity records from GenDiv and CaliPopGen
├── 02_add_env_covariates.py        # Extract WorldClim v2.1 bioclimatic variables
├── 02b_add_marine_covariates.py    # Extract Bio-ORACLE v2.2 marine surface variables
├── 03_train_model.py               # Train CNP: Phase 1 global pre-training, Phase 2 CA fine-tuning
├── 04_predict_california.py        # Generate coastal predictions for each California species
├── 05_resilience_assessment.py     # Compute vulnerability scores, classify resilience, produce maps
├── 06_future_climate.py            # Future (SSP5-8.5) exposure and diversity, 5-GCM ensemble
├── 06b_gcm_comparison_figure.py    # Per-GCM comparison figure
├── 07_future_diversity_map.py      # Maps of projected end-of-century diversity
├── 06c_future_diversity.py         # CNP-projected end-of-century diversity columns
├── 06d_future_perGCM.py            # Per-GCM future vulnerability/resilience + model-agreement figure
├── 11_gcm_ensemble_diversity.py    # Per-GCM 1 km future diversity ensemble (historical vs future)
├── 08_exposure_diagnostics.py      # Per-component view of the change signal + climate velocity
├── 10_global_delta_context_figure.py  # California He species vs the global cloud
├── climate_delta.py                # THE definition of exposure: projected climate CHANGE
├── archive/                        # Superseded parallel delta scripts and figures
├── environment.yml                 # Conda environment specification
│
├── data/
│   ├── data_raw/                   # Original source data (not tracked in version control)
│   │   ├── GenDivRange/            #   Global genetic diversity database (spec_tab, pop_tab)
│   │   ├── CaliPopGen/             #   California Population Genomics dataset
│   │   └── single.gen.div.DF.xlsx  #   Curated California records with expert contributions
│   ├── env_rasters/                # WorldClim v2.1 GeoTIFFs (wc2.1_10m_bio_*.tif)
│   └── processed/                  # Pipeline outputs
│       ├── global_train.csv        #   Cleaned global records (19,163 populations, 1,108 species)
│       ├── california.csv          #   Cleaned California records (557 populations, 31 species)
│       ├── *_env.csv               #   Records with environmental covariates appended
│       ├── predictions.csv         #   Coastal grid predictions per species
│       └── vulnerability_scores.csv#   Per-species vulnerability and resilience classification
│
├── models/
│   ├── cnp_global_pretrained.pt    # Phase 1 model weights (global pre-training)
│   ├── cnp_california_finetuned.pt # Phase 2 model weights (California fine-tuning)
│   └── feature_processor.pkl       # Fitted StandardScaler and label encoders
│
└── figures/                        # Generated figures
    ├── resilience_map.png          #   Species resilience classification maps
    ├── diversity_map.png           #   Predicted genetic diversity maps
    ├── ecosystem_resilience.png    #   Ecosystem-level vulnerability assessment
    ├── species_vulnerability_ranking.png
    ├── obs_vs_pred.png             #   Model validation: observed vs predicted
    ├── calibration.png             #   Uncertainty calibration
    ├── global_diversity_*.png      #   Global diversity predictions and validation
    ├── resilience_agreement_perGCM.png         # Present-day per-GCM resilience + model agreement
    ├── future_diversity_perGCM_uncertainty.png # Projected 2100 diversity + between-GCM uncertainty
    └── ...
```

## Reproducibility pipeline

The pipeline runs sequentially. Each script reads the output of the previous step.

### Step 0: Set up the environment

```bash
conda env create -f environment.yml
conda activate genom
```

### Step 1: Prepare genetic diversity data

```bash
python 01_prepare_data.py
```

Compiles population-level genetic diversity records from the GenDiv database (global He) and the CaliPopGen dataset (California He and pi). Classifies California species as heterozygosity-type (He) or nucleotide-diversity-type (pi) based on observed value distributions. Outputs cleaned CSV files to `data/processed/`.

### Step 2: Add environmental covariates

```bash
python 02_add_env_covariates.py
python 02b_add_marine_covariates.py
```

Extracts nine WorldClim v2.1 bioclimatic variables from GeoTIFFs at each population coordinate. Coastal locations outside the terrestrial land mask are gap-filled via nearest-neighbor interpolation. Derives three climate indices: normalized temperature seasonality, precipitation extremity, and composite climate stress. Then adds seven Bio-ORACLE v2.2 marine surface variables (SST, salinity, chlorophyll-a, dissolved oxygen, pH) and classifies populations as marine or terrestrial.

**Required data:**
- WorldClim v2.1 bioclimatic rasters (`data/env_rasters/wc2.1_10m_bio_*.tif`) — download from https://www.worldclim.org/data/worldclim21.html
- Bio-ORACLE v2.2 — fetched automatically via ERDDAP (requires internet)

### Step 3: Train the Conditional Neural Process

```bash
python 03_train_model.py
```

Two-phase meta-transfer learning:
- **Phase 1** — Global pre-training on 1,108 species (100 epochs, 300 episodes/epoch). Episode types: 70% within-species, 30% cross-species (same taxonomic order).
- **Phase 2** — California fine-tuning on 31 coastal species (80 epochs, 100 episodes/epoch). Episode types: 50% within-species, 50% mixed-context (augmented with global records from the same taxonomic family).

Saves model weights to `models/` and the fitted feature processor to `models/feature_processor.pkl`.

### Step 4: Predict genetic diversity along the California coast

```bash
python 04_predict_california.py
```

Generates a 150-point coastal grid (30.5 N to 42 N) and predicts genetic diversity (with uncertainty) for each species within its observed latitudinal range (+/- 1.5 degrees). Outputs predictions to `data/processed/predictions.csv`.

### Step 5: Resilience assessment

```bash
python 05_resilience_assessment.py
python 06c_future_diversity.py     # writes the joint present-future diversity scale
python 05_resilience_assessment.py # rerun so both slices share that scale
```

Computes per-species climate exposure, vulnerability scores, and resilience classification (resilient, at-risk, latent vulnerability, critical). Generates resilience maps, diversity maps, species vulnerability rankings, and ecosystem-level assessment figures.

**Exposure is the projected CHANGE in climate parameters**, not their absolute value in any single time slice — see `climate_delta.py`, which is the one place the definition lives:

| Domain | Weight | Component |
|---|---|---|
| Marine (63%) | 0.10 / 0.10 | Δ mean SST, Δ max SST |
| | 0.08 | \|Δ SST annual range\| |
| | 0.15 | acidification (pH decline) |
| | 0.12 | deoxygenation (O₂ decline) |
| | 0.08 | thermal novelty (Δ SST / baseline SST range) |
| Terrestrial (30%) | 0.10 / 0.08 | Δ bio5, Δ bio1 |
| | 0.06 | \|Δ bio4\| |
| | 0.06 | relative drying of bio14 |

Baselines: WorldClim 1970–2000 → CMIP6 SSP5-8.5 2081–2100 per GCM (terrestrial); Bio-ORACLE SSP5-8.5 2020 → avg(2080, 2090) (marine — the 2020 step of the same product as the scenario, so the change carries no model-vs-observation bias). Each component is normalized against a fixed reference *change* range so scores are absolute rather than min–max within the sample. Exposure is evaluated per GCM, giving `climate_exposure` (ensemble mean), `climate_exposure_sd/min/max`, and per-GCM columns.

The superseded state-based indices are still written as `climate_exposure_v1` and `climate_exposure_state` so the comparison figures can show what changed.

**Diversity is normalized within each species** — min–max against that species' own predicted values, so a population scores as low-diversity relative to the rest of its species rather than against a pooled cross-species reference. The cross-species version is retained as `diversity_norm_global` for the comparison figures.

The per-species scale spans **both time slices** (`data/processed/diversity_scaling.csv`, written by step 6c). This matters: a species' spatial spread along its coastal range is much narrower than the change it undergoes by 2100, so scaling on the present alone pushes every 2100 value off the scale — He species clip to 0, π species to 1. Taking min/max over present and future together keeps both spatial and temporal variation visible. Because of that dependency the intended order is **05 → 06c → 05**; on a first run, with no scaling file present, step 5 bootstraps from the present-day range and says so.

### Step 6: Future climate (SSP5-8.5)

```bash
python 06_future_climate.py
```

Re-predicts genetic diversity under end-of-century conditions (WorldClim CMIP6 2081-2100 across 5 GCMs; Bio-ORACLE SSP5-8.5 average of the 2080 and 2090 steps) and re-scores exposure per GCM. Since exposure is the projected change, it is the same quantity as in step 5 — what differs between now and 2100 is the diversity term, so `vulnerability_future` = (1 − projected 2100 diversity) × exposure + 0.3σ.

### Step 6c: Projected diversity columns

```bash
python 06c_future_diversity.py
```

Recomputes the CNP-projected end-of-century diversity into `data/processed/future_diversity.csv`, row-aligned with `vulnerability_scores.csv`, without overwriting any step 6 output. Step 6 also writes these columns itself; this script exists so downstream figures can get them without rerunning the whole of step 6.

### Step 6d: Per-GCM resilience (present day) and projected diversity (future)

```bash
python 11_gcm_ensemble_diversity.py   # per-GCM 1 km future diversity ensemble (prerequisite)
python 06d_future_perGCM.py
```

Carries the full 5-GCM ensemble through the analysis rather than collapsing to the ensemble mean first, and separates the two epochs by what is well-posed for each.

**Vulnerability / resilience — present day only, quantified per GCM.** Exposure is defined as the projected climate *change* (present → SSP5-8.5 2081–2100), which is exactly the pressure a currently-existing population faces, and it varies by GCM — so present-day resilience carries genuine climate-model spread through that term. Each of the five GCMs is classified independently into resilience quadrants with the present-day rule (`classify_resilience`): within-species diversity normalization on the joint present∪future scale, and an LULC-augmented total stressor ((1−w)·Δ-climate exposure + w·land-use pressure, w = 0.25, LULC held at present). Per population we report the modal class, the fraction of GCMs backing it, and the per-model class probabilities. Outputs `figures/resilience_agreement_perGCM.{pdf,png}` (model-agreement map, per-latitude class probability, consensus-strength breakdown) and `data/processed/resilience_agreement_perGCM.csv` (`modal_class_perGCM`, `model_agreement`, `p_critical_perGCM`). The modal class is Resilient for 37.7% of populations, Latent Vulnerability for 36.9%, Critical for 14.0%, and At Risk for 11.4%; 90.3% of populations are classified unanimously across all five GCMs and 100% carry a ≥3/5 majority — present-day classification is robust to GCM choice because only the exposure term varies.

**Future — projected diversity, not a vulnerability class.** A well-posed *future* vulnerability would need the climate change a population faces *from 2100 onward*, but CMIP6 runs end at 2100, so that forward exposure is undefined. The future is therefore reported as projected end-of-century genetic diversity with its between-GCM uncertainty, not as a resilience classification. Outputs `figures/future_diversity_perGCM_uncertainty.{pdf,png}` (ensemble-mean projected within-species diversity map with certainty encoded as opacity; present-vs-2100 diversity by latitude with an inter-GCM band; and a per-latitude between-GCM SD profile) and `data/processed/future_diversity_perGCM_uncertainty.csv`. Projected within-species diversity falls from a present mean of 0.453 to 0.404 by 2100 (change −0.049), with a mean between-GCM SD of 0.062. The intermediate table `data/processed/future_perGCM_diversity_exposure.csv` retains the per-GCM exposure and projected-diversity inputs (no vulnerability columns).

### Step 8: Exposure diagnostics

```bash
python 08_exposure_diagnostics.py
```

Opens up the exposure definition component by component along the coastal grid, and adds along-shore climate velocity — the speed at which a present-day isotherm must travel up the coast to stay in the same conditions. Outputs `data/processed/climate_delta_grid.csv` and `figures/delta_climate_profiles.{png,pdf}`.

### Step 10: Global-context figure

```bash
python 10_global_delta_context_figure.py
```

Places the California He species inside the global cloud of 19,163 populations on the same exposure axis, scoring the global populations with `climate_delta.compute_climate_exposure` (terrestrial change from the local CMIP6 rasters; marine change from Bio-ORACLE on a 1-degree global stride, native resolution inside the California box). Because exposure is a fixed per-site quantity, the current → future arrow in panel (d) is vertical: what moves is projected diversity, not exposure. Outputs `figures/div_vs_exposure_He_combined.{pdf,png}` and `data/processed/global_exposure.csv`.

### Archived

`archive/` holds the earlier scripts and figures that computed the change-based exposure as a *parallel* index alongside the state-based one (`08_delta_exposure_vulnerability.py`, `09_delta_figures.py`, and their `*_delta` outputs). That duplication is gone: exposure is now defined once, in `climate_delta.py`, and the main pipeline figures carry it.

## Data sources

| Source | Description | Access |
|--------|-------------|--------|
| GenDiv v2025-03-31 | Global population genetic diversity (He, Nei's GD) for 1,108 species | https://gendiv.ethz.ch |
| CaliPopGen | California population genomics dataset | https://calipopgen.org |
| WorldClim v2.1 | Bioclimatic variables, 10-arc-minute resolution | https://www.worldclim.org |
| Bio-ORACLE v2.2 | Marine environmental layers (surface) | https://www.bio-oracle.org |

## Hardware

Training was performed on Apple Silicon (MPS backend). The pipeline automatically selects MPS if available, otherwise falls back to CPU. Total training time is approximately 2 hours on M-series hardware.
