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
├── 06_global_diversity_map.py      # Global diversity predictions and leave-one-out validation
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
```

Computes per-species climate exposure (70% marine-weighted, 30% terrestrial), vulnerability scores, and resilience classification (resilient, at-risk, latent vulnerability, critical). Generates resilience maps, diversity maps, species vulnerability rankings, and ecosystem-level assessment figures.

### Step 6: Global diversity analysis (optional)

```bash
python 06_global_diversity_map.py
```

Leave-one-out cross-validation of global predictions across all species. Produces global observed/predicted diversity maps, gridded comparisons, and latitudinal gradient figures.

## Data sources

| Source | Description | Access |
|--------|-------------|--------|
| GenDiv v2025-03-31 | Global population genetic diversity (He, Nei's GD) for 1,108 species | https://gendiv.ethz.ch |
| CaliPopGen | California population genomics dataset | https://calipopgen.org |
| WorldClim v2.1 | Bioclimatic variables, 10-arc-minute resolution | https://www.worldclim.org |
| Bio-ORACLE v2.2 | Marine environmental layers (surface) | https://www.bio-oracle.org |

## Hardware

Training was performed on Apple Silicon (MPS backend). The pipeline automatically selects MPS if available, otherwise falls back to CPU. Total training time is approximately 2 hours on M-series hardware.
