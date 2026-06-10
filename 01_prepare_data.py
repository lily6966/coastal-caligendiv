#!/usr/bin/env python3
"""
Step 1: Prepare unified training data from GenDivRange (global) and
single.gen.div.DF (California, including expert records).

Outputs:
  data/processed/global_train.csv   – GenDivRange population-level He records
  data/processed/california.csv     – single.gen.div.DF with metric_type label
  data/processed/species_meta.csv   – species-level metadata (taxonomy, life form)
"""

import os, csv, warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "data_raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────
# 1. GenDivRange spec_tab → species metadata
# ─────────────────────────────────────────────────────────
spec = pd.read_excel(RAW / "GenDivRange" / "spec_tab_v2025-03-31.xlsx")
spec_meta = spec[[
    "Spec_id", "Study_id", "Spec_Latin_GenDivRange", "Life_form",
    "Marker_type", "N_pops", "N_loci",
    "Family_GBIF", "Order_GBIF", "Class_GBIF", "Phylum_GBIF", "BIOME",
]].copy()
spec_meta.rename(columns={"Spec_Latin_GenDivRange": "species"}, inplace=True)

# ─────────────────────────────────────────────────────────
# 2. GenDivRange pop_tab → global training records
# ─────────────────────────────────────────────────────────
pop = pd.read_excel(RAW / "GenDivRange" / "pop_tab_v2025-03-31.xlsx")
# Drop empty trailing columns
pop = pop.loc[:, pop.columns.notna()]

pop_clean = pop[[
    "Spec_id", "Study_id", "Pop_id",
    "Latitude", "Longitude", "N",
    "Ar", "Ho", "He", "GD_Nei", "F_is", "Ploidy",
]].copy()

# Merge species metadata
pop_clean = pop_clean.merge(
    spec_meta[["Study_id", "species", "Life_form", "Marker_type",
               "N_loci", "Family_GBIF", "Order_GBIF", "Class_GBIF",
               "Phylum_GBIF", "BIOME"]],
    on="Study_id", how="left",
)

# Use He as primary diversity metric; fall back to GD_Nei
pop_clean["gen_div"] = pop_clean["He"]
mask_no_he = pop_clean["gen_div"].isna()
pop_clean.loc[mask_no_he, "gen_div"] = pop_clean.loc[mask_no_he, "GD_Nei"]

# metric_type for global data: all are heterozygosity-based
pop_clean["metric_type"] = "He"

# Filter: need valid lat, lon, gen_div
valid = (
    pop_clean["Latitude"].notna()
    & pop_clean["Longitude"].notna()
    & pop_clean["gen_div"].notna()
    & (pop_clean["gen_div"] >= 0)
    & (pop_clean["gen_div"] <= 1)
)
global_df = pop_clean[valid].copy()
global_df["source"] = "GenDivRange"

print(f"Global records: {len(global_df)} pops, {global_df['species'].nunique()} species")
print(f"  He range: {global_df['gen_div'].min():.4f} – {global_df['gen_div'].max():.4f}")

# ─────────────────────────────────────────────────────────
# 3. single.gen.div.DF → California records with metric_type
# ─────────────────────────────────────────────────────────
cali_raw = pd.read_excel(RAW / "single.gen.div.DF.xlsx")
cali_raw.columns = ["species", "Latitude", "Longitude", "gen_div"]
cali_raw["Latitude"] = pd.to_numeric(cali_raw["Latitude"], errors="coerce")
cali_raw["Longitude"] = pd.to_numeric(cali_raw["Longitude"], errors="coerce")
cali_raw["gen_div"] = pd.to_numeric(cali_raw["gen_div"], errors="coerce")

# Classify metric type per species based on value range
species_means = cali_raw.groupby("species")["gen_div"].mean()
# He-type species: mean gen_div > 0.05 (microsatellite heterozygosity)
# pi-type species: mean gen_div <= 0.05 (nucleotide diversity from mtDNA/sequence)
he_species = set(species_means[species_means > 0.05].index)
pi_species = set(species_means[species_means <= 0.05].index)

cali_raw["metric_type"] = cali_raw["species"].apply(
    lambda s: "He" if s in he_species else "pi"
)
cali_raw["source"] = "CaliPopGen_expert"

# Mark expert-contributed species
calipopgen_species = set(pd.read_csv(
    RAW / "CaliPopGen" / "CaliPopGen_dataset_1_population_genetic_diversity_TSV.tsv",
    sep="\t", usecols=["ScientificName"]
)["ScientificName"])

cali_raw["is_expert"] = ~cali_raw["species"].isin(calipopgen_species)

valid_cali = (
    cali_raw["Latitude"].notna()
    & cali_raw["Longitude"].notna()
    & cali_raw["gen_div"].notna()
)
cali_df = cali_raw[valid_cali].copy()

print(f"\nCalifornia records: {len(cali_df)} pops, {cali_df['species'].nunique()} species")
print(f"  He-type species ({len(he_species)}): {sorted(he_species)}")
print(f"  pi-type species ({len(pi_species)}): {sorted(pi_species)}")
print(f"  Expert-only records: {cali_df['is_expert'].sum()}")

# ─────────────────────────────────────────────────────────
# 4. Save processed data
# ─────────────────────────────────────────────────────────
global_df.to_csv(OUT / "global_train.csv", index=False)
cali_df.to_csv(OUT / "california.csv", index=False)
spec_meta.to_csv(OUT / "species_meta.csv", index=False)

print(f"\nSaved to {OUT}/")
print(f"  global_train.csv: {len(global_df)} rows")
print(f"  california.csv:   {len(cali_df)} rows")
print(f"  species_meta.csv: {len(spec_meta)} rows")
