"""loaders -- per-disease raw-data loaders (covid, dengue, ebola, influenza).

Each writes its bundle to data/processed/. Run as a module so the flat imports
(to_schema, build_datasets, dengue_aliases) still resolve from the repo root:
    python -m loaders.covid_load
"""
