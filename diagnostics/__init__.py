"""diagnostics -- one-off probes, audits, and verification scripts.

None of these are imported by the pipeline; each is run standalone from the
repo root so the flat imports (bundles, to_schema, score, results_paths,
results_matrix) resolve:
    python -m diagnostics.capacity_probe
"""
