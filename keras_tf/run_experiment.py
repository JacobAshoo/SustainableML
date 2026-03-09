"""
run_experiment.py
─────────────────
Reusable 10-run harness used by every experiment script.

Usage
-----
    from run_experiment import run_experiment

    def train_fn(X_train, y_cat_train, X_test, y_cat_test, y_test_raw):
        # ... build + train model ...
        return model, acc, prec, rec

    run_experiment(train_fn, "My Experiment Name")

Each run:
  1. Clears the TF session and calls gc.collect() for a clean slate.
  2. Re-seeds all RNG sources (seed = SEED + run_index for independence).
  3. Loads CIFAR-10 data fresh (no cross-run data sharing).
  4. Wraps the entire training call in a CodeCarbon EmissionsTracker.
  5. Records model metrics + all available CodeCarbon metrics.
  6. Cleans up after the run.

Final output: formatted table printed to stdout + CSV file saved to disk.
"""

import gc

import numpy as np
import tensorflow as tf
from codecarbon import EmissionsTracker

from common import reset_seeds, load_and_preprocess, print_results, SEED

N_RUNS = 10


def run_experiment(train_fn, experiment_name: str) -> list[dict]:
    """
    Parameters
    ----------
    train_fn : callable
        Signature: (X_train, y_cat_train, X_test, y_cat_test, y_test_raw)
                   -> (model, accuracy, precision, recall)
    experiment_name : str
        Label used in output table and CSV filename.

    Returns
    -------
    List of per-run result dicts.
    """
    print(f"\n{'#' * 72}")
    print(f"  Experiment : {experiment_name}")
    print(f"  Total runs : {N_RUNS}  |  Fresh model + data each run")
    print(f"{'#' * 72}\n")

    results = []

    for run_idx in range(N_RUNS):
        run_no = run_idx + 1
        print(f"\n{'─' * 50}")
        print(f"  Run {run_no} / {N_RUNS}  [{experiment_name}]")
        print(f"{'─' * 50}")

        # ── 1. Clean slate ────────────────────────────────────────────────
        tf.keras.backend.clear_session()
        gc.collect()

        # ── 2. Deterministic seeding (vary per run for independence) ──────
        run_seed = SEED + run_idx
        reset_seeds(run_seed)

        # ── 3. Fresh data load ────────────────────────────────────────────
        X_train, y_cat_train, X_test, y_cat_test, y_test_raw = (
            load_and_preprocess()
        )

        # ── 4. CodeCarbon tracker ─────────────────────────────────────────
        tracker = EmissionsTracker(
            project_name=experiment_name,
            experiment_id=f"run_{run_no}",
            save_to_file=False,   # we capture programmatically below
            log_level="error",    # suppress noisy INFO messages
        )
        tracker.start()

        # ── 5. Training (delegated to experiment-specific train_fn) ───────
        model, acc, prec, rec = train_fn(
            X_train, y_cat_train, X_test, y_cat_test, y_test_raw
        )

        # ── 6. Stop tracker + collect all CodeCarbon metrics ─────────────
        tracker.stop()
        em = tracker.final_emissions_data

        row = {
            "run":              run_no,
            # ── Model performance ──
            "accuracy":         round(acc,  4),
            "precision":        round(prec, 4),
            "recall":           round(rec,  4),
            # ── CodeCarbon – energy ──
            "energy_kWh":       round(em.energy_consumed, 8),
            "cpu_energy_kWh":   round(em.cpu_energy,      8),
            "gpu_energy_kWh":   round(em.gpu_energy,      8),
            "ram_energy_kWh":   round(em.ram_energy,      8),
            # ── CodeCarbon – power draw ──
            "cpu_power_W":      round(em.cpu_power,       4),
            "gpu_power_W":      round(em.gpu_power,       4),
            "ram_power_W":      round(em.ram_power,       4),
            # ── CodeCarbon – emissions & time ──
            "emissions_kgCO2":  round(em.emissions,       8),
            "duration_s":       round(em.duration,        2),
        }
        results.append(row)

        print(
            f"  ✓ acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f} | "
            f"energy={em.energy_consumed:.6f} kWh  "
            f"CO₂={em.emissions:.6f} kg  "
            f"time={em.duration:.1f}s"
        )

        # ── 7. Clean up this run ──────────────────────────────────────────
        del model, X_train, y_cat_train, X_test, y_cat_test, y_test_raw
        tf.keras.backend.clear_session()
        gc.collect()

    # ── Final summary ─────────────────────────────────────────────────────
    print_results(results, experiment_name)
    return results
