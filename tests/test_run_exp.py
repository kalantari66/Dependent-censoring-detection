"""Regression coverage for synthetic experiment dimensions."""

import json
import os
import sys
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from cmi.null_sampling import prepare_null_nonparametric
from experiments.run_exp import main, prepare_experiment_dataset, resolve_dataset

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "real_exp.json"


class SyntheticExperimentTests(unittest.TestCase):
    """Check generated dimensions and complete experiment output."""

    def test_generated_dimensions_for_all_synthetic_modes(self) -> None:
        """Every synthetic mode generates the requested subjects and covariates."""
        config = json.loads(CONFIG_PATH.read_text())
        for dependency_kind in ("copula", "frailty"):
            for feature_kind in ("discrete", "continuous"):
                for n_features in (3, 4, 5):
                    with self.subTest(dependency=dependency_kind, features=feature_kind, count=n_features):
                        _, raw_df = resolve_dataset(
                            dataset="SYNTH",
                            dependency_kind=dependency_kind,
                            copula_type="clayton",
                            feature_kind=feature_kind,
                            seed=2026,
                            theta=3.0,
                            alpha=4.0,
                            n_samples=2000,
                            n_features=n_features,
                        )
                        df, features, _, _ = prepare_experiment_dataset(raw_df, config, "SYNTH")
                        self.assertEqual(len(df), 2000)
                        self.assertEqual(len(features), n_features)
                        self.assertEqual(df.shape, (2000, n_features + 2))

    def test_complete_synthetic_trials(self) -> None:
        """Generated covariates reach the detector and are recorded in the CSV."""
        original_cwd = Path.cwd()
        config = json.loads(CONFIG_PATH.read_text())
        # Keep actual generation and detection, with shorter training and fewer bootstraps.
        config["bootstrap_samples"] = [10]
        config["n_quantiles"] = [3]
        prepare_null = partial(
            prepare_null_nonparametric,
            rsf_params={"n_epochs": 1, "latent_dim": 4, "batch_size": 256, "device": "cpu"},
        )
        for n_features in (3, 4, 5):
            with self.subTest(n_features=n_features), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "config").mkdir()
                config["synthetic"]["n_features"] = [n_features]
                (root / "config" / "real_exp.json").write_text(json.dumps(config))
                try:
                    os.chdir(root)
                    with (
                        patch.object(sys, "argv", ["run_exp", "--dataset", "SYNTH", "--n-trials", "1"]),
                        patch("cmi.cmi.prepare_null_nonparametric", side_effect=prepare_null),
                    ):
                        main()
                finally:
                    os.chdir(original_cwd)
                result_files = list((root / "results").glob("*.csv"))
                self.assertEqual(len(result_files), 1)
                results = pd.read_csv(result_files[0])
                self.assertEqual(len(results), 1)
                row = results.iloc[0]
                self.assertEqual(row["n_samples"], 2000)
                self.assertEqual(row["n_features"], n_features)
                self.assertEqual(row["ncov_used"], n_features)
                self.assertEqual(len(json.loads(row["cov_used"])), n_features)
                self.assertEqual(row["status"], "success", row["error"])
                self.assertTrue(np.isfinite(row["p_value"]))
                self.assertTrue(0 <= row["p_value"] <= 1)


if __name__ == "__main__":
    unittest.main()
