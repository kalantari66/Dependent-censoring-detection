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
from scipy.stats import kendalltau

from cmi.null_sampling import prepare_null_nonparametric
from data import dgp
from experiments.run_exp import main, prepare_experiment_dataset, resolve_dataset

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
CONFIG_PATH = CONFIG_DIR / "synth_exp.json"


class SyntheticExperimentTests(unittest.TestCase):
    """Check generated dimensions and complete experiment output."""

    def test_generated_dimensions_for_all_synthetic_modes(self) -> None:
        """Every synthetic mode generates the requested subjects and covariates."""
        config = json.loads(CONFIG_PATH.read_text())
        for dependency_kind in ("copula", "frailty"):
            for feature_kind in ("discrete", "continuous"):
                for n_features in (3, 4, 5):
                    with (
                        self.subTest(dependency=dependency_kind, features=feature_kind, count=n_features),
                        patch("data.data_generation.kendalltau", wraps=kendalltau) as tau,
                    ):
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
                        tau.assert_called_once()
                        event_times, censoring_times = tau.call_args.args
                        np.testing.assert_array_equal(raw_df["time"], np.minimum(event_times, censoring_times))
                        np.testing.assert_array_equal(raw_df["event"], (event_times <= censoring_times).astype(int))
                        self.assertAlmostEqual(
                            raw_df.attrs["kendall_tau"], kendalltau(event_times, censoring_times).statistic
                        )
                        self.assertNotIn("kendall_tau", df.columns)

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
                (root / "config" / "synth_exp.json").write_text(json.dumps(config))
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
                expected_tau = dgp(
                    kind="frailty_discrete",
                    n_subjects=2000,
                    n_features=n_features,
                    seed=2026,
                    alpha_E=4.0,
                    alpha_C=4.0,
                ).attrs["kendall_tau"]
                self.assertAlmostEqual(row["kendall_tau"], expected_tau)
                self.assertEqual(row["n_samples"], 2000)
                self.assertEqual(row["n_features"], n_features)
                self.assertEqual(row["ncov_used"], n_features)
                self.assertEqual(len(json.loads(row["cov_used"])), n_features)
                self.assertEqual(row["status"], "success", row["error"])
                self.assertTrue(np.isfinite(row["p_value"]))
                self.assertTrue(0 <= row["p_value"] <= 1)

    def test_dataset_uses_its_own_config(self) -> None:
        """Use separate real and synthetic settings through sampling and detection."""
        original_cwd = Path.cwd()
        frame = pd.DataFrame({"time": np.arange(1, 21), "event": [1, 0] * 10, "x0": [0] * 20})
        frame.attrs["kendall_tau"] = 0.5
        configs = {name: json.loads((CONFIG_DIR / name).read_text()) for name in ("real_exp.json", "synth_exp.json")}
        self.assertEqual(configs["real_exp.json"]["bootstrap_samples"], [100, 200, 300, 400])
        self.assertEqual(configs["synth_exp.json"]["bootstrap_samples"], [200, 300, 400, 500])
        self.assertNotIn("synthetic", configs["real_exp.json"])
        # Disjoint choices reveal the wrong config even with only two trials.
        configs["real_exp.json"]["bootstrap_samples"] = [100]
        configs["synth_exp.json"]["bootstrap_samples"] = [500]
        details = {
            "final_p_value": 0.5,
            "observed_fisher_stat": 1.0,
            "per_stratum_p_values": {"0": 0.5},
            "excluded_strata": [],
        }
        for dataset, config_name, expected_bootstraps in (
            ("GBSG2", "real_exp.json", 100),
            ("SYNTH", "synth_exp.json", 500),
            ("SEMI_GBSG2", "synth_exp.json", 500),
        ):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "config").mkdir()
                for name, config in configs.items():
                    (root / "config" / name).write_text(json.dumps(config))
                try:
                    os.chdir(root)
                    with (
                        patch.object(sys, "argv", ["run_exp", "--dataset", dataset, "--n-trials", "2"]),
                        patch("experiments.run_exp.resolve_dataset", return_value=(dataset, frame)) as resolve,
                        patch(
                            "experiments.run_exp.prepare_experiment_dataset",
                            return_value=(frame, ["x0"], "time", "event"),
                        ) as prepare,
                        patch("experiments.run_exp.detect_dependent_censoring", return_value=details) as detect,
                    ):
                        main()
                finally:
                    os.chdir(original_cwd)
                results = pd.read_csv(next((root / "results").glob("*.csv")))
                self.assertEqual(len(results), 2)
                self.assertTrue((results["status"] == "success").all())
                self.assertTrue((results["B"] == expected_bootstraps).all())
                self.assertEqual(prepare.call_args.kwargs["config"], configs[config_name])
                self.assertEqual(detect.call_count, 2)
                for call in detect.call_args_list:
                    self.assertEqual(call.kwargs["B"], expected_bootstraps)
                self.assertEqual(resolve.call_count, 2 if dataset == "SYNTH" else 1)
                if dataset == "SYNTH":
                    for call in resolve.call_args_list:
                        self.assertEqual(call.kwargs["n_samples"], 2000)
                        self.assertIn(call.kwargs["n_features"], [3, 4, 5])

    def test_tau_output_when_detection_fails(self) -> None:
        """Keep synthetic metadata on error rows and leave real-data tau undefined."""
        original_cwd = Path.cwd()
        frame = pd.DataFrame({"time": np.arange(1, 21), "event": [1, 0] * 10, "x0": [0] * 20})
        frame.attrs["kendall_tau"] = -0.5
        for dataset in ("SYNTH", "GBSG2", "SEMI_GBSG2"):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "config").mkdir()
                for config_name in ("real_exp.json", "synth_exp.json"):
                    (root / "config" / config_name).write_text((CONFIG_DIR / config_name).read_text())
                try:
                    os.chdir(root)
                    with (
                        patch.object(sys, "argv", ["run_exp", "--dataset", dataset, "--n-trials", "2"]),
                        patch("experiments.run_exp.resolve_dataset", return_value=(dataset, frame)),
                        patch(
                            "experiments.run_exp.prepare_experiment_dataset",
                            return_value=(frame, ["x0"], "time", "event"),
                        ),
                        patch("experiments.run_exp.detect_dependent_censoring", side_effect=ValueError("test failure")),
                    ):
                        main()
                finally:
                    os.chdir(original_cwd)
                results = pd.read_csv(next((root / "results").glob("*.csv")))
                self.assertEqual(len(results), 2)
                self.assertTrue((results["status"] == "error").all())
                self.assertTrue((results["error"] == "test failure").all())
                if dataset == "SYNTH":
                    self.assertTrue((results["kendall_tau"] == -0.5).all())
                else:
                    self.assertTrue(results["kendall_tau"].isna().all())


if __name__ == "__main__":
    unittest.main()
