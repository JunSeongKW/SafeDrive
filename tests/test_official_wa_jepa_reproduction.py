"""Provenance/protocol guards; these tests do not execute GPU inference."""
import json
import runpy
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]


class OfficialWAJEPAReproductionTests(unittest.TestCase):
    def setUp(self):
        self.specification = json.loads((WORKSPACE / "configs/official_wa_jepa/reproduction_v1.json").read_text())
        self.results_root = WORKSPACE / "results/official_wa_jepa_reproduction"

    def test_sampling_protocol_is_registered_before_scores(self):
        self.assertEqual(self.specification["paper_inference"]["steps"], 12)
        self.assertEqual(self.specification["official_source_commit"], "404d8afd4f5e3334fe4b15a6ca7587b98cc35243")
        self.assertFalse(self.specification["training_authorized"])

    def test_owner_labeled_entrypoint_preserves_original_conda_environment(self):
        launcher = runpy.run_path(str(WORKSPACE / "scripts/launch_official_wa_jepa_workers.py"))
        original_environment = WORKSPACE / self.specification["conda_environment"]
        entrypoint = launcher["prepare_owner_labeled_python_entrypoint"](
            WORKSPACE, original_environment, self.specification["process_environment_alias"]
        )
        self.assertEqual(self.specification["process_owner_initials"], "kjs")
        self.assertEqual(entrypoint.parent.parent.name, "kjs-wa-jepa-eval")
        self.assertEqual(entrypoint.parent.parent.parent, WORKSPACE.parent / "envs")
        self.assertEqual(entrypoint.resolve(), (original_environment / "bin/python").resolve())

    def test_no_sparse_full_benchmark_or_training(self):
        benchmark = self.specification["sparse_benchmark"]
        self.assertFalse(benchmark["sparse_full_benchmark_authorized"])
        self.assertEqual(benchmark["all_id_tolerance"]["absolute"], 0)
        self.assertEqual(benchmark["all_id_tolerance"]["relative"], 0)

    def test_checkpoint_integrity_matches_registered_metadata(self):
        assets = json.loads((self.results_root / "verified_assets.json").read_text())
        for filename, expected in self.specification["checkpoint_files"].items():
            self.assertEqual(assets[filename]["bytes"], expected["bytes"])
            self.assertEqual(len(assets[filename]["sha256"]), 64)
            if expected.get("sha256"):
                self.assertEqual(assets[filename]["sha256"], expected["sha256"])

    def test_complete_official_inputs_and_compatible_cache(self):
        preflight = json.loads((self.results_root / "evaluation_preflight.json").read_text())
        self.assertTrue(preflight["complete"])
        self.assertEqual(preflight["expected_scene_count"], 12146)
        self.assertEqual(preflight["loaded_scene_count"], 12146)
        self.assertEqual(len(set(preflight["smoke_scene_tokens_fixed_before_gpu_inference"])), 6)
        for field in ("missing_scene_tokens", "unexpected_scene_tokens", "missing_metric_cache_tokens", "missing_metric_cache_files", "missing_current_camera_files"):
            self.assertEqual(preflight[field], [])
        self.assertTrue(preflight["metric_cache_compatibility"]["critical_files_identical"])

    def test_preset_checkpoint_shapes_and_steps_match(self):
        preflight = json.loads((self.results_root / "evaluation_preflight.json").read_text())
        agreement = preflight["checkpoint_metadata"]["checkpoint_preset_agreement"]
        self.assertTrue(all(item["matches"] for item in agreement.values()))
        self.assertEqual(agreement["num_inference_steps"]["checkpoint"], 12)

    def test_parallel_scene_partition_has_no_overlap_or_omission(self):
        expected = json.loads((self.results_root / "expected_scene_tokens.json").read_text())["tokens"]
        num_workers = self.specification["parallel_evaluation"]["execution_shards"]
        partitions = [expected[index::num_workers] for index in range(num_workers)]
        combined = [token for partition in partitions for token in partition]
        self.assertEqual(len(combined), len(set(combined)))
        self.assertEqual(set(combined), set(expected))
        self.assertEqual(self.specification["memory_safety"]["user_gpu_memory_cap_bytes"], 45_000_000_000)


if __name__ == "__main__":
    unittest.main()
