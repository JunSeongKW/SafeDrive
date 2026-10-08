"""Execution changes must preserve stochastic forward and parameter gradients."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_small_corpus_joint_adaptive_checkpointing import ActivationStoragePolicy


class JointActivationStorageExecutionTests(unittest.TestCase):
    def test_recomputation_preserves_stochastic_outputs_gradients_and_rng(self):
        torch.manual_seed(14)
        network = torch.nn.Sequential(torch.nn.Linear(6, 12), torch.nn.Dropout(.3), torch.nn.SiLU(), torch.nn.Linear(12, 4))
        inputs = torch.randn(5, 6, requires_grad=True)
        initial_rng = torch.get_rng_state()
        result = network(inputs)
        result.square().sum().backward()
        native_gradients = [value.grad.clone() for value in network.parameters()]
        native_input_gradient = inputs.grad.clone()
        native_rng = torch.get_rng_state()
        original_keys = list(network.state_dict())
        policy = ActivationStoragePolicy(Path("unused"), Path("unused_marker"))
        policy.install(network)
        network.zero_grad(set_to_none=True)
        inputs.grad = None
        torch.set_rng_state(initial_rng)
        checkpointed = network(inputs)
        checkpointed.square().sum().backward()
        self.assertTrue(torch.equal(result, checkpointed))
        self.assertTrue(torch.equal(native_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(native_input_gradient, inputs.grad))
        self.assertEqual(original_keys, list(network.state_dict()))
        for before, after in zip(native_gradients, network.parameters()):
            self.assertTrue(torch.equal(before, after.grad))
        policy.recompute = False
        torch.set_rng_state(initial_rng)
        self.assertTrue(torch.equal(result, network(inputs)))

    def test_native_mode_requires_both_marker_and_low_external_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = ActivationStoragePolicy(root, root / "native.json")
            with patch.object(policy, "original_memory_limit") as set_limit:
                policy.update_before_forward()
                set_limit.assert_not_called()
                policy.native_marker.write_text("{}")
                with patch.dict("os.environ", LOCAL_RANK="0"), \
                     patch("train_small_corpus_common_planner.card_memory", return_value=8_000_000_000), \
                     patch("torch.cuda.memory_reserved", return_value=4_000_000_000):
                    policy.update_before_forward()
                self.assertTrue(policy.recompute)
                set_limit.assert_not_called()
                with patch.dict("os.environ", LOCAL_RANK="0"), \
                     patch("train_small_corpus_common_planner.card_memory", return_value=5_000_000_000), \
                     patch("torch.cuda.memory_reserved", return_value=4_000_000_000), \
                     patch("torch.cuda.get_device_properties") as properties:
                    properties.return_value.total_memory = 50_000_000_000
                    policy.update_before_forward()
                self.assertFalse(policy.recompute)
                set_limit.assert_called_once_with(.88)
                self.assertTrue((root / "native_activation_storage_rank0.json").exists())


if __name__ == "__main__":
    unittest.main()
