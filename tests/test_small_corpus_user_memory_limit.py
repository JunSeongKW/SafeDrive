import ast
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from resume_small_corpus_user_memory_limit import replace_card_guard


class UserMemoryLimitTests(unittest.TestCase):
    def test_only_memory_boundary_changes(self):
        def stopped(bytes_used, user_pause=False, loss=3.0):
            return user_pause or bytes_used > 46_500_000_000, loss * .1
        updated = replace_card_guard(stopped)
        self.assertTrue(stopped(47_000_000_000)[0])
        self.assertFalse(updated(47_000_000_000)[0])
        self.assertFalse(updated(48_000_000_000)[0])
        self.assertTrue(updated(48_000_000_001)[0])
        self.assertTrue(updated(1, user_pause=True)[0])
        self.assertEqual(stopped(1)[1], updated(1)[1])
        self.assertEqual(stopped.__code__.co_code, updated.__code__.co_code)

    def test_missing_constant_rejected(self):
        def unrelated(value):
            return value > 48_000_000_000
        with self.assertRaises(AssertionError):
            replace_card_guard(unrelated)

    def test_registered_trainer_bytecode_preserved_except_guard_value(self):
        source = Path(__file__).resolve().parents[1] / "scripts/train_small_corpus_common_planner.py"
        parsed = ast.parse(source.read_text())
        definition = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        namespace = {}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
        original = namespace["main"]
        updated = replace_card_guard(original)
        self.assertEqual(original.__code__.co_code, updated.__code__.co_code)
        differences = [(before, after) for before, after in zip(original.__code__.co_consts, updated.__code__.co_consts)
                       if before != after]
        self.assertEqual(differences, [(46_500_000_000, 48_000_000_000)])


if __name__ == "__main__":
    unittest.main()
