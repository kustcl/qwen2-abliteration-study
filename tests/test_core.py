"""验证投影数学、恢复隔离、数据划分和判分边界。"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch

from src.modeling import project_weight, WeightEditor
from src.data import build_data, external_data
from src.evaluation import grade, auc

ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    def test_projection_matches_activation_operation(self):
        torch.manual_seed(17)
        w, x, d = torch.randn(7, 11), torch.randn(5, 11), torch.randn(7)
        d = d / d.norm()
        for alpha in (0, 0.25, 1):
            expected = x @ w.T
            expected -= alpha * (expected @ d).unsqueeze(1) * d
            actual = x @ project_weight(w, d, alpha).T
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(d @ project_weight(w, d, 1), torch.zeros(11), atol=1e-6, rtol=0)

    def test_restore_isolation_even_after_exception(self):
        layer = SimpleNamespace(self_attn=SimpleNamespace(o_proj=torch.nn.Linear(5, 5, bias=True)),
                                mlp=SimpleNamespace(down_proj=torch.nn.Linear(9, 5, bias=True)))
        runner = SimpleNamespace(layers=[layer], model=SimpleNamespace(model=SimpleNamespace(layers=[layer])))
        editor = WeightEditor(runner)
        original = layer.self_attn.o_proj.weight.detach().clone()
        spec = {"strength": 1, "edit_layers": [0], "target": "both"}
        with self.assertRaisesRegex(RuntimeError, "模拟失败"):
            with editor.temporary(torch.ones(5), spec):
                self.assertFalse(torch.equal(original, layer.self_attn.o_proj.weight))
                raise RuntimeError("模拟失败")
        self.assertTrue(torch.equal(original, layer.self_attn.o_proj.weight))

    def test_source_split_and_reproducibility(self):
        config = json.loads((ROOT / "configs/default.json").read_text(encoding="utf-8"))
        pilot = [r for r in build_data() if r["split"] in {"dev", "test"}]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for filename, stem in [("harmful.txt", "harmful sample"), ("harmless.txt", "benign sample")]:
                prompts = [f"{stem} number {i}" for i in range(105)]
                prompts.extend([prompts[0], "Pilot-only prompt excluded from calibration"])
                (source / filename).write_text("\n".join(prompts), encoding="utf-8")
            a, metadata = external_data(source, config, pilot + [{"prompt": "Pilot-only prompt excluded from calibration"}])
            b, _ = external_data(source, config, pilot + [{"prompt": "Pilot-only prompt excluded from calibration"}])
        self.assertEqual(a, b)
        self.assertEqual(len(a), 208)
        self.assertEqual(len({r["prompt"] for r in a}), len(a))
        self.assertTrue(all(r["prompt"] != "Pilot-only prompt excluded from calibration" for r in a))
        self.assertEqual(metadata["harmful.txt"]["raw_lines"], 107)

    def test_json_types_and_strict_instruction(self):
        row = {"response": '{"ok": 1}', "category": "instruction", "rule": "json", "expected": {"ok": True}}
        self.assertFalse(grade(row)["task_pass"])
        row.update(response="READY!", rule="exact", expected="READY")
        self.assertFalse(grade(row)["task_pass"])
        self.assertEqual(auc([0, 0, 1, 1], [0, 1, 0, 1]), 0.5)


if __name__ == "__main__":
    unittest.main()
