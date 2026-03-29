from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from attcon.data import TaskConfig, expand_cues_for_probe, generate_batch
from attcon.eval import run_ablations
from attcon.models import ModelConfig, RecurrentAttentionController, StaticAttentionBaseline
from attcon.train import train_experiment


class AttentionControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task_cfg = TaskConfig(num_steps=3)
        self.model_cfg = ModelConfig(hidden_size=16, cue_embedding_dim=6, scene_embedding_dim=8)

    def test_data_targets_align_with_digits(self) -> None:
        batch = generate_batch(16, self.task_cfg.num_steps, self.task_cfg)
        for idx in range(batch.scene.shape[0]):
            cue = batch.cue[idx].item()
            target_positions = torch.nonzero(batch.target_types[idx] == cue, as_tuple=False).flatten()
            self.assertEqual(target_positions.numel(), 1)
            pos = target_positions.item()
            self.assertEqual(pos, batch.target_pos[idx].item())
            self.assertEqual(batch.digits[idx, pos].item(), batch.target[idx].item())

    def test_model_shapes(self) -> None:
        batch = generate_batch(4, self.task_cfg.num_steps, self.task_cfg)
        static_model = StaticAttentionBaseline(self.task_cfg, self.model_cfg)
        recurrent_model = RecurrentAttentionController(self.task_cfg, self.model_cfg)

        for model in (static_model, recurrent_model):
            outputs = model(batch.scene, batch.cue, target=batch.target, num_steps=self.task_cfg.num_steps)
            self.assertEqual(outputs["logits"].shape, (4, self.task_cfg.digit_vocab_size))
            self.assertEqual(outputs["attention_seq"].shape, (4, self.task_cfg.num_steps, self.task_cfg.num_cells))
            self.assertEqual(
                outputs["observation_seq"].shape,
                (4, self.task_cfg.num_steps, 1 + self.task_cfg.digit_vocab_size),
            )
            self.assertEqual(outputs["confidence_seq"].shape, (4, self.task_cfg.num_steps, 1))
            self.assertEqual(outputs["loss_seq"].shape, (4, self.task_cfg.num_steps, 1))
        recurrent_outputs = recurrent_model(
            batch.scene,
            batch.cue,
            target=batch.target,
            num_steps=self.task_cfg.num_steps,
        )
        self.assertEqual(
            recurrent_outputs["inspection_seq"].shape,
            (4, self.task_cfg.num_steps, self.task_cfg.num_cells),
        )
        self.assertEqual(
            recurrent_outputs["self_model_seq"].shape,
            (4, self.task_cfg.num_steps, self.task_cfg.num_cells),
        )

    def test_attention_is_normalized(self) -> None:
        batch = generate_batch(4, self.task_cfg.num_steps, self.task_cfg)
        recurrent_model = RecurrentAttentionController(self.task_cfg, self.model_cfg)
        outputs = recurrent_model(batch.scene, batch.cue, target=batch.target, num_steps=self.task_cfg.num_steps)
        sums = outputs["attention_seq"].sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_models_accept_cue_sequences(self) -> None:
        batch = generate_batch(4, self.task_cfg.num_steps, self.task_cfg)
        cue_seq = batch.cue.unsqueeze(1).repeat(1, self.task_cfg.num_steps)
        cue_seq[:, -1] = (cue_seq[:, -1] + 1) % self.task_cfg.num_types
        for model in (
            StaticAttentionBaseline(self.task_cfg, self.model_cfg),
            RecurrentAttentionController(self.task_cfg, self.model_cfg),
        ):
            outputs = model(
                batch.scene,
                batch.cue,
                cue_seq=cue_seq,
                target=batch.target,
                num_steps=self.task_cfg.num_steps,
            )
            self.assertEqual(outputs["attention_seq"].shape, (4, self.task_cfg.num_steps, self.task_cfg.num_cells))

    def test_recurrence_responds_to_feedback(self) -> None:
        batch = generate_batch(2, self.task_cfg.num_steps, self.task_cfg)
        model = RecurrentAttentionController(self.task_cfg, self.model_cfg)
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
            prev_loss_index = 2 * self.task_cfg.num_cells + (1 + self.task_cfg.digit_vocab_size)
            model.summary_adapter[0].weight[0, prev_loss_index] = 5.0
            model.policy_head.weight[0, 0] = 1.0
            model.policy_head.bias.zero_()

        outputs_with_target = model(
            batch.scene,
            batch.cue,
            target=batch.target,
            num_steps=self.task_cfg.num_steps,
        )
        outputs_without_target = model(
            batch.scene,
            batch.cue,
            target=None,
            num_steps=self.task_cfg.num_steps,
        )
        self.assertFalse(
            torch.allclose(
                outputs_with_target["attention_seq"][:, 1],
                outputs_without_target["attention_seq"][:, 1],
                atol=1e-6,
            )
        )

    def test_probe_expansion_covers_all_cues(self) -> None:
        batch = generate_batch(3, self.task_cfg.num_steps, self.task_cfg)
        expanded = expand_cues_for_probe(batch, self.task_cfg.num_types)
        self.assertEqual(expanded.scene.shape[0], 3 * self.task_cfg.num_types)
        self.assertEqual(expanded.cue.unique().numel(), self.task_cfg.num_types)

    def test_state_intervention_changes_attention(self) -> None:
        batch = generate_batch(2, self.task_cfg.num_steps, self.task_cfg)
        model = RecurrentAttentionController(self.task_cfg, self.model_cfg)
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
            model.policy_head.weight[0, 0] = 1.0
            model.policy_head.bias.zero_()

        baseline = model(batch.scene, batch.cue, target=batch.target, num_steps=self.task_cfg.num_steps)
        delta = torch.zeros(2, self.model_cfg.hidden_size)
        delta[:, 0] = 5.0
        intervention = model(
            batch.scene,
            batch.cue,
            target=batch.target,
            num_steps=self.task_cfg.num_steps,
            intervention={"step": 0, "delta": delta},
        )
        self.assertFalse(
            torch.allclose(
                baseline["attention_seq"][:, 0],
                intervention["attention_seq"][:, 0],
                atol=1e-6,
            )
        )

    def test_inspection_state_tracks_visited_cells(self) -> None:
        batch = generate_batch(2, self.task_cfg.num_steps, self.task_cfg)
        model = RecurrentAttentionController(self.task_cfg, self.model_cfg)
        outputs = model(batch.scene, batch.cue, target=batch.target, num_steps=self.task_cfg.num_steps)
        inspection = outputs["inspection_seq"]
        self.assertTrue(torch.all(inspection[:, 1:] >= inspection[:, :-1]))
        self.assertTrue(torch.all((inspection == 0.0) | (inspection == 1.0)))

    def test_train_and_eval_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
                "output_dir": tmp_dir,
                "device": "cpu",
                "task": {"num_steps": 2},
                "model": {"hidden_size": 12, "cue_embedding_dim": 4, "scene_embedding_dim": 8},
                "training": {
                    "batch_size": 16,
                    "train_steps": 4,
                    "val_batches": 2,
                    "val_interval": 2,
                    "log_interval": 2,
                    "learning_rate": 1e-3,
                    "aux_loss_weight": 0.1,
                    "attention_entropy_weight": 0.01,
                    "cue_switch_probability": 0.5,
                    "cue_switch_step": 1,
                },
                "evaluation": {
                    "test_batches": 2,
                    "probe_scenes": 2,
                    "predictive_probe": {
                        "train_batches": 2,
                        "test_batches": 1,
                        "epochs": 10,
                        "learning_rate": 0.05,
                    },
                    "intervention_test": {
                        "enabled": True,
                        "probe_scenes": 2,
                        "step": 1,
                    },
                    "cue_switch": {
                        "enabled": True,
                        "probe_scenes": 2,
                        "switch_step": 1,
                    },
                    "report_probes": {
                        "enabled": True,
                        "train_batches": 2,
                        "test_batches": 1,
                        "epochs": 10,
                        "learning_rate": 0.05,
                    },
                    "self_modeling": {
                        "enabled": True,
                        "train_batches": 2,
                        "test_batches": 1,
                        "epochs": 10,
                        "learning_rate": 0.05,
                    },
                    "reduced_shaping": {
                        "enabled": True,
                        "weights": [0.0],
                    },
                    "ablations": ["freeze_recurrence"],
                },
            }
            result = train_experiment(config)
            self.assertTrue(Path(result["checkpoint_path"]).exists())
            report = run_ablations(config, result["checkpoint_path"])
            self.assertIn("baseline", report)
            self.assertIn("recurrent", report)
            self.assertIn("predictive_probe", report)
            self.assertIn("report_probes", report)
            self.assertIn("self_modeling", report)
            self.assertIn("cue_switch", report)
            self.assertIn("intervention_test", report)
            self.assertIn("reduced_shaping", report)
            self.assertIn("controller_state_probe", report["predictive_probe"])
            self.assertIn("observation_only_probe", report["predictive_probe"])
            self.assertIn("current_search_type", report["report_probes"])
            self.assertIn("target_inspected_report", report["self_modeling"])
            self.assertIn("observation_only_probe", report["self_modeling"]["native_cell_report"])
            self.assertIn("recurrent", report["cue_switch"])
            self.assertIn("explicit_attention_modeling", report["evidence"])
            self.assertIn("self_modeling_of_attention", report["evidence"])
            self.assertIn("reportable_internal_content", report["evidence"])
            self.assertIn("cue_switch_adaptation", report["evidence"])
            self.assertIn("causal_attention_intervention", report["evidence"])
            self.assertIn("reduced_shaping_resilience", report["evidence"])
            self.assertTrue(Path(report["artifacts"]["report"]).exists())
            self.assertTrue(report["artifacts"]["plots"])


if __name__ == "__main__":
    unittest.main()
