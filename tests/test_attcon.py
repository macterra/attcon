from __future__ import annotations

import os
import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

import attcon.eval as eval_module
import attcon.nl_report as nl_report_module
from attcon.data import TaskConfig, expand_cues_for_probe, generate_batch
from attcon.eval import (
    build_evidence_summary,
    build_stage3_summary,
    nl_report_metrics,
    run_ablations,
    self_model_diagnostics,
    self_state_diagnostics,
    stage3_multi_seed_metrics,
)
from attcon.models import ModelConfig, RecurrentAttentionController, StaticAttentionBaseline
from attcon.nl_report import (
    _extract_response_json,
    collect_cue_switch_nl_examples,
    collect_intervention_nl_examples,
    collect_nl_examples,
    run_nl_report_mode,
)
from attcon.train import train_experiment
from attcon.train import load_config


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

    def test_stage3_robustness_is_enabled_in_repo_configs(self) -> None:
        for path in (
            ROOT / "configs" / "minimal.yaml",
            ROOT / "configs" / "tune_prob_025.yaml",
            ROOT / "configs" / "tune_prob_035.yaml",
        ):
            cfg = load_config(path)
            self.assertTrue(cfg["evaluation"]["stage3_multi_seed"]["enabled"], path.name)
            self.assertGreaterEqual(cfg["evaluation"]["stage3_multi_seed"]["num_seeds"], 3, path.name)
            self.assertIn("thresholds", cfg["evaluation"]["predictive_probe"], path.name)
            self.assertIn("thresholds", cfg["evaluation"]["intervention_test"], path.name)

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
        self.assertEqual(
            recurrent_outputs["found_state_seq"].shape,
            (4, self.task_cfg.num_steps, 1),
        )
        self.assertEqual(
            recurrent_outputs["target_found_seq"].shape,
            (4, self.task_cfg.num_steps, 1),
        )
        self.assertEqual(
            recurrent_outputs["relevant_region_seq"].shape,
            (4, self.task_cfg.num_steps, 1),
        )
        self.assertEqual(
            recurrent_outputs["unresolved_search_seq"].shape,
            (4, self.task_cfg.num_steps, 1),
        )
        self.assertEqual(
            recurrent_outputs["current_wrong_candidate_seq"].shape,
            (4, self.task_cfg.num_steps, 1),
        )
        self.assertEqual(
            recurrent_outputs["revisit_unresolved_seq"].shape,
            (4, self.task_cfg.num_steps, 1),
        )
        self.assertEqual(
            recurrent_outputs["allocation_error_seq"].shape,
            (4, self.task_cfg.num_steps, 1),
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

    def test_stage6b_signals_are_bounded(self) -> None:
        batch = generate_batch(8, self.task_cfg.num_steps, self.task_cfg)
        model = RecurrentAttentionController(self.task_cfg, self.model_cfg)
        outputs = model(
            batch.scene,
            batch.cue,
            target=batch.target,
            target_pos=batch.target_pos,
            num_steps=self.task_cfg.num_steps,
        )
        relevant = outputs["relevant_region_seq"]
        unresolved = outputs["unresolved_search_seq"]
        current_wrong_candidate = outputs["current_wrong_candidate_seq"]
        wrong_candidate_history = outputs["wrong_candidate_history_seq"]
        revisit_unresolved = outputs["revisit_unresolved_seq"]
        allocation_error = outputs["allocation_error_seq"]
        self.assertTrue(torch.all((relevant == 0.0) | (relevant == 1.0)))
        self.assertTrue(torch.all((unresolved == 0.0) | (unresolved == 1.0)))
        self.assertTrue(torch.all((current_wrong_candidate == 0.0) | (current_wrong_candidate == 1.0)))
        self.assertTrue(torch.all((wrong_candidate_history == 0.0) | (wrong_candidate_history == 1.0)))
        self.assertTrue(torch.all((revisit_unresolved == 0.0) | (revisit_unresolved == 1.0)))
        self.assertTrue(torch.all((allocation_error == 0.0) | (allocation_error == 1.0)))
        self.assertTrue(torch.allclose(relevant + unresolved, torch.ones_like(relevant)))
        self.assertTrue(torch.all(current_wrong_candidate <= wrong_candidate_history))
        self.assertTrue(torch.all(revisit_unresolved <= unresolved))
        self.assertLess(allocation_error.float().mean().item(), 0.95)

    def test_collect_nl_examples_exports_structured_state(self) -> None:
        batch = generate_batch(2, self.task_cfg.num_steps, self.task_cfg)
        model = RecurrentAttentionController(self.task_cfg, self.model_cfg)
        outputs = model(batch.scene, batch.cue, target=batch.target, num_steps=self.task_cfg.num_steps)
        examples = collect_nl_examples(model, self.task_cfg, batch, outputs)

        self.assertEqual(len(examples), 2 * self.task_cfg.num_steps)
        example = examples[0]
        self.assertTrue(example.example_id.startswith("scene0_step0"))
        self.assertIn("search_type=", example.symbolic_state)
        self.assertIn("previous_search_type=", example.symbolic_state)
        self.assertIn("cue_switched=", example.symbolic_state)
        self.assertIn("attended_cell=", example.symbolic_state)
        self.assertIn("attended_visible_type=", example.symbolic_state)
        self.assertIn("attended_digit=", example.symbolic_state)
        self.assertIn("glimpse_digit=", example.symbolic_state)
        self.assertIn("previous_attended_cell=", example.symbolic_state)
        self.assertIn("previous_attended_visible_type=", example.symbolic_state)
        self.assertIn("previous_attended_digit=", example.symbolic_state)
        self.assertIn("previous_glimpse_digit=", example.symbolic_state)
        self.assertIn("previous_found_target=", example.symbolic_state)
        self.assertIn("relevant_region_inspected=", example.symbolic_state)
        self.assertIn("unresolved_search=", example.symbolic_state)
        self.assertIn("current_wrong_candidate=", example.symbolic_state)
        self.assertIn("wrong_candidate_history=", example.symbolic_state)
        self.assertIn("revisit_unresolved=", example.symbolic_state)
        self.assertIn("allocation_error=", example.symbolic_state)
        self.assertIn("inspected_count=", example.symbolic_state)
        self.assertIn("previous_inspected_count=", example.symbolic_state)
        self.assertIn("attended_cell_previously_inspected=", example.symbolic_state)
        self.assertIn("unresolved_rows=", example.symbolic_state)
        self.assertIn("unresolved_cols=", example.symbolic_state)
        self.assertIn("unresolved_count=", example.symbolic_state)
        self.assertIn("cue=", example.observation_only)
        self.assertIn("previous_cue=", example.observation_only)
        self.assertIn("cue_switched=not_available", example.observation_only)
        self.assertIn("current_attention_location=not_available", example.observation_only)
        self.assertIn("current_attention_content=not_available", example.observation_only)
        self.assertIn("current_wrong_candidate=not_available", example.observation_only)
        self.assertIn("wrong_candidate_history=not_available", example.observation_only)
        self.assertIn("revisit_unresolved=not_available", example.observation_only)
        self.assertTrue(example.tokenized_state.startswith("x"))
        token_list = example.tokenized_state.split()
        self.assertIn("x900", token_list)
        self.assertIn("x901", token_list)
        self.assertIn("x910", token_list)
        self.assertIn("x920", token_list)
        self.assertIn("x930", token_list)
        self.assertIsInstance(example.unresolved_cells, list)
        self.assertIsInstance(example.unresolved_rows, list)
        self.assertIsInstance(example.unresolved_cols, list)
        self.assertIsInstance(example.unresolved_count, int)
        self.assertGreaterEqual(len(example.unresolved_cells), 0)
        self.assertIsInstance(example.attended_visible_type, int)
        self.assertIsInstance(example.attended_digit, int)
        self.assertIsInstance(example.glimpse_digit, int)
        self.assertIsInstance(example.prev_attended_cell, int)
        self.assertIsInstance(example.prev_attended_visible_type, int)
        self.assertIsInstance(example.prev_attended_digit, int)
        self.assertIsInstance(example.prev_glimpse_digit, int)
        self.assertIsInstance(example.glimpse_target_match, bool)
        self.assertIsInstance(example.previous_cue, int)
        self.assertIsInstance(example.cue_switched, bool)
        self.assertIsInstance(example.previous_found_target, bool)
        self.assertIsInstance(example.relevant_region_inspected, bool)
        self.assertIsInstance(example.unresolved_search, bool)
        self.assertIsInstance(example.current_wrong_candidate, bool)
        self.assertIsInstance(example.wrong_candidate_history, bool)
        self.assertIsInstance(example.revisit_unresolved, bool)
        self.assertIsInstance(example.allocation_error, bool)
        self.assertIsInstance(example.inspected_count, int)
        self.assertIsInstance(example.previous_inspected_count, int)
        self.assertIsInstance(example.attended_cell_previously_inspected, bool)

    def test_extract_response_json_accepts_multiple_sdk_shapes(self) -> None:
        class FakeContent:
            def __init__(self, *, text=None, parsed=None) -> None:
                self.text = text
                self.parsed = parsed

        class FakeItem:
            def __init__(self, *, content=None, parsed=None) -> None:
                self.content = content or []
                self.parsed = parsed

        class FakeResponse:
            def __init__(self, *, output_text="", output=None, output_parsed=None) -> None:
                self.output_text = output_text
                self.output = output or []
                self.output_parsed = output_parsed
                self.status = "completed"

        expected = {"search_type": 1}
        self.assertEqual(_extract_response_json(FakeResponse(output_parsed=expected)), expected)
        self.assertEqual(
            _extract_response_json(FakeResponse(output_text='{"search_type": 1}')),
            expected,
        )
        self.assertEqual(
            _extract_response_json(FakeResponse(output=[FakeItem(parsed=expected)])),
            expected,
        )
        self.assertEqual(
            _extract_response_json(
                FakeResponse(output=[FakeItem(content=[FakeContent(parsed=expected)])])
            ),
            expected,
        )
        self.assertEqual(
            _extract_response_json(
                FakeResponse(output=[FakeItem(content=[FakeContent(text='{"search_type": 1}')])])
            ),
            expected,
        )

    def test_collect_cue_switch_nl_examples_marks_post_switch_steps(self) -> None:
        batch = generate_batch(2, self.task_cfg.num_steps, self.task_cfg)
        model = RecurrentAttentionController(self.task_cfg, self.model_cfg)
        examples = collect_cue_switch_nl_examples(
            model,
            self.task_cfg,
            batch,
            switch_step=1,
        )

        self.assertEqual(len(examples), 2 * self.task_cfg.num_steps)
        pre_switch = [example for example in examples if example.step_index == 0]
        switch_transition = [example for example in examples if example.step_index == 1]
        post_switch = [example for example in examples if example.step_index >= 1]
        self.assertTrue(pre_switch)
        self.assertTrue(switch_transition)
        self.assertTrue(post_switch)
        self.assertTrue(all(example.example_id.startswith("cue_switch_") for example in examples))
        self.assertTrue(all(not example.cue_switched for example in pre_switch))
        self.assertTrue(all(example.cue_switched for example in switch_transition))
        self.assertTrue(
            all(
                example.cue != int(batch.cue[int(example.example_id.split("scene", 1)[1].split("_", 1)[0])].item())
                for example in post_switch
            )
        )

    def test_collect_intervention_nl_examples_returns_aligned_pairs(self) -> None:
        batch = generate_batch(2, self.task_cfg.num_steps, self.task_cfg)
        model = RecurrentAttentionController(self.task_cfg, self.model_cfg)
        result = collect_intervention_nl_examples(
            model,
            self.task_cfg,
            batch,
            intervention_step=1,
        )

        baseline_examples = result["baseline_examples"]
        intervened_examples = result["intervened_examples"]
        self.assertEqual(len(baseline_examples), 2 * self.task_cfg.num_steps)
        self.assertEqual(len(intervened_examples), len(baseline_examples))
        self.assertEqual(result["intervention_step"], 1)
        self.assertGreater(result["delta_norm"], 0.0)
        self.assertTrue(0.0 <= result["attention_change_fraction"] <= 1.0)
        self.assertTrue(
            all(example.example_id.startswith("baseline_intervention_") for example in baseline_examples)
        )
        self.assertTrue(
            all(example.example_id.startswith("intervened_intervention_") for example in intervened_examples)
        )
        self.assertEqual(
            [example.step_index for example in baseline_examples],
            [example.step_index for example in intervened_examples],
        )

    def test_build_evidence_summary_surfaces_nl_uncertainty_metrics(self) -> None:
        summary = build_evidence_summary(
            {
                "baseline": {
                    "accuracy": 0.1,
                    "temporal_reallocation": 0.0,
                    "target_attention_gain": 0.0,
                    "cue_accuracy_delta": 0.0,
                    "cue_target_attention_delta": 0.0,
                },
                "recurrent": {
                    "accuracy": 0.2,
                    "temporal_reallocation": 0.1,
                    "target_attention_gain": 0.1,
                    "cue_accuracy_delta": 0.1,
                    "cue_target_attention_delta": 0.1,
                },
                "ablations": {
                    "freeze_recurrence": {
                        "accuracy": 0.1,
                        "temporal_reallocation": 0.0,
                        "target_attention_gain": 0.0,
                    }
                },
                "stage3_multi_seed": {
                    "num_seeds": 3,
                    "predictive_supported_fraction": 2.0 / 3.0,
                    "intervention_supported_fraction": 1.0 / 3.0,
                    "all_predictive_supported": False,
                    "all_intervention_supported": False,
                    "supported": False,
                },
                "predictive_probe": {
                    "supported": True,
                },
                "intervention_test": {
                    "supported": True,
                },
                "reduced_shaping": {
                    "summary": {
                        "supported": True,
                    }
                },
                "nl_report": {
                    "model": "gpt-5-mini",
                    "tokenized_joint_accuracy_advantage": 0.25,
                    "tokenized_uncertainty_content_joint_accuracy_advantage": 0.5,
                    "tokenized_relevant_region_accuracy_advantage": 0.25,
                    "tokenized_unresolved_search_accuracy_advantage": 0.25,
                    "tokenized_current_wrong_candidate_accuracy_advantage": 0.4,
                    "tokenized_wrong_candidate_history_accuracy_advantage": 0.5,
                    "tokenized_revisit_unresolved_accuracy_advantage": 0.3,
                    "tokenized_allocation_error_accuracy_advantage": 0.25,
                    "content_supported": True,
                    "supported": True,
                    "tokenized_state": {"joint_accuracy": 0.75},
                    "observation_only": {"joint_accuracy": 0.25},
                    "cue_switch_slice": {
                        "supported": True,
                        "tokenized_joint_accuracy_advantage": 0.1,
                        "tokenized_memory_content_joint_accuracy_advantage": 0.2,
                    },
                    "intervention_slice": {
                        "supported": False,
                        "tokenized_joint_accuracy_advantage": 0.05,
                        "tokenized_memory_content_joint_accuracy_advantage": 0.15,
                        "attention_change_fraction": 0.5,
                    },
                }
                ,
                "uncertainty_report_probes": {
                    "relevant_region_inspected": {
                        "native_accuracy_advantage": 0.0,
                        "native_positive_recall_advantage": 0.0,
                    },
                    "unresolved_search": {
                        "native_accuracy_advantage": 0.1,
                        "native_positive_recall_advantage": 0.1,
                    },
                    "current_wrong_candidate": {
                        "native_accuracy_advantage": 0.2,
                        "native_positive_recall_advantage": 0.3,
                    },
                    "wrong_candidate_history": {
                        "native_accuracy_advantage": 0.1,
                        "native_positive_recall_advantage": 0.4,
                    },
                    "revisit_unresolved": {
                        "native_accuracy_advantage": 0.05,
                        "native_positive_recall_advantage": 0.2,
                    },
                    "allocation_error": {
                        "native_accuracy_advantage": 0.0,
                        "native_positive_recall_advantage": 0.1,
                    },
                    "supported": True,
                },
            }
        )
        nl_summary = summary["natural_language_reportability"]
        self.assertEqual(
            nl_summary["tokenized_uncertainty_content_joint_accuracy_advantage"],
            0.5,
        )
        self.assertEqual(nl_summary["tokenized_relevant_region_accuracy_advantage"], 0.25)
        self.assertEqual(nl_summary["tokenized_unresolved_search_accuracy_advantage"], 0.25)
        self.assertEqual(
            nl_summary["tokenized_current_wrong_candidate_accuracy_advantage"],
            0.4,
        )
        self.assertEqual(
            nl_summary["tokenized_wrong_candidate_history_accuracy_advantage"],
            0.5,
        )
        self.assertEqual(
            nl_summary["tokenized_revisit_unresolved_accuracy_advantage"],
            0.3,
        )
        self.assertEqual(nl_summary["tokenized_allocation_error_accuracy_advantage"], 0.25)
        self.assertTrue(nl_summary["cue_switch_slice_supported"])
        self.assertEqual(nl_summary["cue_switch_tokenized_joint_accuracy_advantage"], 0.1)
        self.assertEqual(
            nl_summary["cue_switch_tokenized_memory_content_joint_accuracy_advantage"],
            0.2,
        )
        self.assertFalse(nl_summary["intervention_slice_supported"])
        self.assertEqual(nl_summary["intervention_tokenized_joint_accuracy_advantage"], 0.05)
        self.assertEqual(
            nl_summary["intervention_tokenized_memory_content_joint_accuracy_advantage"],
            0.15,
        )
        self.assertEqual(nl_summary["intervention_attention_change_fraction"], 0.5)
        self.assertTrue(nl_summary["supported"])
        explicit_attention = summary["explicit_attention_modeling"]
        self.assertEqual(explicit_attention["stage3_num_seeds"], 3)
        self.assertEqual(explicit_attention["stage3_predictive_supported_fraction"], 2.0 / 3.0)
        self.assertEqual(explicit_attention["stage3_intervention_supported_fraction"], 1.0 / 3.0)
        self.assertFalse(explicit_attention["stage3_all_predictive_supported"])
        self.assertFalse(explicit_attention["stage3_all_intervention_supported"])
        self.assertFalse(explicit_attention["stage3_multi_seed_supported"])
        self.assertTrue(explicit_attention["single_run_supported"])
        self.assertFalse(explicit_attention["robust_supported"])
        self.assertFalse(explicit_attention["supported"])
        stage6b = summary["structured_reportability_uncertainty_and_allocation_error"]
        self.assertTrue(stage6b["supported"])

    def test_build_stage3_summary_separates_single_run_and_robust(self) -> None:
        summary = build_stage3_summary(
            {
                "predictive_probe": {"supported": True},
                "intervention_test": {"supported": True},
                "reduced_shaping": {"summary": {"supported": True}},
                "stage3_multi_seed": {
                    "supported": False,
                    "predictive_supported_fraction": 2.0 / 3.0,
                    "intervention_supported_fraction": 1.0 / 3.0,
                    "failure_reasons": [
                        "predictive_probe_instability",
                        "intervention_instability",
                    ],
                    "predictive_cross_entropy_advantage_min_gap": -0.01,
                    "predictive_top1_advantage_min_gap": -0.25,
                    "intervention_attention_change_kl_min_gap": -0.005,
                    "intervention_alternate_target_gain_min_gap": -0.01,
                },
            }
        )
        self.assertTrue(summary["single_run_supported"])
        self.assertFalse(summary["robust_supported"])
        self.assertEqual(len(summary["failure_reasons"]), 2)
        self.assertEqual(summary["predictive_supported_fraction"], 2.0 / 3.0)

    def test_run_nl_report_mode_scores_stage6b_fields(self) -> None:
        batch = generate_batch(2, self.task_cfg.num_steps, self.task_cfg)
        model = RecurrentAttentionController(self.task_cfg, self.model_cfg)
        with torch.no_grad():
            outputs = model(
                batch.scene,
                batch.cue,
                target=batch.target,
                target_pos=batch.target_pos,
                num_steps=self.task_cfg.num_steps,
            )
        examples = [example for example in collect_nl_examples(model, self.task_cfg, batch, outputs) if example.step_index > 0]
        calibration_examples = examples[:2]
        evaluation_examples = examples[2:4]

        def expected_payload(example):
            return {
                "natural_language_report": "faithful summary",
                "search_type": example.cue,
                "previous_search_type": example.previous_cue,
                "cue_switched": example.cue_switched,
                "attended_cell": list(divmod(example.attended_cell, self.task_cfg.grid_size)),
                "attended_visible_type": example.attended_visible_type,
                "attended_digit": example.attended_digit,
                "glimpse_digit": example.glimpse_digit,
                "previous_attended_cell": list(divmod(example.prev_attended_cell, self.task_cfg.grid_size)),
                "previous_attended_visible_type": example.prev_attended_visible_type,
                "previous_attended_digit": example.prev_attended_digit,
                "previous_glimpse_digit": example.prev_glimpse_digit,
                "glimpse_target_match": example.glimpse_target_match,
                "previous_found_target": example.previous_found_target,
                "found_target": example.found_target,
                "relevant_region_inspected": example.relevant_region_inspected,
                "unresolved_search": example.unresolved_search,
                "current_wrong_candidate": example.current_wrong_candidate,
                "wrong_candidate_history": example.wrong_candidate_history,
                "revisit_unresolved": example.revisit_unresolved,
                "allocation_error": example.allocation_error,
                "inspected_count": example.inspected_count,
                "previous_inspected_count": example.previous_inspected_count,
                "attended_cell_previously_inspected": example.attended_cell_previously_inspected,
                "unresolved_rows": example.unresolved_rows,
                "unresolved_cols": example.unresolved_cols,
                "unresolved_count": example.unresolved_count,
            }

        class FakeResponse:
            def __init__(self, payload) -> None:
                self.output_text = ""
                self.output = []
                self.output_parsed = payload
                self.status = "completed"

        class FakeResponses:
            def __init__(self, payloads) -> None:
                self._payloads = payloads

            def create(self, **kwargs):
                return FakeResponse(self._payloads.pop(0))

        class FakeOpenAI:
            payloads = []

            def __init__(self, **kwargs) -> None:
                self.responses = FakeResponses(type(self).payloads)

        original_openai = nl_report_module.OpenAI
        nl_report_module.OpenAI = FakeOpenAI
        FakeOpenAI.payloads = [expected_payload(example) for example in evaluation_examples]
        try:
            metrics = run_nl_report_mode(
                mode="symbolic_state",
                model_name="fake-model",
                calibration_examples=calibration_examples,
                evaluation_examples=evaluation_examples,
                grid_size=self.task_cfg.grid_size,
                max_output_tokens=128,
            )
        finally:
            nl_report_module.OpenAI = original_openai

        self.assertEqual(metrics["relevant_region_inspected_accuracy"], 1.0)
        self.assertEqual(metrics["unresolved_search_accuracy"], 1.0)
        self.assertEqual(metrics["current_wrong_candidate_accuracy"], 1.0)
        self.assertEqual(metrics["wrong_candidate_history_accuracy"], 1.0)
        self.assertEqual(metrics["revisit_unresolved_accuracy"], 1.0)
        self.assertEqual(metrics["allocation_error_accuracy"], 1.0)
        self.assertEqual(metrics["previous_search_type_accuracy"], 1.0)
        self.assertEqual(metrics["cue_switched_accuracy"], 1.0)
        self.assertEqual(metrics["previous_found_target_accuracy"], 1.0)
        self.assertEqual(metrics["inspected_count_accuracy"], 1.0)
        self.assertEqual(metrics["previous_inspected_count_accuracy"], 1.0)
        self.assertEqual(metrics["attended_cell_previously_inspected_accuracy"], 1.0)
        self.assertEqual(metrics["uncertainty_content_joint_accuracy"], 1.0)
        self.assertEqual(metrics["joint_accuracy"], 1.0)

    def test_nl_report_metrics_includes_cue_switch_and_intervention_slices(self) -> None:
        model = RecurrentAttentionController(self.task_cfg, self.model_cfg)
        config = {
            "evaluation": {
                "probe_scenes": 2,
                "cue_switch": {
                    "enabled": True,
                    "switch_step": 1,
                },
                "intervention_test": {
                    "enabled": True,
                    "step": 1,
                },
                "nl_report": {
                    "enabled": True,
                    "calibration_examples": 1,
                    "evaluation_examples": 1,
                    "translator_train_examples": 1,
                    "probe_scenes": 2,
                    "max_output_tokens": 64,
                },
            }
        }

        original_openai = eval_module.OpenAI
        original_run = eval_module.run_nl_report_mode
        original_api_key = os.environ.get("OPENAI_API_KEY")

        def fake_run_nl_report_mode(**kwargs):
            return {
                "mode": kwargs["mode"],
                "joint_accuracy": 1.0,
                "current_content_joint_accuracy": 1.0,
                "memory_content_joint_accuracy": 1.0,
                "content_only_joint_accuracy": 1.0,
                "attended_visible_type_accuracy": 1.0,
                "attended_digit_accuracy": 1.0,
                "glimpse_digit_accuracy": 1.0,
                "previous_attended_cell_accuracy": 1.0,
                "previous_attended_visible_type_accuracy": 1.0,
                "previous_attended_digit_accuracy": 1.0,
                "previous_glimpse_digit_accuracy": 1.0,
                "glimpse_target_match_accuracy": 1.0,
                "relevant_region_inspected_accuracy": 1.0,
                "unresolved_search_accuracy": 1.0,
                "current_wrong_candidate_accuracy": 1.0,
                "wrong_candidate_history_accuracy": 1.0,
                "revisit_unresolved_accuracy": 1.0,
                "allocation_error_accuracy": 1.0,
                "uncertainty_content_joint_accuracy": 1.0,
                "unresolved_rows_accuracy": 1.0,
                "unresolved_cols_accuracy": 1.0,
                "unresolved_count_accuracy": 1.0,
            }

        eval_module.OpenAI = object()
        eval_module.run_nl_report_mode = fake_run_nl_report_mode
        os.environ["OPENAI_API_KEY"] = "test-key"
        try:
            metrics = nl_report_metrics(
                model,
                config,
                self.task_cfg,
                torch.device("cpu"),
                seed=23,
            )
        finally:
            eval_module.OpenAI = original_openai
            eval_module.run_nl_report_mode = original_run
            if original_api_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = original_api_key

        self.assertFalse(metrics["skipped"])
        self.assertIn("cue_switch_slice", metrics)
        self.assertIn("intervention_slice", metrics)
        self.assertFalse(metrics["cue_switch_slice"]["skipped"])
        self.assertFalse(metrics["intervention_slice"]["skipped"])
        self.assertGreater(metrics["intervention_slice"]["delta_norm"], 0.0)
        self.assertTrue(0.0 <= metrics["intervention_slice"]["attention_change_fraction"] <= 1.0)

    def test_self_state_diagnostics_exposes_stepwise_series(self) -> None:
        model = RecurrentAttentionController(self.task_cfg, self.model_cfg)
        diagnostics = self_state_diagnostics(
            model,
            self.task_cfg,
            batch_size=2,
            device=torch.device("cpu"),
            seed=11,
        )
        expected_keys = {
            "inspection_coverage_by_step",
            "found_state_rate_by_step",
            "relevant_region_rate_by_step",
            "unresolved_search_rate_by_step",
            "current_wrong_candidate_rate_by_step",
            "wrong_candidate_history_rate_by_step",
            "revisit_unresolved_rate_by_step",
            "allocation_error_rate_by_step",
            "self_model_mass_on_inspected_cells_by_step",
            "self_model_mass_on_uninspected_cells_by_step",
        }
        self.assertEqual(set(diagnostics), expected_keys)
        for key, values in diagnostics.items():
            self.assertEqual(len(values), self.task_cfg.num_steps, key)
            for value in values:
                self.assertIsInstance(value, float, key)
        bounded_series = (
            "inspection_coverage_by_step",
            "found_state_rate_by_step",
            "relevant_region_rate_by_step",
            "unresolved_search_rate_by_step",
            "current_wrong_candidate_rate_by_step",
            "wrong_candidate_history_rate_by_step",
            "revisit_unresolved_rate_by_step",
            "allocation_error_rate_by_step",
        )
        for key in bounded_series:
            self.assertTrue(all(0.0 <= value <= 1.0 for value in diagnostics[key]), key)

    def test_self_model_diagnostics_exposes_stepwise_series(self) -> None:
        model = RecurrentAttentionController(self.task_cfg, self.model_cfg)
        diagnostics = self_model_diagnostics(
            model,
            self.task_cfg,
            batch_size=2,
            device=torch.device("cpu"),
            seed=13,
        )
        expected_keys = {
            "target_self_model_mass_by_step",
            "target_inspection_state_by_step",
            "target_attention_by_step",
            "self_model_cell_error_by_step",
            "target_self_model_alignment_by_step",
        }
        self.assertEqual(set(diagnostics), expected_keys)
        for key, values in diagnostics.items():
            self.assertEqual(len(values), self.task_cfg.num_steps, key)
            for value in values:
                self.assertIsInstance(value, float, key)
        bounded_series = (
            "target_self_model_mass_by_step",
            "target_inspection_state_by_step",
            "target_attention_by_step",
            "target_self_model_alignment_by_step",
        )
        for key in bounded_series:
            self.assertTrue(all(0.0 <= value <= 1.0 for value in diagnostics[key]), key)

    def test_stage3_multi_seed_metrics_exposes_seed_runs(self) -> None:
        model = RecurrentAttentionController(self.task_cfg, self.model_cfg)
        config = {
            "training": {
                "batch_size": 4,
            },
            "evaluation": {
                "probe_scenes": 2,
                "predictive_probe": {
                    "train_batches": 1,
                    "test_batches": 1,
                    "epochs": 2,
                    "learning_rate": 0.05,
                    "thresholds": {
                        "min_advantage_cross_entropy": -1.0,
                        "min_advantage_mse": -1.0,
                        "min_advantage_top1_match": -1.0,
                    },
                },
                "intervention_test": {
                    "enabled": True,
                    "probe_scenes": 2,
                    "step": 1,
                    "thresholds": {
                        "min_attention_change_kl": -1.0,
                        "min_original_target_attention_drop": -1.0,
                        "min_alternate_target_attention_gain": -1.0,
                    },
                },
                "stage3_multi_seed": {
                    "enabled": True,
                    "num_seeds": 2,
                    "seed_stride": 10,
                },
            }
        }
        metrics = stage3_multi_seed_metrics(
            model,
            config,
            self.task_cfg,
            torch.device("cpu"),
            seed=17,
            reduced_shaping_summary={"supported": True},
        )
        self.assertEqual(metrics["num_seeds"], 2)
        self.assertEqual(len(metrics["predictive_probe_runs"]), 2)
        self.assertEqual(len(metrics["intervention_runs"]), 2)
        self.assertTrue(metrics["reduced_shaping_supported"])
        self.assertIn("predictive_cross_entropy_advantage_mean", metrics)
        self.assertIn("predictive_cross_entropy_advantage_min", metrics)
        self.assertIn("predictive_top1_advantage_mean", metrics)
        self.assertIn("predictive_top1_advantage_min", metrics)
        self.assertIn("intervention_attention_change_kl_mean", metrics)
        self.assertIn("intervention_attention_change_kl_min", metrics)
        self.assertIn("intervention_alternate_target_gain_mean", metrics)
        self.assertIn("intervention_alternate_target_gain_min", metrics)
        self.assertIn("predictive_cross_entropy_advantage_min_gap", metrics)
        self.assertIn("predictive_top1_advantage_min_gap", metrics)
        self.assertIn("intervention_attention_change_kl_min_gap", metrics)
        self.assertIn("intervention_alternate_target_gain_min_gap", metrics)
        self.assertIn("failure_reasons", metrics)
        self.assertTrue(0.0 <= metrics["predictive_supported_fraction"] <= 1.0)
        self.assertTrue(0.0 <= metrics["intervention_supported_fraction"] <= 1.0)

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
                        "thresholds": {
                            "min_advantage_cross_entropy": 0.0,
                            "min_advantage_mse": 0.0,
                            "min_advantage_top1_match": 0.0,
                        },
                    },
                    "intervention_test": {
                        "enabled": True,
                        "probe_scenes": 2,
                        "step": 1,
                        "thresholds": {
                            "min_attention_change_kl": 0.0,
                            "min_original_target_attention_drop": 0.0,
                            "min_alternate_target_attention_gain": 0.0,
                        },
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
                    "uncertainty_report_probes": {
                        "enabled": True,
                        "train_batches": 2,
                        "test_batches": 1,
                        "epochs": 10,
                        "learning_rate": 0.05,
                    },
                    "nl_report": {
                        "enabled": False,
                    },
                    "reduced_shaping": {
                        "enabled": True,
                        "weights": [0.0],
                        "thresholds": {
                            "min_accuracy": 0.0,
                            "min_temporal_reallocation": 0.0,
                            "min_target_attention_gain": -1.0,
                        },
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
            self.assertIn("uncertainty_report_probes", report)
            self.assertIn("nl_report", report)
            self.assertIn("cue_switch", report)
            self.assertIn("intervention_test", report)
            self.assertIn("reduced_shaping", report)
            self.assertIn("stage3_summary", report)
            self.assertIn("self_state_diagnostics", report)
            self.assertIn("self_model_diagnostics", report)
            self.assertIn("stage7_visual_report_summary", report)
            self.assertIn("stage3_multi_seed", report)
            self.assertIn("controller_state_probe", report["predictive_probe"])
            self.assertIn("observation_only_probe", report["predictive_probe"])
            self.assertIn("current_search_type", report["report_probes"])
            self.assertIn("target_inspected_report", report["self_modeling"])
            self.assertIn("observation_only_probe", report["self_modeling"]["native_cell_report"])
            self.assertIn(
                "relevant_region_inspected",
                report["uncertainty_report_probes"],
            )
            self.assertIn("current_wrong_candidate", report["uncertainty_report_probes"])
            self.assertIn("wrong_candidate_history", report["uncertainty_report_probes"])
            self.assertIn("revisit_unresolved", report["uncertainty_report_probes"])
            self.assertIn("allocation_error", report["uncertainty_report_probes"])
            self.assertIn("recurrent", report["cue_switch"])
            self.assertIn("explicit_attention_modeling", report["evidence"])
            self.assertIn("engineered_self_state_tracking", report["evidence"])
            self.assertIn("learned_self_modeling_of_attention", report["evidence"])
            self.assertIn("structured_reportability", report["evidence"])
            self.assertIn(
                "structured_reportability_uncertainty_and_allocation_error",
                report["evidence"],
            )
            self.assertIn("natural_language_reportability", report["evidence"])
            self.assertIn("cue_switch_adaptation", report["evidence"])
            self.assertIn("causal_attention_intervention", report["evidence"])
            self.assertIn("reduced_shaping_resilience", report["evidence"])
            explicit_attention = report["evidence"]["explicit_attention_modeling"]
            self.assertIn("intervention_supported", explicit_attention)
            self.assertIn("reduced_shaping_supported", explicit_attention)
            self.assertEqual(
                explicit_attention["supported"],
                (
                    report["predictive_probe"]["supported"]
                    and report["intervention_test"]["supported"]
                    and report["reduced_shaping"]["summary"]["supported"]
                ),
            )
            self.assertTrue(Path(report["artifacts"]["report"]).exists())
            self.assertTrue(report["artifacts"]["plots"])
            self.assertTrue(report["artifacts"]["self_state_plots"])
            self.assertTrue(Path(report["artifacts"]["self_state_plots"][0]).exists())
            self.assertTrue(report["artifacts"]["self_model_plots"])
            self.assertTrue(Path(report["artifacts"]["self_model_plots"][0]).exists())
            self.assertTrue(report["artifacts"]["uncertainty_plots"])
            self.assertTrue(Path(report["artifacts"]["uncertainty_plots"][0]).exists())
            self.assertTrue(report["artifacts"]["cue_switch_plots"])
            self.assertTrue(Path(report["artifacts"]["cue_switch_plots"][0]).exists())
            self.assertTrue(report["artifacts"]["stage3_multi_seed_plots"])
            self.assertTrue(Path(report["artifacts"]["stage3_multi_seed_plots"][0]).exists())
            self.assertTrue(Path(report["artifacts"]["stage3_summary"]).exists())
            self.assertTrue(report["artifacts"]["stage7_visual_reports"])
            self.assertTrue(Path(report["artifacts"]["stage7_visual_reports"][0]).exists())
            self.assertTrue(Path(report["artifacts"]["stage7_visual_report_metadata"]).exists())
            self.assertGreaterEqual(report["stage7_visual_report_summary"]["num_examples"], 1)


if __name__ == "__main__":
    unittest.main()
