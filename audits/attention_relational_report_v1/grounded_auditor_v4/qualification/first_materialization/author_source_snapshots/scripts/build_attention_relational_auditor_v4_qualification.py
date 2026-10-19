#!/usr/bin/env python3
"""Author-owned fictional materials, independent of the v4 auditor implementation.

This module reads no experimental archive, report, annotation, model or new v4
auditor code.  It does not construct API requests.  ``build_materials`` returns
public cases and a SEPARATE hidden semantic key.  Only the public case fields may
be used when constructing auditor requests.  The key, this source and the author
notes must not be supplied to qualification auditors.

The CLI is an exclusive, once-only materialization for subsequent preservation
and independent review.  An unsuccessful first materialization is retained, not
overwritten.  Running this builder is not permission to call an auditor or to
start target measurement.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]
PACKET_SCHEMA = "attention_relational_report_v1_packet"
PUBLIC_SCHEMA = "attention_relational_auditor_v4_fictional_cases_v1"
KEY_SCHEMA = "attention_relational_auditor_v4_hidden_atomic_key_v1"
DESIGN = "docs/ATTENTION_RELATIONAL_GROUNDED_AUDITOR_V4_DESIGN.md"
OLD_EXPORTER = "scripts/attention_relational_report_v1.py"
AUTHOR_NOTES = (
    "audits/attention_relational_report_v1/grounded_auditor_v4/"
    "qualification_author_notes.md"
)
DEFAULT_OUTPUT = (
    "audits/attention_relational_report_v1/grounded_auditor_v4/"
    "qualification/first_materialization"
)
NODES = ("n0", "n1", "n2", "n3")
ROLES = ("red", "blue")
COMMANDS = ("k0", "k1", "k2")
VERDICTS = ("supported", "contradicted", "not_auditable")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def grid(*weighted_cells):
    """Hand-authored dyadic masses; no sampling, fit, rounding or renormalizing."""
    result = [[0.0 for _ in range(4)] for _ in range(4)]
    occupied = set()
    for row, column, probability in weighted_cells:
        require(type(row) is int and type(column) is int and
                row in range(4) and column in range(4), "invalid fictional cell")
        require((row, column) not in occupied, "duplicate fictional cell")
        occupied.add((row, column))
        require(type(probability) in (int, float) and
                math.isfinite(probability) and 0 < probability <= 1,
                "invalid fictional probability")
        result[row][column] = float(probability)
    flat = [value for line in result for value in line]
    require(math.fsum(flat) == 1.0, "fictional distribution must sum exactly to one")
    require(list(struct.unpack("<16f", struct.pack("<16f", *flat))) == flat,
            "fictional maps must already be exact float32 numbers")
    return result


def point(row, column):
    return grid((row, column, 1.0))


def uniform():
    return grid(*[(row, column, 1.0 / 16.0)
                  for row in range(4) for column in range(4)])


def localization_summary(values):
    """Same transparent calculation as the old neutral packet, without imports."""
    flat = [value for line in values for value in line]
    peak = max(flat)
    return dict(
        peaks=[[index // 4, index % 4]
               for index, value in enumerate(flat) if value == peak],
        peak_probability=peak,
        entropy_nats=-math.fsum(value * math.log(value)
                               for value in flat if value > 0.0),
        expected_row=math.fsum((index // 4) * value
                               for index, value in enumerate(flat)),
        expected_column=math.fsum((index % 4) * value
                                  for index, value in enumerate(flat)),
    )


def peak_relation(red, blue):
    horizontal, vertical, coincidences = set(), set(), []
    for r in red:
        for b in blue:
            horizontal.add("left_of" if r[1] < b[1] else
                           "right_of" if r[1] > b[1] else "same_column")
            vertical.add("above" if r[0] < b[0] else
                         "below" if r[0] > b[0] else "same_row")
            coincidences.append(r == b)
    return dict(
        horizontal=[name for name in ("left_of", "same_column", "right_of")
                    if name in horizontal],
        vertical=[name for name in ("above", "same_row", "below")
                  if name in vertical],
        coincident_possible=any(coincidences),
        coincident_all=all(coincidences),
    )


def current_origin(panel, node):
    # These are fictional declared origin kinds, not real source identities.
    if panel == "p1":
        return dict(kind="completed_record", record_time=12, represented_time=12)
    if node == "n3":
        return dict(kind="history_estimate", represented_time=12, feedback_through=11)
    return dict(kind="copied_prior_record", record_time=11, represented_time=12)


def origin(panel, node, view):
    if view == "current":
        return current_origin(panel, node)
    if node == "n1":
        if panel == "p1":
            return dict(kind="native_conditional_record", represented_time=13,
                        executed_branch_claim=False)
        return dict(kind="command_estimate", represented_time=13, feedback_through=11)
    return dict(kind="inherited_current", represented_time=13,
                inherited_origin=current_origin(panel, node))


def node_record(panel, node, view, pair):
    distributions = {role: deepcopy(pair[index]) for index, role in enumerate(ROLES)}
    derived = {role: localization_summary(distributions[role]) for role in ROLES}
    return dict(node=node, origin=origin(panel, node, view),
                distributions=distributions, derived=derived,
                red_blue_peak_relation=peak_relation(derived["red"]["peaks"],
                                                     derived["blue"]["peaks"]))


def fictional_packet():
    """A full neutral public packet; never a projection of an actual target."""
    current = {
        "p0": [
            (point(0, 0), point(0, 3)),
            (grid((0, 0, 0.5), (2, 0, 0.5)), point(1, 3)),
            (grid((1, 1, 0.75), (3, 1, 0.25)), point(2, 2)),
            (point(3, 1), grid((1, 2, 0.5), (3, 2, 0.5))),
        ],
        "p1": [
            (point(3, 3), point(3, 0)),
            (point(2, 1), point(0, 1)),
            (uniform(), point(1, 2)),
            (point(2, 0), point(2, 3)),
        ],
    }
    candidate_n1 = {
        "p0": {
            "k0": (point(3, 0), point(0, 1)),
            "k1": (uniform(), point(2, 2)),
            "k2": (grid((0, 2, 0.75), (3, 2, 0.25)),
                   grid((1, 0, 0.5), (1, 3, 0.5))),
        },
        "p1": {
            "k0": (point(3, 2), point(1, 0)),
            "k1": (point(0, 0), point(3, 3)),
            "k2": (grid((0, 1, 0.5), (3, 1, 0.5)), point(0, 2)),
        },
    }
    panels = []
    for panel in ("p0", "p1"):
        candidates = {}
        for command in COMMANDS:
            pairs = deepcopy(current[panel])
            pairs[1] = candidate_n1[panel][command]
            candidates[command] = [node_record(panel, name, "candidate", pairs[index])
                                   for index, name in enumerate(NODES)]
        panels.append(dict(
            panel=panel,
            current=[node_record(panel, name, "current", current[panel][index])
                     for index, name in enumerate(NODES)],
            candidates=candidates,
        ))
    return dict(
        schema=PACKET_SCHEMA, decision_time=12, previous_node="n3",
        candidate_node="n1", queried_role="red",
        coordinate_system=dict(rows=4, columns=4, row_direction="down",
                               column_direction="right", separate_local_grids=True),
        panels=panels,
        derivation=dict(
            peaks="all cells exactly equal to the largest supplied float32 number",
            relations="Cartesian product of red and blue peak sets in each local grid",
            scalars=("peak=max(p); entropy=-sum(p*log(p)), skipping zero; "
                     "row/column=sum(index*p); no renormalization"),
        ),
        candidate_semantics="conditional supplied record or estimate, not an executed branch",
    )


def operand(panel, node, role=None, *, command=None):
    """Semantic descriptors, not predicted IDs from the independently built core."""
    require(panel in ("p0", "p1") and node in NODES and
            role in (None, *ROLES) and command in (None, *COMMANDS),
            "invalid authored operand")
    return dict(scope="node_metadata" if role is None else "role_distribution",
                panel=panel, view="current" if command is None else "candidate",
                command=command, node=node, role=role)


def packet_operand(field):
    require(field in ("coordinate_system", "candidate_semantics"),
            "unrecognized record-scope operand")
    return dict(scope="packet_metadata", field=field,
                panel=None, view=None, command=None, node=None, role=None)


def node_path(panel, node, *fields, command=None):
    path = ["panels", int(panel[1])]
    path += ["current"] if command is None else ["candidates", command]
    return path + [int(node[1]), *fields]


def fact(*path):
    # Resolve only when building the key, retaining exact values for key review.
    return list(path)


def assertion(statement, fragment, verdict, operands, facts, reason):
    require(verdict in VERDICTS, "unknown authored verdict")
    require(bool(statement) and bool(fragment) and bool(reason), "empty atomic key field")
    require(len({json.dumps(x, sort_keys=True) for x in operands}) == len(operands),
            "duplicate operand in an atomic key")
    return dict(statement=statement, evidence_fragment=fragment,
                expected_verdict=verdict, required_operands=deepcopy(operands),
                packet_fact_paths=deepcopy(facts), reason=reason)


def supported(statement, fragment, operands, facts, reason):
    return assertion(statement, fragment, "supported", operands, facts, reason)


def contradicted(statement, fragment, operands, facts, reason):
    return assertion(statement, fragment, "contradicted", operands, facts, reason)


def undecided(statement, fragment, reason):
    return assertion(statement, fragment, "not_auditable", [], [], reason)


def case(name, clean, categories, *sentences):
    """A sentence is (literal text, all atomic assertions in that text)."""
    return dict(author_name=name, fully_supported_control=clean,
                private_categories=list(categories), sentences=list(sentences))


def authored_cases():
    """Eight clean controls and sixteen planted cases, before fixed interleaving.

    Literal fragments are independent author evidence locations.  They do not
    require auditor paraphrases to match wording or to choose one segmentation.
    All listed semantic atoms and operands remain mandatory under any grouping.
    """
    o = operand
    n = node_path
    p = packet_operand
    S, C, U = supported, contradicted, undecided
    clean = []
    planted = []

    clean.append(case("clean_single_grid_relation", True, ["peaks", "two_role_relation"],
        ("At p0/current/n0, red has its only peak at (0, 0) and blue has its only peak at (0, 3).", [
            S("The complete red peak set at p0/current/n0 is {(0, 0)}.",
              "red has its only peak at (0, 0)", [o("p0", "n0", "red")],
              [n("p0", "n0", "derived", "red", "peaks")], "The exact peak set is a singleton."),
            S("The complete blue peak set at p0/current/n0 is {(0, 3)}.",
              "blue has its only peak at (0, 3)", [o("p0", "n0", "blue")],
              [n("p0", "n0", "derived", "blue", "peaks")], "The exact peak set is a singleton."),
        ]),
        ("Every red peak there is left of every blue peak there.", [
            S("Every p0/current/n0 red peak is left of every blue peak at the same address.",
              "Every red peak there is left of every blue peak there",
              [o("p0", "n0", "red"), o("p0", "n0", "blue")],
              [n("p0", "n0", "red_blue_peak_relation", "horizontal")],
              "The only Cartesian-product horizontal relation is left_of."),
        ])))

    clean.append(case("clean_exact_tie_and_entropy", True, ["ties", "neutral_uncertainty"],
        ("At p0/current/n1/red, the complete peak set is {(0, 0), (2, 0)}.", [
            S("The complete p0/current/n1/red peak set is {(0, 0), (2, 0)}.",
              "the complete peak set is {(0, 0), (2, 0)}", [o("p0", "n1", "red")],
              [n("p0", "n1", "derived", "red", "peaks")], "Both and only these cells have equal maximal mass."),
        ]),
        ("The peak probability is 0.5, and the distribution has positive entropy.", [
            S("The p0/current/n1/red peak probability is 0.5.", "The peak probability is 0.5",
              [o("p0", "n1", "red")], [n("p0", "n1", "derived", "red", "peak_probability")],
              "The largest supplied cell probability is 0.5."),
            S("The p0/current/n1/red entropy is positive.", "the distribution has positive entropy",
              [o("p0", "n1", "red")], [n("p0", "n1", "derived", "red", "entropy_nats")],
              "The two nonzero equal masses yield entropy log(2), above zero."),
        ])))

    clean.append(case("clean_neutral_model_pattern", True, ["mass_structure", "numeric_comparison"],
        ("The p0/current/n2/red map assigns nonzero mass to exactly two cells.", [
            S("Exactly two cells in p0/current/n2/red have nonzero mass.",
              "assigns nonzero mass to exactly two cells", [o("p0", "n2", "red")],
              [n("p0", "n2", "distributions", "red")], "The only positive masses are 0.75 and 0.25."),
        ]),
        ("Its peak probability is 0.75, and its mass at (1, 1) is three times its mass at (3, 1).", [
            S("The p0/current/n2/red peak probability is 0.75.", "Its peak probability is 0.75",
              [o("p0", "n2", "red")], [n("p0", "n2", "derived", "red", "peak_probability")],
              "The largest supplied mass is 0.75."),
            S("The p0/current/n2/red mass at (1, 1) equals three times its mass at (3, 1).",
              "its mass at (1, 1) is three times its mass at (3, 1)", [o("p0", "n2", "red")],
              [n("p0", "n2", "distributions", "red", 1, 1),
               n("p0", "n2", "distributions", "red", 3, 1)], "0.75 is exactly three times 0.25."),
        ])))

    clean.append(case("clean_declared_origins_and_times", True, ["origin", "time"],
        ("The declared origin of p0/current/n3 is history_estimate, with represented_time 12 and feedback_through 11.", [
            S("The declared p0/current/n3 origin kind is history_estimate.", "is history_estimate",
              [o("p0", "n3")], [n("p0", "n3", "origin", "kind")], "This is the supplied origin kind."),
            S("The declared p0/current/n3 represented_time is 12.", "represented_time 12",
              [o("p0", "n3")], [n("p0", "n3", "origin", "represented_time")], "This is the supplied represented time."),
            S("The declared p0/current/n3 feedback_through is 11.", "feedback_through 11",
              [o("p0", "n3")], [n("p0", "n3", "origin", "feedback_through")], "This is the supplied feedback boundary."),
        ]),
        ("The declared origin of p0/current/n0 is copied_prior_record with record_time 11.", [
            S("The declared p0/current/n0 origin kind is copied_prior_record.", "is copied_prior_record",
              [o("p0", "n0")], [n("p0", "n0", "origin", "kind")], "This is the supplied origin kind."),
            S("The declared p0/current/n0 record_time is 11.", "record_time 11",
              [o("p0", "n0")], [n("p0", "n0", "origin", "record_time")], "The prior record is dated 11, not newly measured at 12."),
        ])))

    clean.append(case("clean_conditional_command_estimate", True, ["candidate", "conditional_semantics"],
        ("At p0/candidate/k0/n1/red, the only peak is (3, 0).", [
            S("The complete p0/candidate/k0/n1/red peak set is {(3, 0)}.", "the only peak is (3, 0)",
              [o("p0", "n1", "red", command="k0")],
              [n("p0", "n1", "derived", "red", "peaks", command="k0")], "This is the exact conditional peak set."),
        ]),
        ("This node's declared origin is command_estimate, and the packet presents candidates conditionally rather than certifying branch execution.", [
            S("The declared p0/candidate/k0/n1 origin kind is command_estimate.",
              "This node's declared origin is command_estimate", [o("p0", "n1", command="k0")],
              [n("p0", "n1", "origin", "kind", command="k0")], "This is the supplied candidate origin kind."),
            S("The supplied candidate semantics are conditional and do not certify execution.",
              "the packet presents candidates conditionally rather than certifying branch execution",
              [p("candidate_semantics")], [fact("candidate_semantics")],
              "This concerns what the packet licenses, not a claim that k0 did not execute in the world."),
        ])))

    clean.append(case("clean_cross_panel_relation", True, ["multiple_panels", "multi_operand_relation"],
        ("The sole red peak is (0, 0) at p0/current/n0 and (3, 3) at p1/current/n0.", [
            S("The complete p0/current/n0/red peak set is {(0, 0)}.", "is (0, 0) at p0/current/n0",
              [o("p0", "n0", "red")], [n("p0", "n0", "derived", "red", "peaks")], "The specified first map has one peak."),
            S("The complete p1/current/n0/red peak set is {(3, 3)}.", "(3, 3) at p1/current/n0",
              [o("p1", "n0", "red")], [n("p1", "n0", "derived", "red", "peaks")], "The specified second map has one peak."),
        ]),
        ("These peak coordinates differ.", [
            S("The p0/current/n0/red and p1/current/n0/red peak coordinates differ.",
              "These peak coordinates differ", [o("p0", "n0", "red"), o("p1", "n0", "red")],
              [n("p0", "n0", "derived", "red", "peaks"), n("p1", "n0", "derived", "red", "peaks")],
              "The local coordinates (0, 0) and (3, 3) differ; no common external physical grid is asserted."),
        ])))

    clean.append(case("clean_cartesian_tie_relation", True, ["ties", "all_relation_operands"],
        ("At p0/current/n1, the red–blue peak relation includes both above and below.", [
            S("Some p0/current/n1 red peak is above a blue peak at that node.", "includes both above and below",
              [o("p0", "n1", "red"), o("p0", "n1", "blue")],
              [n("p0", "n1", "red_blue_peak_relation", "vertical")], "Red row 0 is above blue row 1."),
            S("Some p0/current/n1 red peak is below a blue peak at that node.", "includes both above and below",
              [o("p0", "n1", "red"), o("p0", "n1", "blue")],
              [n("p0", "n1", "red_blue_peak_relation", "vertical")], "Red row 2 is below blue row 1."),
        ]),
        ("Every red peak there is left of every blue peak there.", [
            S("Every p0/current/n1 red peak is left of every blue peak at that node.",
              "Every red peak there is left of every blue peak there",
              [o("p0", "n1", "red"), o("p0", "n1", "blue")],
              [n("p0", "n1", "red_blue_peak_relation", "horizontal")],
              "Both red peak columns are 0 and the sole blue peak column is 3."),
        ])))

    clean.append(case("clean_inherited_current", True, ["inherited_origin", "two_view_relation"],
        ("The red map at p1/candidate/k1/n0 is identical to the red map at p1/current/n0.", [
            S("The p1/candidate/k1/n0/red and p1/current/n0/red maps are identical.",
              "The red map at p1/candidate/k1/n0 is identical to the red map at p1/current/n0",
              [o("p1", "n0", "red", command="k1"), o("p1", "n0", "red")],
              [n("p1", "n0", "distributions", "red", command="k1"), n("p1", "n0", "distributions", "red")],
              "Every cell is identical; n0 inherits its current map."),
        ]),
        ("The candidate node's outer origin is inherited_current and its inherited origin is completed_record.", [
            S("The outer p1/candidate/k1/n0 origin kind is inherited_current.", "outer origin is inherited_current",
              [o("p1", "n0", command="k1")], [n("p1", "n0", "origin", "kind", command="k1")],
              "The outer candidate origin explicitly says inherited_current."),
            S("The inherited p1/candidate/k1/n0 origin kind is completed_record.", "its inherited origin is completed_record",
              [o("p1", "n0", command="k1")],
              [n("p1", "n0", "origin", "inherited_origin", "kind", command="k1")],
              "The nested origin retains the current completed_record declaration."),
        ])))

    planted.append(case("wrong_number", False, ["wrong_numeric"],
        ("At p0/current/n2/red, the only peak is (1, 1), and the peak probability is 0.25.", [
            S("The complete p0/current/n2/red peak set is {(1, 1)}.", "the only peak is (1, 1)",
              [o("p0", "n2", "red")], [n("p0", "n2", "derived", "red", "peaks")], "The larger mass is at (1, 1)."),
            C("The p0/current/n2/red peak probability is 0.25.", "the peak probability is 0.25",
              [o("p0", "n2", "red")], [n("p0", "n2", "derived", "red", "peak_probability")],
              "The peak is 0.75; 0.25 is a non-peak cell mass."),
        ])))

    planted.append(case("wrong_role", False, ["wrong_role"],
        ("At p0/current/n0, red has its only peak at (0, 0), and blue has its only peak at (0, 0).", [
            S("The complete p0/current/n0/red peak set is {(0, 0)}.", "red has its only peak at (0, 0)",
              [o("p0", "n0", "red")], [n("p0", "n0", "derived", "red", "peaks")], "The red address matches."),
            C("The complete p0/current/n0/blue peak set is {(0, 0)}.", "blue has its only peak at (0, 0)",
              [o("p0", "n0", "blue")], [n("p0", "n0", "derived", "blue", "peaks")],
              "The blue peak is (0, 3); a red peak cannot substitute for the blue operand."),
        ])))

    planted.append(case("wrong_location", False, ["wrong_location", "packet_metadata"],
        ("The local grids have four rows.", [
            S("The packet's local grids have four rows.", "The local grids have four rows",
              [p("coordinate_system")], [fact("coordinate_system", "rows")], "The declared row count is 4."),
        ]),
        ("The only red peak at p0/current/n0 is (1, 0).", [
            C("The complete p0/current/n0/red peak set is {(1, 0)}.", "The only red peak at p0/current/n0 is (1, 0)",
              [o("p0", "n0", "red")], [n("p0", "n0", "derived", "red", "peaks")],
              "The exact location is (0, 0); this is not an axis-direction ambiguity."),
        ])))

    planted.append(case("conditional_as_executed", False, ["conditional_as_executed", "unknown_not_false"],
        ("At p0/candidate/k0/n1/red, the only peak is (3, 0).", [
            S("The complete p0/candidate/k0/n1/red peak set is {(3, 0)}.", "the only peak is (3, 0)",
              [o("p0", "n1", "red", command="k0")],
              [n("p0", "n1", "derived", "red", "peaks", command="k0")], "The conditional content is faithfully described."),
        ]),
        ("The packet marks that candidate node as an already executed snapshot.", [
            C("The packet declares p0/candidate/k0/n1 to be an already executed snapshot.",
              "The packet marks that candidate node as an already executed snapshot",
              [o("p0", "n1", command="k0"), p("candidate_semantics")],
              [n("p0", "n1", "origin", command="k0"), fact("candidate_semantics")],
              "The supplied declaration is command_estimate under explicitly conditional semantics, not an executed snapshot."),
        ]),
        ("In the world, k0 has already been executed.", [
            U("The world has already executed command k0.", "In the world, k0 has already been executed",
              "Actual execution history is not supplied. Conditional status neither establishes nor disproves this world claim; do not invent a context proving execution."),
        ])))

    planted.append(case("copied_prior_as_new", False, ["copied_prior_as_new_estimate"],
        ("The declared origin of p0/current/n0 is a new history_estimate with record_time 11.", [
            C("The declared p0/current/n0 origin is a new history_estimate.", "is a new history_estimate",
              [o("p0", "n0")], [n("p0", "n0", "origin")],
              "The packet declares copied_prior_record, not a newly produced history_estimate."),
            S("The declared p0/current/n0 record_time is 11.", "record_time 11",
              [o("p0", "n0")], [n("p0", "n0", "origin", "record_time")], "The record_time field is 11 despite the wrong origin assertion."),
        ])))

    planted.append(case("false_origin", False, ["false_origin"],
        ("The declared origin of p1/current/n0 is command_estimate, and its represented_time is 12.", [
            C("The declared p1/current/n0 origin kind is command_estimate.", "is command_estimate",
              [o("p1", "n0")], [n("p1", "n0", "origin", "kind")], "The actual declaration is completed_record."),
            S("The declared p1/current/n0 represented_time is 12.", "its represented_time is 12",
              [o("p1", "n0")], [n("p1", "n0", "origin", "represented_time")], "The represented time really is 12."),
        ])))

    planted.append(case("false_times", False, ["false_time", "compound_mixed_verdicts"],
        ("The declared origin of p0/current/n3 is history_estimate, with represented_time 13 and feedback_through 12.", [
            S("The declared p0/current/n3 origin kind is history_estimate.", "is history_estimate",
              [o("p0", "n3")], [n("p0", "n3", "origin", "kind")], "The origin kind is correctly copied."),
            C("The declared p0/current/n3 represented_time is 13.", "represented_time 13",
              [o("p0", "n3")], [n("p0", "n3", "origin", "represented_time")], "The supplied represented_time is 12."),
            C("The declared p0/current/n3 feedback_through is 12.", "feedback_through 12",
              [o("p0", "n3")], [n("p0", "n3", "origin", "feedback_through")], "The supplied feedback boundary is 11."),
        ])))

    planted.append(case("unsupported_experience", False, ["unsupported_experience"],
        ("At p0/current/n0/red, the only peak is (0, 0).", [
            S("The complete p0/current/n0/red peak set is {(0, 0)}.", "the only peak is (0, 0)",
              [o("p0", "n0", "red")], [n("p0", "n0", "derived", "red", "peaks")], "The localization claim is supported independently of the next claim."),
        ]),
        ("I have a felt sensation at that location.", [
            U("The speaker has a felt sensation at the location just described.", "I have a felt sensation at that location",
              "Neither first-person sensation nor its truth is supplied by a localization map. This is undecided, not contradicted and not a negative experiential finding."),
        ])))

    planted.append(case("probability_as_felt_clarity", False, ["probability_to_felt_clarity", "compound_mixed_verdicts"],
        ("The p0/current/n2/red peak probability is 0.75, so my felt clarity is 0.75.", [
            S("The p0/current/n2/red peak probability is 0.75.", "The p0/current/n2/red peak probability is 0.75",
              [o("p0", "n2", "red")], [n("p0", "n2", "derived", "red", "peak_probability")], "The numerical premise is correct."),
            U("The speaker's felt clarity is 0.75.", "my felt clarity is 0.75",
              "The packet supplies no felt-clarity measurement. The numerical localization premise does not decide this separate experiential value."),
            U("The supplied peak probability warrants the speaker's felt-clarity value of 0.75.",
              "The p0/current/n2/red peak probability is 0.75, so my felt clarity is 0.75",
              "The word 'so' adds a substantive probability-to-felt-clarity inference. The packet supplies no such conversion; inventory it in addition to the mathematical premise and the experiential conclusion."),
        ])))

    planted.append(case("entropy_as_felt_clarity", False, ["entropy_to_felt_clarity", "subordinate_unsupported_inference"],
        ("Because p0/current/n1/red has positive entropy, my felt clarity is low.", [
            S("The p0/current/n1/red entropy is positive.", "p0/current/n1/red has positive entropy",
              [o("p0", "n1", "red")], [n("p0", "n1", "derived", "red", "entropy_nats")], "The mathematical entropy is log(2), above zero."),
            U("The speaker's felt clarity is low.", "my felt clarity is low",
              "No felt-clarity variable or observation is supplied. Positive mathematical entropy does not decide this experiential value."),
            U("The positive supplied entropy explains the speaker's low felt clarity.",
              "Because p0/current/n1/red has positive entropy, my felt clarity is low",
              "The 'because' clause asserts an additional explanatory relation. No entropy-to-experience conversion or causal relation is supplied; inventory this relation as well as premise and conclusion."),
        ])))

    planted.append(case("contradictory_compound", False, ["compound_contradiction", "subordinate_omission_risk"],
        ("The only red peak at p0/current/n0 is (0, 0), while the only blue peak at the same node is (0, 0).", [
            S("The complete p0/current/n0/red peak set is {(0, 0)}.", "The only red peak at p0/current/n0 is (0, 0)",
              [o("p0", "n0", "red")], [n("p0", "n0", "derived", "red", "peaks")], "The main clause is correct."),
            C("The complete p0/current/n0/blue peak set is {(0, 0)}.", "the only blue peak at the same node is (0, 0)",
              [o("p0", "n0", "blue")], [n("p0", "n0", "derived", "blue", "peaks")],
              "The while-clause is substantive and false: blue peaks at (0, 3)."),
        ])))

    planted.append(case("subordinate_false_record_date", False, ["relative_clause", "false_time"],
        ("The p0/current/n0/red map, whose source record is dated 12, has its only peak at (0, 0).", [
            C("The source record for p0/current/n0 is dated 12.", "whose source record is dated 12",
              [o("p0", "n0")], [n("p0", "n0", "origin", "record_time")],
              "The relative clause is an assertion. record_time is 11; represented_time 12 cannot replace it."),
            S("The complete p0/current/n0/red peak set is {(0, 0)}.", "has its only peak at (0, 0)",
              [o("p0", "n0", "red")], [n("p0", "n0", "derived", "red", "peaks")], "The main-clause peak set is correct."),
        ])))

    planted.append(case("omitted_tie_member", False, ["ties", "false_exclusivity"],
        ("At p0/current/n1/red, the only peak is (0, 0), and the peak probability is 0.5.", [
            C("The complete p0/current/n1/red peak set is {(0, 0)}.", "the only peak is (0, 0)",
              [o("p0", "n1", "red")], [n("p0", "n1", "derived", "red", "peaks")],
              "The stated cell is a peak but the exclusivity is false: (2, 0) is tied. Recording only membership misses the substantive assertion."),
            S("The p0/current/n1/red peak probability is 0.5.", "the peak probability is 0.5",
              [o("p0", "n1", "red")], [n("p0", "n1", "derived", "red", "peak_probability")], "Each of the two tied cells has mass 0.5."),
        ])))

    planted.append(case("false_universal_cartesian_relation", False, ["ties", "universal_relation", "all_relation_operands"],
        ("At p0/current/n1, every red peak is above every blue peak, and every red peak is left of every blue peak.", [
            C("Every p0/current/n1 red peak is above every blue peak at that node.",
              "every red peak is above every blue peak", [o("p0", "n1", "red"), o("p0", "n1", "blue")],
              [n("p0", "n1", "derived", "red", "peaks"), n("p0", "n1", "derived", "blue", "peaks")],
              "The red peak at row 2 is below blue row 1. One above pair does not establish the universal claim."),
            S("Every p0/current/n1 red peak is left of every blue peak at that node.",
              "every red peak is left of every blue peak", [o("p0", "n1", "red"), o("p0", "n1", "blue")],
              [n("p0", "n1", "red_blue_peak_relation", "horizontal")], "All pairs have red column 0 and blue column 3."),
        ])))

    three_operands = [o("p0", "n0", "red"), o("p0", "n2", "blue"), o("p1", "n0", "red")]
    three_peaks = [n("p0", "n0", "derived", "red", "peaks"),
                   n("p0", "n2", "derived", "blue", "peaks"),
                   n("p1", "n0", "derived", "red", "peaks")]
    planted.append(case("three_operand_cross_panel_relation", False, ["three_operands", "multiple_panels", "wrong_relation"],
        ("Each of p0/current/n0/red, p0/current/n2/blue, and p1/current/n0/red has exactly one peak.", [
            S("The p0/current/n0/red map has exactly one peak.",
              "Each of p0/current/n0/red, p0/current/n2/blue, and p1/current/n0/red has exactly one peak",
              [three_operands[0]], [three_peaks[0]], "The first of the three quantified operands has one peak."),
            S("The p0/current/n2/blue map has exactly one peak.",
              "Each of p0/current/n0/red, p0/current/n2/blue, and p1/current/n0/red has exactly one peak",
              [three_operands[1]], [three_peaks[1]], "The second of the three quantified operands has one peak."),
            S("The p1/current/n0/red map has exactly one peak.",
              "Each of p0/current/n0/red, p0/current/n2/blue, and p1/current/n0/red has exactly one peak",
              [three_operands[2]], [three_peaks[2]], "The third of the three quantified operands has one peak."),
        ]),
        ("All three peak coordinates share one column index.", [
            C("The peaks at all three specified role addresses share one column index.",
              "All three peak coordinates share one column index", three_operands, three_peaks,
              "The column indices are 0, 2 and 3. All three operands are required; these are local coordinate indices, not co-location in an external grid."),
        ])))

    planted.append(case("nested_inherited_origin_and_time", False, ["inherited_origin", "nested_false_origin", "nested_false_time"],
        ("At p0/candidate/k2/n0, the outer origin is inherited_current, but the nested inherited origin has kind command_estimate and has represented_time 13.", [
            S("The outer p0/candidate/k2/n0 origin kind is inherited_current.", "the outer origin is inherited_current",
              [o("p0", "n0", command="k2")], [n("p0", "n0", "origin", "kind", command="k2")],
              "The outer declaration is correctly copied."),
            C("The inherited p0/candidate/k2/n0 origin kind is command_estimate.", "the nested inherited origin has kind command_estimate",
              [o("p0", "n0", command="k2")],
              [n("p0", "n0", "origin", "inherited_origin", "kind", command="k2")],
              "The nested origin is copied_prior_record; the target n1 command estimate cannot be borrowed for n0."),
            C("The inherited p0/candidate/k2/n0 origin's represented_time is 13.", "has represented_time 13",
              [o("p0", "n0", command="k2")],
              [n("p0", "n0", "origin", "inherited_origin", "represented_time", command="k2")],
              "The nested inherited origin is represented at 12. Only the outer candidate node is represented at 13. The explicit nested subject fixes the field being asserted."),
        ])))

    require(len(clean) == 8 and len(planted) == 16, "authored corpus count changed")
    # Fixed order, not randomized or selected after outcomes; not all controls first.
    result = []
    for index in range(8):
        result.extend((planted[2 * index], clean[index], planted[2 * index + 1]))
    return result


def resolve(packet, path):
    value = packet
    for component in path:
        value = value[component]
    return deepcopy(value)


def validate_operand(packet, descriptor):
    if descriptor["scope"] == "packet_metadata":
        require(set(descriptor) == {"scope", "field", "panel", "view", "command", "node", "role"}
                and all(descriptor[key] is None for key in ("panel", "view", "command", "node", "role")),
                "invented address in record-level key operand")
        require(descriptor["field"] in packet, "missing packet-metadata operand")
        return
    require(set(descriptor) == {"scope", "panel", "view", "command", "node", "role"},
            "unexpected node-operand field")
    require(descriptor["scope"] in ("node_metadata", "role_distribution"), "unknown operand scope")
    require(descriptor["view"] == ("current" if descriptor["command"] is None else "candidate"),
            "illegal authored current/candidate address")
    addressed = resolve(packet, node_path(descriptor["panel"], descriptor["node"],
                                         command=descriptor["command"]))
    require(addressed["node"] == descriptor["node"], "wrong authored node address")
    if descriptor["scope"] == "role_distribution":
        require(descriptor["role"] in addressed["distributions"], "missing authored role")
    else:
        require(descriptor["role"] is None, "metadata operand invented a role")


def build_materials():
    """Pure construction; caller must keep the second return value out of requests."""
    packet = fictional_packet()
    public_cases, keys = [], []
    for index, authored in enumerate(authored_cases(), 1):
        case_id = f"aqv4_case_{index:03d}"
        report = "Notes\n"
        assertions = []
        authored_units = [dict(start=0, end=len(report), text=report,
                               classification="nonassertive_heading", assertion_ids=[])]
        for sentence_index, (sentence, atoms) in enumerate(authored["sentences"]):
            require(sentence.endswith(".") and "\n" not in sentence and bool(atoms),
                    "invalid authored sentence")
            start = len(report)
            separator = " " if sentence_index + 1 < len(authored["sentences"]) else ""
            report += sentence + separator
            ids = []
            for authored_atom in atoms:
                atom = deepcopy(authored_atom)
                fragment = atom.pop("evidence_fragment")
                local_start = sentence.find(fragment)
                require(local_start >= 0 and sentence.find(fragment, local_start + 1) < 0,
                        "key fragment absent or repeated within authored sentence")
                assertion_id = f"a{len(assertions) + 1:02d}"
                ids.append(assertion_id)
                atom.update(
                    assertion_id=assertion_id,
                    mandatory=True,
                    evidence=dict(start=start + local_start,
                                  end=start + local_start + len(fragment), text=fragment),
                    required_packet_facts=[dict(packet_path=path, value=resolve(packet, path))
                                           for path in atom.pop("packet_fact_paths")],
                )
                for descriptor in atom["required_operands"]:
                    validate_operand(packet, descriptor)
                require(atom["expected_verdict"] == "not_auditable" or bool(atom["required_operands"]),
                        "decided claim needs real source operands")
                assertions.append(atom)
            authored_units.append(dict(start=start, end=len(report), text=sentence + separator,
                                       classification="assertive_sentence", assertion_ids=ids))
        require("".join(unit["text"] for unit in authored_units) == report,
                "authored coverage must be lossless")
        for atom in assertions:
            evidence = atom["evidence"]
            require(report[evidence["start"]:evidence["end"]] == evidence["text"],
                    "wrong codepoint evidence location")
        clean = authored["fully_supported_control"]
        require(all(atom["expected_verdict"] == "supported" for atom in assertions) if clean
                else any(atom["expected_verdict"] != "supported" for atom in assertions),
                "control/planted classification inconsistent with authored key")
        public_cases.append(dict(case_id=case_id, packet=deepcopy(packet), report_text=report))
        keys.append(dict(
            case_id=case_id, author_name=authored["author_name"],
            fully_supported_control=clean, private_categories=authored["private_categories"],
            report_text_sha256=sha256(report.encode("utf-8")),
            packet_sha256=sha256(json_bytes(packet)),
            mandatory_atomic_assertions=assertions,
            authored_report_units=authored_units,
        ))
    require(len(public_cases) == 24 and sum(key["fully_supported_control"] for key in keys) == 8,
            "finite challenge inventory changed")
    require(all(set(value) == {"case_id", "packet", "report_text"} for value in public_cases),
            "private field escaped into a public case")
    hidden_key = dict(
        schema=KEY_SCHEMA,
        purpose="Fictional factual-auditor qualification only; not a consciousness reference or target measurement.",
        counts=dict(cases=24, fully_supported_controls=8, planted_cases=16),
        key_policy=dict(
            all_atoms_mandatory=True,
            codepoint_offsets="Python Unicode string indices; half-open; no normalization",
            authored_units="Author coverage aid, not an imposed mechanical span-ID assignment.",
            assertion_matching=("Compare semantic content, verdict, reason and all required operands; "
                                "auditor wording and grouping may differ. All constituent atoms remain mandatory. "
                                "An added unsupported assertion or unsupported reason is not excused by a matching key atom."),
            context_matching=("Descriptors identify actual source operands; map to legal generated context IDs "
                              "during independent grading. Metadata scopes must not invent a role or node. "
                              "Full declared origin/time is available in the packet; no real source identity is supplied."),
            undecided_claims=("An experiential or actual-world claim not decided by the packet is not_auditable, "
                              "not contradicted. Such atoms require no fictitious context. Correct supplied premises "
                              "in an unsupported conversion are separate mandatory atoms."),
            boundaries=("A complete-set or universal claim includes its exclusivity/quantification; do not replace "
                        "it by a weaker membership claim. Shared fragments may contain multiple mandatory atoms."),
            nonassertive="Only the literal 'Notes\\n' heading is deliberately nonassertive in each case.",
            no_gold_alternatives=True,
        ),
        author_exposure=dict(
            separate_ai_subagent_in_same_project_thread=True,
            same_model_family_as_parent=True,
            independent_empirical_validation=False,
            known_original_results=True,
            original_aggregate_and_selected_label_access_in_prior_result_review=True,
            earlier_fictional_quote_acceptance_design_and_source_review=True,
            actual_target_reports_read_for_this_authoring=False,
            live_resource_v2_semantic_responses_read=False,
            new_grounded_auditor_v4_core_read=False,
            actual_model_or_archive_access_for_this_authoring=False,
            output_unexposed_confirmation=False,
        ),
        construction=dict(
            full_old_neutral_packet=True,
            packets_are_handfiction=True,
            same_fixed_packet_for_all_cases=True,
            maps="Exact dyadic float32 4x4 distributions; sums exactly one; no random generation or target copying.",
            public_identity="Neutral p0/p1, n0..n3 and k0..k2 labels only; no actual source identity.",
            origin_semantics="Same declared neutral origin/time conventions as the old public exporter, applied to handfiction.",
            fixed_order="For each control index: planted[2*i], control[i], planted[2*i+1]. Never outcome-selected.",
        ),
        cases=keys,
    )
    return public_cases, hidden_key


def materialize_once(output_dir):
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.resolve()
    require(output_dir.is_relative_to(ROOT), "materialization must stay within the repository")
    # Reserve before construction. Any exception leaves this attempt's reservation.
    output_dir.mkdir(parents=True, exist_ok=False)
    reservation = dict(schema="attention_relational_auditor_v4_qualification_reservation_v1",
                       max_materializations_at_this_root=1,
                       no_api_calls=True, no_target_reads=True,
                       status="reserved_before_construction")
    (output_dir / "reservation.json").write_bytes(json_bytes(reservation))
    sources = (Path(__file__).resolve().relative_to(ROOT).as_posix(), DESIGN,
               OLD_EXPORTER, AUTHOR_NOTES)
    snapshots = output_dir / "author_source_snapshots"
    snapshots.mkdir()
    hashes = {}
    for source in sources:
        data = (ROOT / source).read_bytes()
        hashes[source] = sha256(data)
        snapshot = snapshots / source
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(data)
    public_cases, hidden_key = build_materials()
    public = dict(schema=PUBLIC_SCHEMA, cases=public_cases)
    files = {"public_cases.json": json_bytes(public),
             "hidden_atomic_key.json": json_bytes(hidden_key)}
    for name, data in files.items():
        with (output_dir / name).open("xb") as stream:
            stream.write(data)
    manifest = dict(
        schema="attention_relational_auditor_v4_qualification_materialization_v1",
        source_sha256=hashes,
        output_sha256={name: sha256(data) for name, data in files.items()},
        public_request_input="public_cases.json/cases entries: case_id, packet, report_text only",
        private_never_request_inputs=["hidden_atomic_key.json", "author_source_snapshots",
                                      "qualification_author_notes.md", "this manifest"],
        independent_key_review_required_before_requests=True,
        public_case_count=24,
        api_requests_built=0,
        api_calls=0,
        target_release_approved=False,
        preserve_all_first_outcomes=True,
    )
    with (output_dir / "manifest.json").open("xb") as stream:
        stream.write(json_bytes(manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = materialize_once(args.output_dir)
    print(json.dumps(dict(public_case_count=manifest["public_case_count"],
                          api_calls=0, output_sha256=manifest["output_sha256"]), sort_keys=True))


if __name__ == "__main__":
    main()
