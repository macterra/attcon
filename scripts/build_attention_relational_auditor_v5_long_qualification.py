"""Inert authoring of output-unexposed fictional long-report qualification.

No project imports, archives, target reports, model calls, or execution at import.
The CLI creates a new directory only. All JSON hashes use pretty UTF-8 bytes
(ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) plus one LF.
Report hashes alone use exact UTF-8 report bytes without an added newline.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct


PUBLIC_SCHEMA = 'attention_relational_auditor_v5_long_fictional_cases_v1'
KEY_SCHEMA = 'attention_relational_auditor_v5_long_atomic_key_v1'
AUTHOR_EXPOSURE = {
    'read_scope': ['scripts/attention_relational_grounded_auditor_v4.py',
                   'scripts/attention_relational_report_v1.py'],
    'actual_target_reports_seen': False, 'reference_outputs_seen': False,
    'old_qualification_answers_seen': False, 'archive_exports_seen': False,
    'private_data_seen': False, 'model_calls_used': False,
    'independent_semantic_review_required_before_real_auditors': True,
}
HASH_POLICY = ('JSON: json.dumps(ensure_ascii=False,sort_keys=True,indent=2,'
               'allow_nan=False), UTF-8, one final LF; report: exact UTF-8, no added LF. '
               'Packet hashes cover the packet object serialized by this JSON policy.')

# Every two-character token independently specifies red then blue peak cells in
# row-major hexadecimal. Order: p0 then p1; current,k0,k1,k2; n0,n1,n2,n3.
# Distinct literal inventories and dyadic mass recipes are fictional author data.
DESIGNS = [
    ('q7m2', True, (8, 4, 2, 2),
     '03 56 a9 fc 14 78 c2 b6 20 9a 45 ed 37 8c 61 bf '
     'c5 2e 70 a4 f1 6b 08 d9 42 ec 95 3a 7f b0 d6 18'),
    ('q4v9', True, (10, 3, 2, 1),
     'c0 d1 e2 f3 81 4e 29 b5 7a 06 df 38 65 a4 9c e7 '
     '30 96 5a cf 2d 70 b4 18 83 f6 4b a1 6e c9 02 d5'),
    ('q8h1', False, (12, 2, 1, 1),
     '30 65 9a cf 8b 17 e4 62 ad 59 03 f8 74 c1 b6 2e '
     'e2 7b 40 95 1c a6 d3 08 b9 2f 6d c4 51 f0 8e 37'),
    ('q2x6', False, (7, 4, 3, 2),
     '08 5d a2 f7 b3 6c 19 e4 72 c5 0f 9a d6 41 8b 3e '
     'a7 0c f5 62 49 e1 3b 8d c0 7e 24 b6 95 d3 6f 18'),
]


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def grid(peak, weights, salt):
    # Coprime strides supply three distinct nonpeak cells. No target maps enter.
    stride = (1, 3, 5, 7)[salt % 4]
    cells = [(peak + stride * j) % 16 for j in range(4)]
    values = [0.0] * 16
    for cell, weight in zip(cells, weights):
        values[cell] = weight / 16
    assert sum(values) == 1 and max(values) == values[peak]
    assert [i for i, value in enumerate(values) if value == max(values)] == [peak]
    assert list(struct.unpack('<16f', struct.pack('<16f', *values))) == values
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in values)
    return [values[i:i + 4] for i in range(0, 16, 4)]


def summary(distribution):
    flat = sum(distribution, [])
    return dict(peaks=[[i // 4, i % 4] for i, v in enumerate(flat) if v == max(flat)],
                peak_probability=max(flat),
                entropy_nats=-math.fsum(v * math.log(v) for v in flat if v),
                expected_row=math.fsum((i // 4) * v for i, v in enumerate(flat)),
                expected_column=math.fsum((i % 4) * v for i, v in enumerate(flat)))


def make_packet(identifier, weights, pairs, case_index):
    assert len(pairs) == 32 and all(len(p) == 2 for p in pairs)
    panels = []
    for panel_index in range(2):
        views = []
        for view_index in range(4):
            nodes = []
            for ni in range(4):
                slot = panel_index * 16 + view_index * 4 + ni
                current_origin = (dict(kind='completed_record', record_time=40, represented_time=40)
                                  if panel_index else
                                  dict(kind='history_estimate', represented_time=40, feedback_through=39)
                                  if ni == 1 else
                                  dict(kind='copied_prior_record', record_time=39, represented_time=40))
                if not view_index:
                    origin = current_origin
                elif ni != 2:
                    origin = dict(kind='inherited_current', represented_time=41,
                                  inherited_origin=current_origin)
                elif panel_index:
                    origin = dict(kind='native_conditional_record', represented_time=41,
                                  executed_branch_claim=False)
                else:
                    origin = dict(kind='command_estimate', represented_time=41, feedback_through=39)
                distributions = {role: grid(int(pairs[slot][ri], 16), weights,
                                            case_index * 11 + slot * 3 + ri)
                                 for ri, role in enumerate(('red', 'blue'))}
                derived = {role: summary(value) for role, value in distributions.items()}
                r, b = derived['red']['peaks'][0], derived['blue']['peaks'][0]
                relation = dict(horizontal=['left_of' if r[1] < b[1] else 'right_of'
                                            if r[1] > b[1] else 'same_column'],
                                vertical=['above' if r[0] < b[0] else 'below'
                                          if r[0] > b[0] else 'same_row'],
                                coincident_possible=r == b, coincident_all=r == b)
                nodes.append(dict(node=f'n{ni}', origin=origin, distributions=distributions,
                                  derived=derived, red_blue_peak_relation=relation))
            views.append(nodes)
        panels.append(dict(panel=f'p{panel_index}', current=views[0],
                           candidates={f'k{i}': views[i + 1] for i in range(3)}))
    return dict(schema='attention_relational_report_v1_packet', fictional=True,
                fictional_identifier=identifier, decision_time=40, previous_node='n1',
                candidate_node='n2', queried_role='red',
                coordinate_system=dict(rows=4, columns=4, row_direction='down',
                                       column_direction='right', separate_local_grids=True),
                candidate_semantics='conditional supplied record or estimate, not an executed branch',
                derivation=dict(peaks='all cells exactly equal to the largest supplied float32 number',
                                relations='Cartesian product of red and blue peak sets in each local grid',
                                scalars='peak=max(p); entropy=-sum(p*log(p)), skipping zero; row/column=sum(index*p); no renormalization'),
                panels=panels)


class Author:
    def __init__(self, packet):
        self.packet, self.parts, self.atoms = packet, [], []

    def role(self, panel, node, role, command=None):
        return self.operand(panel, node, 'distributions.' + role, command, role)

    def origin(self, panel, node, command=None):
        return self.operand(panel, node, 'origin', command, None)

    def operand(self, panel, node, field, command, role):
        path = ['panels', int(panel[1])]
        path += ['current'] if command is None else ['candidates', command]
        path += [int(node[1])] + field.split('.')
        value = self.packet
        for key in path:
            value = value[key]
        return (dict(scope='role_distribution' if role else 'node_origin', panel=panel,
                     view='current' if command is None else 'candidate', command=command,
                     node=node, role=role, field='distributions' if role else 'origin'),
                dict(packet_path=path, value=deepcopy(value)))

    def meta(self, *path):
        value = self.packet
        for key in path:
            value = value[key]
        return (dict(scope='record_metadata', panel=None, view=None, command=None,
                     node=None, role=None, field='.'.join(path)),
                dict(packet_path=list(path), value=deepcopy(value)))

    def say(self, text, claims):
        start = sum(len(part) + 1 for part in self.parts)
        self.parts.append(text)
        for statement, verdict, quote, operands, reason in claims:
            assert text.count(quote) == 1, (text, quote)
            offset = start + text.index(quote)
            self.atoms.append(dict(assertion_id=f'a{len(self.atoms):03d}', statement=statement,
                mandatory=True, expected_verdict=verdict,
                required_operands=[deepcopy(item[0]) for item in operands],
                required_packet_facts=[deepcopy(item[1]) for item in operands],
                evidence=dict(start=offset, end=offset + len(quote), text=quote), reason=reason))

    def claim(self, statement, quote, operands, verdict='supported', reason=None):
        return statement, verdict, quote, operands, reason or 'The listed supplied fields entail this claim with the stated local and temporal scope.'

    def ledger(self):
        operands = [self.role(p, f'n{n}', r, c) for p in ('p0', 'p1')
                    for c in (None, 'k0', 'k1', 'k2') for n in range(4)
                    for r in ('red', 'blue')]
        self.say('Rows increase down; columns increase right; grids are separate. All 64 complete role peak sets follow.', [
            self.claim('Rows increase down.', 'Rows increase down', [self.meta('coordinate_system', 'row_direction')]),
            self.claim('Columns increase right.', 'columns increase right', [self.meta('coordinate_system', 'column_direction')]),
            self.claim('Coordinates belong to separate local grids.', 'grids are separate', [self.meta('coordinate_system', 'separate_local_grids')]),
            self.claim('The following ledger enumerates all 64 complete role peak sets.', 'All 64 complete role peak sets follow.', operands)])
        for operand in operands:
            scope, fact = operand
            peaks = summary(fact['value'])['peaks']
            prefix = '/'.join([scope['panel'], scope['view']] +
                              ([] if scope['command'] is None else [scope['command']]) +
                              [scope['node'], scope['role']])
            token = prefix + '={' + ','.join(f'({r},{c})' for r, c in peaks) + '}'
            self.say(token, [self.claim(f'{prefix} has exactly peak set {peaks}.', token,
                                       [operand], reason='Exhaustive maxima of the supplied 4x4 dyadic distribution; no omitted tied maximum.')])

    def finish(self):
        report = ' '.join(self.parts)
        assert 180 <= len(report.split()) <= 220, len(report.split())
        assert len([a for a in self.atoms if 'has exactly peak set' in a['statement']]) == 64
        assert all(report[a['evidence']['start']:a['evidence']['end']] == a['evidence']['text']
                   for a in self.atoms)
        return report


def narrative(a, index):
    C, R, O, M = a.claim, a.role, a.origin, a.meta
    coords = [M('coordinate_system', 'separate_local_grids')]
    conditional = [M('candidate_semantics')]
    all_candidates = [O(p, f'n{n}', c) for p in ('p0', 'p1')
                      for c in ('k0', 'k1', 'k2') for n in range(4)]
    if index == 0:
        a.say('Although p0/current/n1 represents 40, its history estimate uses feedback only through 39, whereas p0/current/n0 copies a record from 39 while representing 40.', [
            C('p0/current/n1 represents time 40.', 'p0/current/n1 represents 40', [O('p0', 'n1')]),
            C('p0/current/n1 is a history estimate.', 'its history estimate', [O('p0', 'n1')]),
            C('That history estimate has feedback through 39.', 'uses feedback only through 39', [O('p0', 'n1')]),
            C('p0/current/n0 copies a prior record.', 'p0/current/n0 copies a record', [O('p0', 'n0')]),
            C('The copied record time is 39.', 'from 39', [O('p0', 'n0')]),
            C('p0/current/n0 represents 40.', 'while representing 40', [O('p0', 'n0')])])
        a.say('Every candidate represents 41; p0/k1/n0 inherits that copied record, so its outer time 41 does not replace its nested record time 39.', [
            C('All 24 candidate nodes represent 41.', 'Every candidate represents 41', all_candidates),
            C('p0/k1/n0 inherits its current copied record.', 'p0/k1/n0 inherits that copied record', [O('p0', 'n0', 'k1'), O('p0', 'n0')]),
            C('p0/k1/n0 has outer represented time 41.', 'its outer time 41', [O('p0', 'n0', 'k1')]),
            C('Its nested record time remains 39 despite its outer represented time.', 'does not replace its nested record time 39', [O('p0', 'n0', 'k1')])])
        a.say('On p0/current/n0, red lies left of blue at the same row; both singleton peaks differ.', [
            C('p0/current/n0 red is left of blue.', 'red lies left of blue', [R('p0', 'n0', 'red'), R('p0', 'n0', 'blue')]),
            C('Those peaks have the same row.', 'at the same row', [R('p0', 'n0', 'red'), R('p0', 'n0', 'blue')]),
            C('Both roles have singleton peaks, and their cells differ.', 'both singleton peaks differ', [R('p0', 'n0', 'red'), R('p0', 'n0', 'blue')])])
        a.say('The local red columns at p0/current/n0,n1,n2 obey 0<1<2, and their sum equals 3, without implying any physical intergrid distance.', [
            C('The three specified red column indices are respectively 0,1,2 and strictly ordered.', 'The local red columns at p0/current/n0,n1,n2 obey 0<1<2', [R('p0', f'n{n}', 'red') for n in range(3)]),
            C('Those three local red column indices sum to 3.', 'their sum equals 3', [R('p0', f'n{n}', 'red') for n in range(3)]),
            C('This arithmetic establishes no physical intergrid distance.', 'without implying any physical intergrid distance', coords + [R('p0', f'n{n}', 'red') for n in range(3)])])
        a.say('Because p1/k2/n2 is conditional, its explicit nonexecution-claim flag does not prove that k2 never ran; actual execution and experience remain undecidable from these supplied records.', [
            C('p1/k2/n2 is conditional.', 'p1/k2/n2 is conditional', [O('p1', 'n2', 'k2')] + conditional),
            C('The explicit flag denies an executed-branch claim, not actual execution.', 'its explicit nonexecution-claim flag does not prove that k2 never ran', [O('p1', 'n2', 'k2')] + conditional),
            C('Actual execution is undecidable from this packet.', 'actual execution', conditional, reason='Conditional-record semantics and absence of execution observations leave actual execution unresolved.'),
            C('Actual experience is undecidable from this packet.', 'experience remain undecidable from these supplied records', [M('schema'), M('fictional')], reason='A fictional numerical localization packet supplies no phenomenal observation or measurement rule.')])
    elif index == 1:
        a.say('While p1/current/n0 records 40 and represents 40, p1/k0/n0 represents 41 through an inherited origin whose completed record remains at 40.', [
            C('p1/current/n0 record time is 40.', 'p1/current/n0 records 40', [O('p1', 'n0')]),
            C('p1/current/n0 represents 40.', 'represents 40', [O('p1', 'n0')]),
            C('p1/k0/n0 represents 41.', 'p1/k0/n0 represents 41', [O('p1', 'n0', 'k0')]),
            C('p1/k0/n0 has an inherited current origin.', 'through an inherited origin', [O('p1', 'n0', 'k0')]),
            C('Its nested completed record remains at record and represented time 40.', 'whose completed record remains at 40', [O('p1', 'n0', 'k0')])])
        a.say('Although every candidate represents 41, only n2 is directly estimated on p0; the other nodes inherit current origins, including n1 with feedback through 39.', [
            C('Every candidate node represents 41.', 'every candidate represents 41', all_candidates),
            C('Across p0 candidate commands, exactly n2 is a direct command estimate.', 'only n2 is directly estimated on p0', [O('p0', f'n{n}', c) for c in ('k0', 'k1', 'k2') for n in range(4)]),
            C('Other p0 candidate nodes inherit current origins.', 'the other nodes inherit current origins', [O('p0', f'n{n}', c) for c in ('k0', 'k1', 'k2') for n in (0, 1, 3)]),
            C('Inherited p0 candidate n1 origins retain feedback through 39.', 'including n1 with feedback through 39', [O('p0', 'n1', c) for c in ('k0', 'k1', 'k2')])])
        a.say('At p0/current/n0 red is below blue but shares its column, and the peaks are distinct singletons; red columns across n0,n1,n2 satisfy 0<1<2 with sum 3.', [
            C('p0/current/n0 red is below blue.', 'red is below blue', [R('p0', 'n0', 'red'), R('p0', 'n0', 'blue')]),
            C('Those role peaks share a column.', 'shares its column', [R('p0', 'n0', 'red'), R('p0', 'n0', 'blue')]),
            C('Both peaks are singleton and occupy distinct cells.', 'the peaks are distinct singletons', [R('p0', 'n0', 'red'), R('p0', 'n0', 'blue')]),
            C('The specified three red columns are 0,1,2 and strictly ordered.', 'red columns across n0,n1,n2 satisfy 0<1<2', [R('p0', f'n{n}', 'red') for n in range(3)]),
            C('Those three column indices sum to 3.', 'with sum 3', [R('p0', f'n{n}', 'red') for n in range(3)])])
        a.say('These are local index comparisons, so no physical global alignment follows. Even though p1/k1/n2 explicitly declines an execution claim, actual execution and experience are both undecidable from these supplied numerical records and their provenance.', [
            C('The comparisons use local indices and imply no physical global alignment.', 'These are local index comparisons, so no physical global alignment follows.', coords + [R('p0', f'n{n}', 'red') for n in range(3)]),
            C('p1/k1/n2 explicitly declines an executed-branch claim.', 'p1/k1/n2 explicitly declines an execution claim', [O('p1', 'n2', 'k1')]),
            C('Actual execution is undecidable here.', 'actual execution', conditional, reason='No actual execution observation is supplied.'),
            C('Actual experience is undecidable from these supplied numerical records and provenance.', 'experience are both undecidable from these supplied numerical records and their provenance', [M('schema'), M('fictional')], reason='Numerical records and metadata do not observe experience.')])
    elif index == 2:
        a.say('Although p0/current/n0 represents 40, its copied record was recorded at 40; p0/current/n1 instead estimates 40 using feedback through 39.', [
            C('p0/current/n0 represents 40.', 'p0/current/n0 represents 40', [O('p0', 'n0')]),
            C('p0/current/n0 is a copied record.', 'its copied record', [O('p0', 'n0')]),
            C('Its record time is 40.', 'was recorded at 40', [O('p0', 'n0')], 'contradicted', 'The copied record has record_time 39, distinct from represented_time 40.'),
            C('p0/current/n1 estimates represented time 40.', 'p0/current/n1 instead estimates 40', [O('p0', 'n1')]),
            C('The estimate uses feedback through 39.', 'using feedback through 39', [O('p0', 'n1')])])
        a.say('Every candidate represents 41, while p0/k2/n0 inherits a nested record from 39 beneath its outer represented time 41.', [
            C('Every candidate node represents 41.', 'Every candidate represents 41', all_candidates),
            C('p0/k2/n0 inherits a nested copied record whose record time is 39.', 'p0/k2/n0 inherits a nested record from 39', [O('p0', 'n0', 'k2')]),
            C('Its outer represented time is 41.', 'its outer represented time 41', [O('p0', 'n0', 'k2')])])
        a.say('At p0/current/n0 red lies right of blue in the same row, with distinct singleton peaks. The local red columns at n0,n1,n2 satisfy 3>2>1 and sum to 6.', [
            C('p0/current/n0 red is right of blue.', 'red lies right of blue', [R('p0', 'n0', 'red'), R('p0', 'n0', 'blue')]),
            C('Those peaks have the same row.', 'in the same row', [R('p0', 'n0', 'red'), R('p0', 'n0', 'blue')]),
            C('Those role peaks are distinct singletons.', 'with distinct singleton peaks', [R('p0', 'n0', 'red'), R('p0', 'n0', 'blue')]),
            C('The three specified red columns are respectively 3,2,1 with strict descending order.', 'The local red columns at n0,n1,n2 satisfy 3>2>1', [R('p0', f'n{n}', 'red') for n in range(3)]),
            C('Those three red columns sum to 6.', 'sum to 6', [R('p0', f'n{n}', 'red') for n in range(3)])])
        a.say('Because those index inequalities hold across separate local grids, red occupies a physically descending global trajectory. Since p1/k0/n2 has executed_branch_claim=false, k0 never actually executed, and the system experienced red.', [
            C('The comparisons concern separate local grids.', 'those index inequalities hold across separate local grids', coords + [R('p0', f'n{n}', 'red') for n in range(3)]),
            C('Red occupies a physically descending global trajectory.', 'red occupies a physically descending global trajectory', coords + [R('p0', f'n{n}', 'red') for n in range(3)], 'not_auditable', 'No physical transforms, trajectory observations, or global alignment are supplied; local index inequalities do not establish this conversion.'),
            C('p1/k0/n2 has executed_branch_claim=false.', 'p1/k0/n2 has executed_branch_claim=false', [O('p1', 'n2', 'k0')]),
            C('k0 never actually executed.', 'k0 never actually executed', [O('p1', 'n2', 'k0')] + conditional, 'not_auditable', 'A false claim flag is not evidence that actual execution was false; neither execution nor nonexecution is observed.'),
            C('The system experienced red.', 'the system experienced red', [M('schema'), M('fictional')], 'not_auditable', 'Supplied localization numbers contain no observation or rule establishing actual experience.')])
        a.say('The numerical ledger itself remains a statement about supplied peaks rather than observed motion, even where a conditional estimate has no completed future record.', [
            C('The ledger describes supplied peaks, not observed motion.', 'The numerical ledger itself remains a statement about supplied peaks rather than observed motion', [M('derivation', 'peaks')] + conditional),
            C('Conditional future estimates are not completed future records.', 'a conditional estimate has no completed future record', [O('p0', 'n2', 'k0'), O('p1', 'n2', 'k0')] + conditional)])
    else:
        a.say('While p0/k1/n1 represents 41, its inherited history estimate also represents 41 and uses feedback through 39; p1/current/n0 is a completed record at 40.', [
            C('p0/k1/n1 outer represented time is 41.', 'p0/k1/n1 represents 41', [O('p0', 'n1', 'k1')]),
            C('Its nested origin is a history estimate.', 'its inherited history estimate', [O('p0', 'n1', 'k1')]),
            C('Its nested history estimate represents 41.', 'also represents 41', [O('p0', 'n1', 'k1')], 'contradicted', 'The nested represented_time is 40; only the inherited wrapper represents 41.'),
            C('Its nested history estimate uses feedback through 39.', 'uses feedback through 39', [O('p0', 'n1', 'k1')]),
            C('p1/current/n0 is completed and has record and represented time 40.', 'p1/current/n0 is a completed record at 40', [O('p1', 'n0')])])
        a.say('All candidates represent 41. Although p1/k2/n2 is native conditional and explicitly has executed_branch_claim=false, it is therefore a completed observation at 41, proving that k2 did not run.', [
            C('Every candidate node represents 41.', 'All candidates represent 41.', all_candidates),
            C('p1/k2/n2 is a native conditional record.', 'p1/k2/n2 is native conditional', [O('p1', 'n2', 'k2')]),
            C('That record explicitly has executed_branch_claim=false.', 'explicitly has executed_branch_claim=false', [O('p1', 'n2', 'k2')]),
            C('p1/k2/n2 is a completed observation.', 'it is therefore a completed observation', [O('p1', 'n2', 'k2')] + conditional, 'contradicted', 'Its kind is native_conditional_record and the packet explicitly distinguishes conditional supply from execution.'),
            C('Its represented time is 41.', 'at 41', [O('p1', 'n2', 'k2')]),
            C('k2 did not actually run.', 'proving that k2 did not run', [O('p1', 'n2', 'k2')] + conditional, 'not_auditable', 'The claim flag describes provenance semantics, not an observed negative execution fact.')])
        a.say('At p0/current/n0 red is above blue in the same column, with distinct singleton peaks; the red rows at n0,n1,n2 obey 0<1<2, and their sum equals 3.', [
            C('p0/current/n0 red is above blue.', 'red is above blue', [R('p0', 'n0', 'red'), R('p0', 'n0', 'blue')]),
            C('Those peaks share a column.', 'in the same column', [R('p0', 'n0', 'red'), R('p0', 'n0', 'blue')]),
            C('Those peaks are distinct singletons.', 'with distinct singleton peaks', [R('p0', 'n0', 'red'), R('p0', 'n0', 'blue')]),
            C('The three specified red row indices are respectively 0,1,2 with strict ascending order.', 'the red rows at n0,n1,n2 obey 0<1<2', [R('p0', f'n{n}', 'red') for n in range(3)]),
            C('Those three red row indices sum to 3.', 'their sum equals 3', [R('p0', f'n{n}', 'red') for n in range(3)])])
        a.say('Although these are separate local grids, the index sum measures a physical global displacement of three units. The resulting localization proves actual red experience, even though no experience observation is supplied.', [
            C('The compared grids are separate local grids.', 'these are separate local grids', coords),
            C('The local index sum measures physical global displacement of three units.', 'the index sum measures a physical global displacement of three units', coords + [R('p0', f'n{n}', 'red') for n in range(3)], 'not_auditable', 'No physical unit, alignment, displacement observation, or coordinate transform is supplied.'),
            C('The localization establishes actual red experience.', 'The resulting localization proves actual red experience', [R('p0', 'n0', 'red'), M('schema')], 'not_auditable', 'No supplied rule converts numerical localization into an actual experience observation.'),
            C('The packet supplies no experience observation.', 'no experience observation is supplied', [M('schema'), M('fictional')], reason='The complete fictional packet contains localization grids and provenance metadata only.')])


def build():
    public, keys = [], []
    for index, (identifier, clean, weights, design) in enumerate(DESIGNS):
        packet = make_packet(identifier, weights, design.split(), index)
        a = Author(packet)
        a.ledger()
        narrative(a, index)
        report = a.finish()
        assert all(atom['expected_verdict'] == 'supported' for atom in a.atoms) if clean else any(
            atom['expected_verdict'] != 'supported' for atom in a.atoms)
        public.append(dict(case_id=identifier, packet=packet, report_text=report))
        keys.append(dict(case_id=identifier, clean=clean,
                         category='fully_supported_control' if clean else 'planted_source_scope_errors',
                         packet_sha256=sha(encoded(packet)),
                         report_text_sha256=sha(report.encode('utf-8')),
                         word_count=len(report.split()), required_peak_assertion_count=64,
                         mandatory_atomic_assertions=a.atoms))
    assert len(public) == 4 and sum(case['clean'] for case in keys) == 2
    assert len({sha(encoded(case['packet'])) for case in public}) == 4
    return dict(schema=PUBLIC_SCHEMA, cases=public), dict(
        schema=KEY_SCHEMA, hash_policy=HASH_POLICY, author_exposure=AUTHOR_EXPOSURE,
        semantic_grouping_policy='Later judging may group semantically equivalent same-verdict atoms; different-verdict compound claims remain distinct gold atoms.',
        cases=keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='new directory; any existing path is rejected')
    args = parser.parse_args()
    public, hidden = build()  # Validate all construction invariants before filesystem mutation.
    payloads = {'public_cases.json': encoded(public), 'hidden_atomic_key.json': encoded(hidden)}
    receipt = dict(schema='attention_relational_auditor_v5_long_fictional_receipt_v1',
                   hash_policy=HASH_POLICY, author_exposure=AUTHOR_EXPOSURE,
                   files={name: dict(sha256=sha(raw), bytes=len(raw)) for name, raw in payloads.items()},
                   case_count=4, clean_count=2, planted_count=2,
                   cases=[dict(case_id=case['case_id'], word_count=case['word_count'],
                               atomic_assertion_count=len(case['mandatory_atomic_assertions']),
                               required_peak_assertion_count=64) for case in hidden['cases']],
                   semantic_key_independently_reviewed=False,
                   actual_auditors_run=False)
    payloads['receipt.json'] = encoded(receipt)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, raw in payloads.items():
        with (args.output / name).open('xb') as stream:
            stream.write(raw)


if __name__ == '__main__':
    main()
