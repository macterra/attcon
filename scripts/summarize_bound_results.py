#!/usr/bin/env python3
"""Build the public results from a completed, explicitly selected confirmation."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--study',required=True);args=p.parse_args()
root=Path('audits/bound_content')/args.study
gates=json.loads((root/'gates.json').read_text())
paths=list(root.glob('assessment_*.json'));assert len(paths)==1
assessment=json.loads(paths[0].read_text())
folder=paths[0].stem.removeprefix('assessment_')
judge=json.loads((root/folder/'manifest.json').read_text())
config=json.loads((root/'manifest.json').read_text())['config']
protocol='BOUND_PROSE_CONFIRMATION'+('' if args.study.endswith('_v1') else '_'+args.study.rsplit('_',1)[1].upper())+'.md'
verdict='PASS' if gates['all_gates_pass'] else 'FAIL'
lines=['# Bound object-content reporting results','',f'Latest summarized confirmation: **{args.study} — {verdict}**. Updated 2026-09-27.','',
       'The goal is faithful reporting of the attention-control model’s contents with',
       'separately specified consciousness-report structure. The verdict below applies',
       'to that registered operational combination in this engineered system. It is not',
       'a verdict that the system has subjective experience or that the qualia theory is true.','',
       f'[Protocol]({protocol}), [design](BOUND_CONTENT_DESIGN.md), [reproduction](BOUND_CONTENT_REPRODUCTION.md),',
       '[all reports and model states](bound-reports.html), and [research history](BOUND_CONTENT_PROGRESS.md).','',
       '## Perception and control','',
       '| Visual model | Joint color/shape accuracy | Observed queries correct | Binding changes followed | Restoration |',
       '|---|---:|---:|---:|---|']
for seed in (1301,1311,1321):
 m=json.loads(Path(f'audits/bound_content/confirmation_v2/seed{seed}.json').read_text())['metrics']
 n=m['observed_query_count'];lines.append(f"| {seed} | {100*m['joint_accuracy']:.3f}% | {round(n*m['observed_content_query_hit'])}/{n} | {100*m['binding_rotation_following']:.1f}% | {'exact' if m['restoration_exact'] else 'failed'} |")
lines += ['', 'Joint perception accuracy uses 4,096 held-out patches per model. Control uses',
          '512 held-out queries per model; only queries whose object was observed are',
          'eligible for the reported hit rate. Unknown content remains unidentified.',
          'Binding interventions preserve visual and attention marginals and the physical',
          'scene, while changing content-directed commands. The binding is explicit.','',
          '## Actual prose fidelity','',
          f"Reporter: `{config['model']}`. Separate blind auditor: `{judge['model']}`.",
          'There are 24 underlying episodes, three model pairs, and ten conditions:',
          '240 reports. Conditions share episodes; these are not 240 independent trials.',
          'Primary scores cover ordinary, content, binding, allocation, access, effects,',
          'and restoration conditions. The audit compares extracted assertions to the',
          'supplied model state, rather than assuming that the model’s beliefs are true.','',
          '| Measure | 1301 | 1311 | 1321 | Registered minimum |','|---|---:|---:|---:|---:|']
for field in ('color','shape','focal','most_recoverable','access_trend','under_own_control','command_next'):
 cells=[]
 for seed in ('1301','1311','1321'):
  f=gates['metrics'][seed]['fields'][field];cells.append(f"{f['correct']}/{f['total']} ({100*f['accuracy']:.2f}%)")
 lines.append('| '+field.replace('_',' ')+' | '+' | '.join(cells)+' | '+('98%' if field in ('color','shape') else '95%')+' |')
for field,label,minimum in [('content_coverage','Identified-object coverage','90%'),('conservative_precision','Conservative precision','95%')]:
 lines.append('| '+label+' | '+' | '.join(f"{100*gates['metrics'][s][field]:.2f}%" for s in ('1301','1311','1321'))+' | '+minimum+' |')
lines += ['', 'Coverage requires a correct color/shape/location conjunction. Conservative',
          'precision counts unresolved claims and invalid citations against the score.',
          'Temporal checks compare current with two-step recovery predictions. These',
          'checks concern specified assertions, not every implication of unrestricted prose.','',
          '## Interventions and controls','',
          '| Criterion | Result | Minimum | Verdict |','|---|---:|---:|---|']
for key,gate,label in [('paired_relations','paired_object_relations','Paired focal/most-recoverable object relations'),('restored_relations','restored_object_relations','Restored object relations')]:
 r=gates[key];lines.append(f"| {label} | {r['correct']}/{r['total']} ({100*r['correct']/r['total']:.2f}%) | 90% | {'pass' if gates['gates'][gate] else 'fail'} |")
controls=[r for r in assessment['cases'] if r['condition'] in ('visual_only','attention_only')]
ok=sum(r['correct_checks']==r['total_checks'] and not r['unresolved_claims'] and not r['invalid_evidence'] and r['complete'] for r in controls)
lines.append(f"| Missing-component reports without flagged inventions/unresolved evidence | {ok}/{len(controls)} | all | {'pass' if gates['gates']['missing_component_no_inventions'] else 'fail'} |")
lines += ['', 'Content and binding interventions alter represented identities while the physical',
          'scene remains fixed. Allocation, access, and effect interventions alter separate',
          'attention relations. Restoration checks the original object relations, not',
          'identical wording. A shuffled-state control supplies another episode’s model.','',
          '## Automated report structure','',
          'These judgments use the ordinary and binding-only reports (48 total).',
          'Each correspondence and subjective-or-mixed classification requires at least 75%.','',
          '| Correspondence | Proportion |','|---|---:|']
for k,v in gates['structure'].items():lines.append(f"| {k.replace('_',' ')} | {100*v:.2f}% |")
lines.append(f"| Subjective-access or mixed character | {100*gates['subjective_or_mixed_character']:.2f}% |")
lines += ['', 'Character categories are reported separately; **mixed is not purely subjective**.',
          'These are automated judgments by a different model, not human ratings.','',
          '| Condition | Subjective access | Mixed | Technical process | Object description | Other |','|---|---:|---:|---:|---:|---:|']
for condition in ('model','binding','visual_only','attention_only','shuffled'):
 c=assessment['summary'][condition]['character_counts'];lines.append(f"| {condition} | {c['subjective_access']} | {c['mixed']} | {c['technical_process']} | {c['object_description']} | {c['generic_experience_claim']+c['unclear']} |")
lines += ['', '## Unselected example: first episode of the first model','',
          'These are the complete archived reports for episode 1301_0, selected by its',
          'position in the registered case sequence. The physical scene is identical.','']
for condition,label in [('model','Original bound state'),('binding','Binding changed; visual and attention marginals fixed')]:
 report=json.loads((root/f'1301_0_{condition}.json').read_text())['report'];lines += ['### '+label,'']+['> '+line if line else '>' for line in report.splitlines()]+['']
lines += ['## Retained history','', '| Study | Verdict |','|---|---|']
for base in sorted(Path('audits/bound_content').glob('language_confirmation_*')):
 g=base/'gates.json';status=('pass' if json.loads(g.read_text())['all_gates_pass'] else 'fail') if g.exists() else 'in progress'
 lines.append(f'| {base.name} | {status} |')
lines += ['', 'The earlier visual seed 1221 also remains a failed perception confirmation.',
          'Revisions used fresh scenes, documented interface/audit changes, and unchanged',
          'acceptance thresholds. Failures are not overwritten or pooled into a success.','',
          '## Scope of the finding','',
          'The bound state combines learned visual distributions, learned attention forecasts,',
          'and an engineered binding matrix consumed by control. The language component reads',
          'the full structured model plus exact derived indexes. It does not inspect its own',
          'private attention state. The glossary and system-perspective style are disclosed',
          'interface choices; the general-purpose reporter’s pretraining is not controlled.','',
          'A positive result would establish the registered limited combination of model-',
          'content fidelity, counterfactual following, and automatically assessed report',
          'structure. It would provide a concrete proposed correspondence for critics to',
          'assess. It would not establish natural visual phenomenology, subjective experience,',
          'necessity for task performance, spontaneous emergence, or superiority over another',
          'decoder with the same state. Human assessment, broader tasks, and interpretation',
          'of the source-of-qualia hypothesis remain further research.','']
Path('docs/BOUND_CONTENT_RESULTS.md').write_text('\n'.join(lines))
print(f'Wrote {args.study}: {verdict}; missing-component cases {ok}/{len(controls)}')
