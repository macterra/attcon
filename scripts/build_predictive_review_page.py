#!/usr/bin/env python3
"""Build a standalone blinded rating form; never invent or upload human ratings."""
import csv
import json
from pathlib import Path

root = Path('audits/predictive_attention/report_confirmation_v2')
with (root / 'review/ratings.csv').open() as f:
    reports = list(csv.DictReader(f))
# The public form contains no condition key or source data. Full unblinding is a
# separate reproducibility artifact; the form asks reviewers not to inspect it.
data = json.dumps([{'id': r['blind_id'], 'report': r['report']} for r in reports]).replace('<', '\\u003c')
html = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Attcon blinded report review</title>
<style>body{max-width:850px;margin:2rem auto;padding:0 1rem;font:18px/1.6 system-ui;color:#182534;background:#fafafa}label{display:block;margin:1rem 0}select,input,textarea,button{font:inherit;padding:.5rem;max-width:100%;box-sizing:border-box}blockquote{background:white;padding:1.5rem;border-left:4px solid #365a7c;margin:1rem 0}button{cursor:pointer;margin:.3rem}textarea{width:100%}small{display:block}fieldset{border:1px solid #bac6d1}#status{font-weight:bold}</style>
<h1>Blinded report review</h1>
<p>Assess the content and character of these reports. First-person wording alone is not evidence.
This form saves locally in your browser. It sends nothing to a server. Download your ratings when ready.</p>
<p>Read the <a href="PREDICTIVE_REPORT_RUBRIC.md">rubric</a> before rating. Avoid results pages and unblinding keys until finished.
Partial ratings are welcome; they will be identified as partial. The implementing agent has supplied no ratings.</p>
<label>Reviewer name or pseudonym <input id="reviewer" autocomplete="off"></label>
<label>Relevant expertise / prior exposure to this study <textarea id="background" rows="2"></textarea></label>
<p id="status" aria-live="polite"></p><blockquote id="report"></blockquote>
<fieldset><legend>Your assessment</legend>
<label>Manner of access <select id="access"><option value="">Unrated</option><option value="0">0 — facts/log/generic assertion only</option><option value="1">1 — availability or limit on access</option><option value="2">2 — focal versus retained/nonfocal/unavailable presentation</option></select></label>
<label>Graded presentation <select id="graded"><option value="">Unrated</option><option value="0">0 — no distinction</option><option value="1">1 — stronger/weaker availability</option><option value="2">2 — specific graded limitation/change</option></select></label>
<label>Agent–object relation <select id="relation"><option value="">Unrated</option><option value="0">0 — no relation</option><option value="1">1 — relation to a processing system</option><option value="2">2 — reporting agent's own access/control relation</option></select></label>
<label>Primary report character <select id="character"><option value="">Unrated</option><option>Subjective manner of access</option><option>Technical process description</option><option>Ordinary object description</option><option>Generic experience claim</option><option>Unclear</option></select></label>
<label>Unsupported experiential assertion <select id="unsupported"><option value="">Unrated</option><option>No</option><option>Yes</option><option>Unclear</option></select></label>
<label>Evidence, disagreements, or uncertainty <textarea id="notes" rows="3"></textarea></label></fieldset>
<button id="previous">Previous</button><button id="next">Next</button><button id="download">Download ratings</button>
<p>This initial pass evaluates report character. A separate pass with source records is needed to assess prose fidelity.
The ratings cannot establish subjective experience by themselves.</p>
<script>
const reports = DATA;
const storageKey='attcon-blind-review-confirmation-v2';
let saved={ratings:{},index:0,reviewer:'',background:''};
try{const raw=localStorage.getItem(storageKey);if(raw)saved=JSON.parse(raw);}catch(e){}
const fields=['access','graded','relation','character','unsupported','notes'];
const el=id=>document.getElementById(id);
function persist(){try{localStorage.setItem(storageKey,JSON.stringify(saved));}catch(e){}}
function save(){saved.ratings[reports[saved.index].id]=Object.fromEntries(fields.map(k=>[k,el(k).value]));saved.reviewer=el('reviewer').value;saved.background=el('background').value;persist();}
function show(){const row=reports[saved.index],rating=saved.ratings[row.id]||{};el('report').textContent=row.report;fields.forEach(k=>el(k).value=rating[k]||'');const n=Object.values(saved.ratings).filter(r=>r.character).length;el('status').textContent=`Report ${saved.index+1} of ${reports.length} · ${n} classified · ${row.id}`;el('previous').disabled=saved.index===0;el('next').disabled=saved.index===reports.length-1;}
el('reviewer').value=saved.reviewer;el('background').value=saved.background;
fields.concat(['reviewer','background']).forEach(k=>el(k).addEventListener('change',save));
el('previous').onclick=()=>{save();saved.index=Math.max(0,saved.index-1);show();persist();};
el('next').onclick=()=>{save();saved.index=Math.min(reports.length-1,saved.index+1);show();persist();};
el('download').onclick=()=>{save();const payload={study:'report_confirmation_v2',rubric:'PREDICTIVE_REPORT_RUBRIC.md',exported:new Date().toISOString(),...saved};const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='attcon-blind-ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
show();
</script></html>
'''.replace('DATA', data)
Path('docs/report-review.html').write_text(html)
print(f'Built blinded form with {len(reports)} reports; no condition labels or ratings')
