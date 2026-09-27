#!/usr/bin/env python3
"""Publish actual reports beside the physical scene and bound internal contents."""
import argparse
import json
from pathlib import Path
import torch
from attcon.bound_content import COLORS,SHAPES,LOCATIONS

p=argparse.ArgumentParser();p.add_argument('--study',default='language_confirmation_v2');args=p.parse_args()
root=Path('audits/bound_content')/args.study
requests=json.loads((root/'requests.json').read_text());states=torch.load(root/'source_states.pt',weights_only=True)
data={}
for req in requests:
    key=f"{req['seed']}_{req['episode']}"
    if key not in data:
        st=states[req['seed']];i=req['episode']
        physical=[{'view':'AB'[v],'objects':[{'location':LOCATIONS[s],'color':COLORS[int(st['colors'][i,v,s])], 'shape':SHAPES[int(st['shapes'][i,v,s])]} for s in range(4)]} for v in range(2)]
        data[key]={'physical':physical,'conditions':{}}
    response=json.loads((root/(req['id']+'.json')).read_text())
    data[key]['conditions'][req['condition']]={'source':req['source'],'report':response.get('report','No completed report.'),'id':req['id']}
protocol='BOUND_PROSE_CONFIRMATION_V2.html' if args.study=='language_confirmation_v2' else 'BOUND_PROSE_CONFIRMATION.html'
encoded=json.dumps(data).replace('<','\\u003c')
html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Attcon: inspect object-linked reports</title>
<style>body{max-width:1050px;margin:2rem auto;padding:0 1rem;color:#192d3e;background:#f8fafc;font:17px/1.5 system-ui}select,button{font:inherit;padding:.4rem;margin:.2rem}label{display:inline-block;margin:.5rem 1rem .5rem 0}.views{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.view{background:white;border:1px solid #ccd5df;border-radius:8px;padding:1rem}.board{display:grid;grid-template-columns:repeat(3,1fr);grid-template-rows:repeat(3,105px);gap:.2rem}.cell{text-align:center;border:2px solid transparent;border-radius:8px;font-size:13px}.cell.focus{border-color:#172e58;background:#eef3ff}.cell svg{display:block;margin:0 auto;width:48px;height:48px}.bar{height:5px;background:#cad4e0;max-width:80px;margin:auto}.fill{height:5px;background:#476d9c}#report{white-space:pre-wrap;background:white;border-left:4px solid #375d83;padding:1.2rem;font:18px/1.6 system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}summary{cursor:pointer}h2{font-size:1.25rem}small{color:#45566a}@media(max-width:650px){.views{grid-template-columns:1fr}.board{grid-template-rows:repeat(3,100px)}}</style>
<h1>Reports of object-linked attention-model contents</h1>
<p><strong>Archive: STUDY_NAME.</strong> Automated report judgments are not evidence of subjective experience.</p>
<p>Select a recorded episode and change the internal-state condition. The physical scene stays fixed during the listed interventions. Reports below are the actual generated prose.</p>
<p><a href="BOUND_CONTENT_PROGRESS.html">Study progress and evidence limits</a> · <a href="PROTOCOL_LINK">Registered confirmation</a></p>
<label>Episode <select id="episode"></select></label><label>Condition <select id="condition"></select></label>
<p id="caseid"></p>
<h2>Represented contents</h2><p><small>Symbols show the model's identified color and shape (probability at least 0.6). A border marks its strongest current attention allocation. Bars show predicted current recoverability; ? means identity is uncertain. These visual encodings illustrate stored values.</small></p>
<div class="views" id="model"></div>
<h2>Actual report</h2><div id="report"></div>
<details><summary>Physical scene</summary><div class="views" id="physical"></div></details>
<details><summary>Complete supplied model record</summary><pre id="source"></pre></details>
<script>
const data=DATA;
const labels={model:'Original model',content:'Object representations changed',binding:'Bindings changed',allocation:'Attention allocation changed',access:'Access forecasts changed',effects:'Command effects changed',restored:'Bindings restored',visual_only:'Object information only',attention_only:'Attention information only',shuffled:'Another episode’s model'};
const el=id=>document.getElementById(id);const colors={red:'#d73737',green:'#29944b',blue:'#3874cd',yellow:'#e0bb24'};
function top(d){if(!d)return null;const xs=Object.entries(d).sort((a,b)=>b[1]-a[1]);return xs[0][1]>=.6?xs[0][0]:null;}
function icon(color,shape){const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 50 50');if(!color||!shape){const t=document.createElementNS(svg.namespaceURI,'text');t.setAttribute('x','18');t.setAttribute('y','35');t.setAttribute('font-size','30');t.textContent='?';svg.append(t);return svg;}let node;
if(shape==='circle'){node=document.createElementNS(svg.namespaceURI,'circle');node.setAttribute('cx','25');node.setAttribute('cy','25');node.setAttribute('r','17');}
else if(shape==='square'){node=document.createElementNS(svg.namespaceURI,'rect');node.setAttribute('x','8');node.setAttribute('y','8');node.setAttribute('width','34');node.setAttribute('height','34');}
else{node=document.createElementNS(svg.namespaceURI,'path');node.setAttribute('d',shape==='triangle'?'M25 5 L45 43 L5 43 Z':'M19 5 H31 V19 H45 V31 H31 V45 H19 V31 H5 V19 H19 Z');}
node.setAttribute('fill',colors[color]||'#888');svg.append(node);return svg;}
const positions={upper:[1,2],right:[2,3],lower:[3,2],left:[2,1]};
function boards(target,views,physical){target.replaceChildren();for(const v of views){const card=document.createElement('div');card.className='view';const heading=document.createElement('strong');heading.textContent='View '+v.view;card.append(heading);const board=document.createElement('div');board.className='board';const ps=v.objects.map(o=>o.selection_probability);const maximum=physical||ps.some(x=>x==null)?null:Math.max(...ps);
for(const o of v.objects){const cell=document.createElement('div');cell.className='cell';if(maximum!=null&&o.selection_probability===maximum&&ps.filter(x=>x===maximum).length===1)cell.classList.add('focus');const [r,c]=positions[o.location];cell.style.gridRow=r;cell.style.gridColumn=c;const color=physical?o.color:top(o.color_distribution),shape=physical?o.shape:top(o.shape_distribution);cell.append(icon(color,shape));const label=document.createElement('div');label.textContent=o.location+': '+(color||'?')+' '+(shape||'?');cell.append(label);const q=o.recoverability_now_then_one_then_two_steps;if(!physical&&q){const bar=document.createElement('div');bar.className='bar';const fill=document.createElement('div');fill.className='fill';fill.style.width=(100*q[0])+'%';bar.append(fill);cell.append(bar);const n=document.createElement('small');n.textContent=Math.round(q[0]*100)+'% recoverability';cell.append(n);}board.append(cell);}card.append(board);target.append(card);}}
function show(){const episode=data[el('episode').value],row=episode.conditions[el('condition').value];el('caseid').textContent=row.id;el('report').textContent=row.report;el('source').textContent=JSON.stringify(row.source,null,2);boards(el('model'),row.source,false);boards(el('physical'),episode.physical,true);}
for(const key of Object.keys(data)){const o=document.createElement('option');o.value=key;o.textContent=key.replace('_',' · episode ');el('episode').append(o);}
for(const [key,label]of Object.entries(labels)){const o=document.createElement('option');o.value=key;o.textContent=label;el('condition').append(o);}
el('episode').onchange=show;el('condition').onchange=show;show();
</script></html>'''.replace('DATA',encoded).replace('STUDY_NAME',args.study).replace('PROTOCOL_LINK',protocol)
Path('docs/bound-reports.html').write_text(html)
print(f'Built {len(data)} episodes and {len(requests)} reports')
