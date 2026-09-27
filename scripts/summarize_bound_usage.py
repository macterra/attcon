#!/usr/bin/env python3
"""Recorded API token usage, valued at the experiment's registered unit prices."""
import json
from collections import defaultdict
from pathlib import Path

root=Path('audits/bound_content')
prices={'gpt-5-mini-2025-08-07':(.25,.025,2.),'gpt-4.1-2025-04-14':(2.,.5,8.)}
groups=defaultdict(lambda:{'responses':0,'input_tokens':0,'cached_input_tokens':0,'output_tokens':0,'estimated_usd':0.})
errors=[];unfinished=[]
for path in sorted(root.rglob('*.json')):
    result=json.loads(path.read_text())
    if not isinstance(result,dict):continue
    if result.get('status')=='error':errors.append(str(path));continue
    if result.get('status')=='attempt_reserved':unfinished.append(str(path));continue
    response=result.get('response')
    if not isinstance(response,dict) or 'usage' not in response:continue
    model=response['model'];usage=response['usage'];key=str(path.parent.relative_to(root))
    row=groups[key];row['model']=model;row['responses']+=1
    inputs=usage['input_tokens'];cached=usage.get('input_tokens_details',{}).get('cached_tokens',0);outputs=usage['output_tokens']
    for field,value in [('input_tokens',inputs),('cached_input_tokens',cached),('output_tokens',outputs)]:row[field]+=value
    a,b,c=prices[model];row['estimated_usd']+=((inputs-cached)*a+cached*b+outputs*c)/1e6
out={'status':'Token-derived estimate at registered prices, not a billing invoice. Failed requests without usage may incur unrecorded charges.',
     'prices_usd_per_million':{m:dict(zip(('uncached_input','cached_input','output'),p)) for m,p in prices.items()},
     'groups':dict(groups),'total_responses':sum(r['responses'] for r in groups.values()),
     'total_estimated_usd':sum(r['estimated_usd'] for r in groups.values()),'errors':errors,'unfinished':unfinished}
(root/'usage.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
