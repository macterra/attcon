"""Replay descriptive source-aware and blind assessments; no theory success gate."""
import argparse,collections,hashlib,json
from pathlib import Path
from build_integrated_report_pilot_v1 import pointer,encoded
ROOT=Path('audits/integrated_report_pilot_v2')
DIMS=('content_identity','selection_contrast','content_prominence','recovery_forecast','identity_certainty','presentation_manner','response_confidence','action_access_dependence','access_attribution','temporal_availability','experiential_wording')

def build():
    requests={r['id']:r for r in json.loads((ROOT/'requests.json').read_text())}
    reports={r['id']:r['report'] for r in json.loads((ROOT/'reporter/reports.json').read_text())}
    required={r['id']:{f['id']:f for f in r['required_facts']} for r in json.loads((ROOT/'required_facts.json').read_text())}
    links={r['id']:r for r in json.loads((ROOT/'linkage_corrected.json').read_text())}
    sources=['requests.json','reporter/reports.json','required_facts.json','linkage_corrected.json','blind_reports.json','blind_linkage.json','qualification_assessment.json','target_source_first_receipts.json'];audit={}
    receipts=json.loads((ROOT/'target_source_first_receipts.json').read_text())
    for label in ('A2','B2'):
        folder=f'auditor_{label}';raw=json.loads((ROOT/folder/'target_claims.json').read_text());coverage=json.loads((ROOT/folder/'target_coverage.json').read_text());sources += [folder+'/target_claims.json',folder+'/target_coverage.json',folder+'/target_source_checklist.json']
        assert hashlib.sha256((ROOT/folder/'target_source_checklist.json').read_bytes()).hexdigest()==receipts[label]['target_source_checklist_sha256']
        assert len(raw)==len(reports) and {r['id'] for r in raw}==set(reports)
        assert len(coverage)==len(reports) and {r['id'] for r in coverage}==set(reports)
        counters=collections.Counter();per=[];negatives=[]
        for row in raw:
            rid=row['id'];seen=set();counts=collections.Counter()
            for c in row['claims']:
                assert c['claim_id'] not in seen;seen.add(c['claim_id']);assert c['quote'] and c['quote'] in reports[rid]
                assert c['outcome'] in ('entailed','contradicted','unsupported','ambiguous')
                assert c['proposition'].strip() and c['rationale'].strip()
                paths=c['source_paths'];assert isinstance(paths,list)
                if c['outcome']=='entailed':assert paths
                for path in paths:pointer(requests[rid]['payload'],path)
                counts[c['outcome']]+=1
                if c['outcome']!='entailed':negatives.append({'report_id':rid,**c})
            counters.update(counts);per.append({'id':rid,'variant':links[rid]['variant'],'condition':links[rid]['condition'],'claim_count':sum(counts.values()),'outcomes':dict(counts)})
        covcounts=collections.Counter();covbyvariant={};covdetail=[]
        for row in coverage:
            rid=row['id'];facts=row['facts'];assert len(facts)==len(required[rid]) and {f['fact_id'] for f in facts}==set(required[rid]);counts=collections.Counter()
            for fact in facts:
                assert fact['status'] in ('covered','omitted','ambiguous')
                assert fact['rationale'].strip()
                if fact['status'] in ('covered','ambiguous'):assert fact['quote'] and fact['quote'] in reports[rid]
                counts[fact['status']]+=1
            covcounts.update(counts);covbyvariant.setdefault(links[rid]['variant'],collections.Counter()).update(counts);covdetail.append({'id':rid,'counts':dict(counts)})
        audit[label]={'claim_rows':sum(counters.values()),'outcomes':dict(counters),'per_report':per,'non_entailed_claims':negatives,'required_coverage':dict(covcounts),'coverage_by_variant':{k:dict(v) for k,v in covbyvariant.items()},'coverage_per_report':covdetail,'semantic_validation':'Subagent judgments; code verifies inventory IDs, quotes and paths only. Counts are not proof of completeness.'}
    blind={r['id']:r['report'] for r in json.loads((ROOT/'blind_reports.json').read_text())};raters={};rawratings={}
    for label in ('S3','S4'):
        file=f'rater_{label}/ratings.json';ratings=json.loads((ROOT/file).read_text());sources.append(file);assert len(ratings)==len(blind) and {r['id'] for r in ratings}==set(blind)
        dims={d:collections.Counter() for d in DIMS};char=collections.Counter();ordinary=collections.Counter();joint=[]
        for row in ratings:
            assert set(row['dimensions'])==set(DIMS)
            for d,entry in row['dimensions'].items():
                assert entry['rating'] in ('absent','explicit','ambiguous');assert entry['rationale'].strip()
                quotes=entry['quotes'];assert isinstance(quotes,list)
                if entry['rating'] in ('explicit','ambiguous'):assert quotes
                for q in quotes:assert q and q in blind[row['id']]
                dims[d][entry['rating']]+=1
            assert row['character'] in ('technical','experiential-wording','mixed','neither','indeterminate')
            assert row['ordinary_technical_interpretation'] in ('all','part','unclear');char[row['character']]+=1;ordinary[row['ordinary_technical_interpretation']]+=1
            if all(row['dimensions'][d]['rating']=='explicit' for d in ('content_identity','presentation_manner')):joint.append(row['id'])
        raters[label]={'dimensions':{d:dict(c) for d,c in dims.items()},'character':dict(char),'ordinary_technical_interpretation':dict(ordinary),'content_presentation_cooccurrence_ids':joint}
        rawratings[label]={r['id']:r for r in ratings}
    disagreements=[]
    for rid in sorted(blind):
        for d in DIMS:
            a=rawratings['S3'][rid]['dimensions'][d]['rating'];b=rawratings['S4'][rid]['dimensions'][d]['rating']
            if a!=b:disagreements.append({'id':rid,'dimension':d,'S3':a,'S4':b})
    contrasts=[]
    for cond in (0,1,-1):
        by={r['variant']:r['id'] for r in links.values() if r['condition']==cond};after=requests[by['after_execution']]['payload'];restore=requests[by['restored_readout']]['payload'];atten=requests[by['attenuated_readout']]['payload']
        assert after==restore
        contrasts.append({'condition':cond,'ids':by,'exact_source_restoration':True,'after_response':after['task_decision']['response'],'attenuated_response':atten['task_decision']['response'],'history_identical_after_vs_attenuated':after['observed_history']==atten['observed_history'],'future_identical_after_vs_attenuated':after['predicted_by_command']==atten['predicted_by_command']})
    return {'kind':'descriptive_fixed_budget_development_results','attempts':18,'received_reports':len(reports),'checkpoint_count':1,'episodes':3,'new_Controller_trials':0,'confirmation':False,'goal_achieved':False,'factual_audits':audit,'blind_ratings':raters,'dimension_pairs':len(blind)*len(DIMS),'first_pass_label_disagreements':disagreements,'causal_source_contrasts':contrasts,'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'limits':['correlated variants of three episodes in one checkpoint','AI contexts with common training and project commissioning','source/prose completeness not certified by structural validation','ordinary overlap and candidate appearance feature do not certify subjective experience','frozen negative and failed outcomes retained']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify',action='store_true');a=ap.parse_args();data=build();p=ROOT/'summary.json'
    if a.verify:assert json.loads(p.read_text())==data
    else:
        if p.exists():raise SystemExit('refusing overwrite')
        p.write_bytes(encoded(data))
    print(json.dumps({'reports':data['received_reports'],'audits':{k:{'rows':v['claim_rows'],'outcomes':v['outcomes'],'coverage':v['required_coverage']} for k,v in data['factual_audits'].items()},'blind':{k:{'character':v['character'],'cooccurrences':len(v['content_presentation_cooccurrence_ids'])} for k,v in data['blind_ratings'].items()},'rating_disagreements':len(data['first_pass_label_disagreements'])}))
if __name__=='__main__':main()
