from review_tools import *
r=row(1)
save(1,[],[],[],[], 'unestablished; Original diagnostics retained: '+json.dumps(r['reference_measurement']['derivation'],ensure_ascii=True,sort_keys=True)+'. Empty semantic inventories are required; no labels are salvaged or inferred.')
