"""Retain terminal status and log for the one reserved run or exact verification."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
folder = root/'audits/grounded_attention_outcomes_v1'
mode = sys.argv[1]
assert mode in ('run', 'verify')
name = 'execution' if mode == 'run' else 'verification'
log = folder/(name+'.log')
receipt = folder/(name+'_receipt.json')
assert not log.exists() and not receipt.exists(), 'execution receipt already reserved'
start = datetime.datetime.now(datetime.timezone.utc).isoformat()
environment = dict(os.environ, PYTHONPATH=str(root/'src')+':'+str(root/'scripts'))
with log.open('x') as stream:
    child = subprocess.Popen([str(root/'.venv/bin/python'), 'scripts/grounded_attention_outcomes_v1.py', mode],
        cwd=root, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in child.stdout:
        stream.write(line); stream.flush(); print(line, end='', flush=True)
    code = child.wait()
receipt.write_text(json.dumps({'mode':mode,'started_utc':start,
    'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'exit_code':code,'terminal_success':code==0,
    'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()},indent=2)+'\n')
sys.exit(code)
