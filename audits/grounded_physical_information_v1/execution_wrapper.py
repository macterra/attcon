"""Preserve full child output and terminal receipt for run or exact replay."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('audits/grounded_physical_information_v1')
assert len(sys.argv) == 2 and sys.argv[1] in ('run', 'verify')
stage = sys.argv[1]
label = 'execution' if stage == 'run' else 'replay'
command = ['.venv/bin/python', '-u', 'scripts/grounded_physical_information_v1.py',
           '--stage', stage]
receipt_path = ROOT/(label+'_receipt.json')
assert not receipt_path.exists()
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
clock_start = time.monotonic()
environment = os.environ.copy()
environment['PYTHONPATH'] = 'src:scripts'
log_path = ROOT/(label+'.log')
with log_path.open('x') as log:
    process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        log.write(line)
        log.flush()
        print(line, end='', flush=True)
    exit_code = process.wait()
receipt = {'stage': stage, 'command': command, 'started_utc': started,
           'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'elapsed_seconds': time.monotonic()-clock_start, 'exit_code': exit_code,
           'log_sha256': hashlib.sha256(log_path.read_bytes()).hexdigest(),
           'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'goal_achieved': False}
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(receipt), flush=True)
sys.exit(exit_code)
