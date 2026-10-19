"""Separate single handcrafted acceptance test of the typed response schema."""
from pathlib import Path

import diagnose_attention_relational_api_v1 as diagnostic

diagnostic.OUTPUT_ROOT = Path(
    'audits/attention_relational_report_v1/typed_schema_acceptance/run')

if __name__ == '__main__':
    diagnostic.main()
