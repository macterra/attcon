"""One unscored handcrafted resource check; never retries scientific requests."""
from pathlib import Path

import diagnose_attention_relational_api_v1 as diagnostic


if __name__ == '__main__':
    diagnostic.OUTPUT_ROOT = Path(
        'audits/attention_relational_report_v1/resource_acceptance/run')
    diagnostic.main()
