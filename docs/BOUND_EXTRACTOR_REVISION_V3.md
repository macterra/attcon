# Extractor v3: retain a failed control-attribution fixture

Extractor v2 passes 7/8 fixtures but omits the explicit control assertion in
"my commands can direct attention to ...". It is not used to pass a confirmation.
V3 clarifies that explicit steering/directing assertions count as command control
and uses fresh wording for the positive and negative control fixtures. The schema,
verbatim-span checks, out-of-vocabulary test, and command-destination test remain.

Run 8 fixture requests, then at most40 pilot-v2 and40 pilot-v3 extraction requests.
Model `gpt-4.1-2025-04-14`, max8192 output tokens, no retries. Maximum88 attempts,
output ceiling$5.767168 at registered prices; actual usage is archived. These
calls replace unperformed v2 extraction calls; v2's eight failed-fixture results
are retained separately. Do not re-run a failed call to obtain a preferred answer.
