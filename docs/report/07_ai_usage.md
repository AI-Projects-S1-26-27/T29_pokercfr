# 7. AI usage

Each member records which tools they used, for what, and how each output was verified. "The AI
generated it" is not a defence, so what matters is the verification.

## Aarush Kanipakam
_Aarush to confirm and edit; this entry was drafted with the tool's help and must reflect what
actually happened._

- **Tool:** Claude Code (Anthropic), used as a coding assistant for steps 1-7 of the individual
  plan: the audit script, the exploitability module, the pre-registration, the results
  aggregation script, the Makefile, and the drafts of this report's sections.
- **Verification:** every script has tests (`tests/test_validation.py`,
  `tests/test_exploitability.py`, `tests/test_registry.py`, `tests/test_results_table.py`,
  `tests/test_report_numbers.py`) and the full suite was run before each push. The audit was
  checked against OpenSpiel. The hypotheses' decision rules and the report text were written as
  proposals by the assistant and must be reviewed and owned by the author before submission.

## Everyone else
_TODO: each member writes their own entry._
