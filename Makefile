# Every number in the report comes from a committed script and traces to results/.
# `make all` regenerates everything from scratch.
PY ?= python
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -ec

.PHONY: test verify results figures report all

test:
	pytest -q

# The audit and the OpenSpiel cross-checks exit non-zero on any disagreement.
verify:
	$(PY) scripts/a1_br_audit.py | tee results/verify/br_audit.log
	$(PY) scripts/s5_openspiel_leduc_crosscheck.py | tee results/verify/leduc_crosscheck.log
	@if [ -f scripts/integration_check.py ]; then \
		$(PY) scripts/integration_check.py | tee results/verify/integration.log; \
	else echo "NOTE: scripts/integration_check.py does not exist yet; skipped"; fi

results:
	$(PY) scripts/a6_results_table.py

# Figures live in pokercfr/analysis/plots.py (Akhil). Fail loudly rather than
# silently producing nothing while it has no entry point.
figures:
	$(PY) -c "import sys, pokercfr.analysis.plots as p; \
	sys.exit(\"pokercfr.analysis.plots has no main(): figures not generated\") \
	if not hasattr(p, \"main\") else p.main()"

report:
	$(PY) scripts/a7_report_numbers.py

all: verify results figures report
