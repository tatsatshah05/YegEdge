---
description: Audit the risk manager — every rule has a deliberate-trigger test
---

Audit the risk-manager test coverage.

Steps:

1. Read `config/risk_rules.yaml` and enumerate every rule (every YAML leaf that represents a configurable risk limit).
2. Read `tests/test_risk.py`.
3. For each rule in `config/risk_rules.yaml`, identify whether a test exists that:
   - Loads the rule's value from config (does not hard-code).
   - Constructs a scenario where the rule would be triggered (e.g., a portfolio at exposure cap, a daily loss approaching the cap, etc.).
   - Asserts the risk manager rejects the trade *or* caps the size *or* triggers the kill switch as appropriate.
4. Report the findings in this exact format:

```
# Risk Manager Audit — {YYYY-MM-DD}

## Rule coverage

| Rule (YAML path)                          | Test exists | Loads from config | Deliberately triggered | Status |
|-------------------------------------------|-------------|-------------------|------------------------|--------|
| per_trade.max_risk_fraction               | ✓           | ✓                 | ✓                      | PASS   |
| per_trade.max_position_fraction           | ✓           | ✓                 | ✓                      | PASS   |
| portfolio.max_sector_exposure             | ✗           |                   |                        | MISSING |
...
```

5. For any rule marked MISSING or PARTIAL, propose the exact pytest function to add. Do not write the test in the audit — only propose. Surface it for review.

6. Run `pytest tests/test_risk.py -m risk -v` and report the result. Any test failure is a hard issue — surface it as CRITICAL.

7. Confirm that no code in `agent/risk/` hard-codes a risk value. Grep for the patterns `0.005`, `0.02`, `0.08`, `6`, `0.15`, etc. and verify each is sourced from config, not literal.

Output format: concise table + actionable list. Maximum 200 lines.

This command is mandatory before any merge that touches `agent/risk/`.
