---
description: Verify a code change does not introduce a SEBI compliance regression
---

Audit pending code changes against the SEBI April 2026 algo-trading framework.

Steps:

1. Run `git diff main` (or `git diff HEAD~5` if main isn't current) to get the diff.
2. Read `docs/sebi_compliance.md`.
3. Check the diff against each of these compliance requirements:

| Requirement | Check |
|-------------|-------|
| White-box only | Any code introducing opaque/encrypted strategy logic that's not viewable by the operator? |
| Algo-ID tagging | Does every new order-placement code path pass through `BrokerAdapter.place_order()` which carries the Algo-ID? |
| 10 OPS threshold | Does the diff introduce any loop or burst that could exceed 2 OPS (our config cap)? |
| Static-IP whitelisting | Does any new code make API calls from non-whitelisted IPs or add network paths that bypass the broker adapter? |
| TOTP login | Does any code attempt to programmatically bypass TOTP (e.g., direct credential POST, headless browser login)? |
| Personal use only | Does any new code support multi-user, multi-account, or third-party order routing? |
| F&O / options / futures | Does the diff introduce any non-equity-cash instrument type? |

4. For each requirement, report:
   - `OK` — no concerns in this diff.
   - `WARN` — borderline; surface for human review.
   - `CRITICAL` — clear violation; block merge.

5. Output format:

```
# SEBI Compliance Check — {commit_range}

## Diff summary
- Files changed: {n}
- Lines added: {n}, removed: {n}

## Compliance scan
- White-box only: {status} — {one-line-finding}
- Algo-ID tagging: {status} — {finding}
- 10 OPS threshold: {status} — {finding}
- Static-IP whitelisting: {status} — {finding}
- TOTP login: {status} — {finding}
- Personal use only: {status} — {finding}
- Equity-cash only: {status} — {finding}

## Verdict: {APPROVED | REVIEW REQUIRED | BLOCK MERGE}

## Action items (if any)
{list}
```

6. If any item is CRITICAL, do not merge — explicitly state "MERGE BLOCKED."

This command should run as part of any pre-commit or pre-merge workflow for code changes touching `agent/execution/`, `agent/data/`, `agent/risk/`, or `.env.example`.
