# SEBI Compliance — Operating Posture

This document summarizes how the trading agent operates within SEBI's algorithmic-trading framework. **Read this before changing any code that touches order routing, broker authentication, or system networking.**

## Legal classification of this system

- **Type:** White-box algorithm. Strategy logic is fully visible to the operator (you).
- **Operator scope:** Own-account only. Single demat account belonging to the owner. No third-party trading.
- **Order rate:** Sub-10 orders per second per exchange (well below the threshold for mandatory registration).
- **Conclusion:** This system falls under SEBI's "regular API user" classification. **No separate SEBI or exchange registration is required.**

## Framework reference

- **Circular:** SEBI/HO/MIRSD/MIRSD-PoD/P/CIR/2025/9 dated February 4, 2025.
- **Title:** "Safer participation of retail investors in Algorithmic trading."
- **Mandatory from:** April 1, 2026 (after timeline extensions).

## What the framework requires (and what we comply with)

### 1. White-box vs black-box

Algorithms must be classified. White-box = operator has full visibility; black-box = proprietary/hidden.

**This system:** Fully white-box. The operator can read every line of strategy code, every risk rule, every feature transform. Black-box providers would require SEBI Research Analyst registration, which we do not need because we are not providing the algorithm to anyone else.

### 2. Algo-ID tagging

Every algorithmic order must carry an exchange-issued unique identifier so the exchange can audit any automated order back to its origin.

**This system:** The broker (Upstox/Dhan/Kite) handles Algo-ID tagging. Our broker adapter must request and pass the Algo-ID through with every order. The `BrokerAdapter` interface requires this; concrete implementations are responsible.

### 3. Order-rate threshold

Below 10 orders per second per exchange = regular API user, no registration. Above 10 OPS = mandatory algo registration.

**This system:** Configured for 2 orders per second maximum (see `config/risk_rules.yaml` → `frequency.max_orders_per_second`). The risk manager enforces this at the application layer; the broker enforces it at the API layer. We are comfortably below the threshold.

### 4. Static IP whitelisting

Brokers must accept API requests only from client-specific API keys tied to static IPs whitelisted in advance. Open APIs are disallowed.

**This system:**
- During development: whitelist your home internet IP (note: residential ISPs may issue dynamic IPs; you'll need to re-whitelist if it changes).
- For paper trading: use a VPS with static IP (Hetzner / DigitalOcean Bangalore / AWS Mumbai).
- For live trading: same VPS, never run from anywhere else.
- The `.env` file has a `STATIC_IP` field which is logged at startup; the system warns if it doesn't match the broker-whitelisted IP.

### 5. Daily TOTP login

Brokers must enforce TOTP-based authentication; trader must log in manually at least once per day.

**This system:**
- The owner logs in manually each morning before market open (~30 seconds).
- The system has a CLI command (`python -m agent.cli login`) that walks through the interactive flow and saves the day's access token to `.env` (under `UPSTOX_ACCESS_TOKEN` or equivalent).
- The token is valid until 6 AM IST the next day; the agent uses it for everything during the trading session.
- **Programmatic TOTP bypass is not implemented and must not be added.** This violates broker T&C and creates regulatory tail risk.

### 6. Personal use only

Registered algorithms may be used within the investor's family (self, spouse, dependent children/parents) and not for anyone else.

**This system:** Single-account by design. The architecture has no concept of "user" or "client" — there is one set of credentials, one portfolio, one journal. Adding multi-user capability is out of scope.

### 7. Broker compliance

Brokers must implement infrastructure: API gateway restrictions, IP whitelisting, authentication, order logging, monitoring, and kill-switch capability. Brokers who don't comply cannot host retail API clients.

**This system:** Uses only SEBI-compliant brokers. As of January 5, 2026, only compliant brokers can onboard new retail API clients, so any broker that lets you sign up today is compliant by definition. We default to Upstox; Dhan and Zerodha Kite are also compliant.

## What changed in 2025–2026 that affects this system

1. **Personal-use exemption clarified.** Before 2025, the algo regulatory landscape was ambiguous for retail. The 2025 framework explicitly carves out white-box self-account low-OPS use, which is our exact case.
2. **Black-box providers now require RA registration.** Doesn't affect us (we're not a provider).
3. **Static IP requirement is now hard.** Pre-2025 retail traders could often run from dynamic-IP home connections; now they cannot. VPS is required for production.
4. **Algo-ID tagging is mandatory.** Pre-2025 it was best-practice; now it's enforced at the broker level.

## Things this system does NOT do (and why)

- **Doesn't bypass TOTP.** Bypass tools exist publicly but violate broker T&C and create non-trivial tail risk. Manual TOTP login is the legitimate posture.
- **Doesn't trade for anyone else.** Multi-user capability would push the operator into provider classification and require SEBI Research Analyst registration.
- **Doesn't run black-box logic.** White-box is the correct compliance posture for own-account use.
- **Doesn't exceed 10 OPS.** Configured cap at 2 OPS. The risk manager rejects bursts above this.
- **Doesn't trade F&O.** Out of V1 scope; F&O has its own regulatory regime (margin rules, lot sizes, expiries) that we don't address.

## What to do if the regulatory framework changes

- Re-read the latest SEBI circular.
- Update `docs/sebi_compliance.md`.
- Use the `/sebi-check` slash command to identify code paths that need adjusting.
- Spawn the `sebi-compliance-auditor` subagent to review the diff.
- Test in paper mode for ≥ 7 sessions before redeploying live.

## Useful primary sources

- [SEBI circular Feb 4, 2025](https://www.sebi.gov.in/legal/circulars/feb-2025/safer-participation-of-retail-investors-in-algorithmic-trading_91614.html)
- [SEBI timeline extension Sep 2025](https://www.sebi.gov.in/legal/circulars/sep-2025/extension-of-timeline-for-implementation-of-sebi-circular-dated-february-04-2025-on-safer-participation-of-retail-investors-in-algorithmic-trading-_96979.html)
- NSE algo circulars: nseindia.com/regulations/exchange-communication-circulars
- Broker terms: upstox.com/developer/api/terms (and equivalents for Dhan / Kite)
