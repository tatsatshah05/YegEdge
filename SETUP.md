# YegEdge — Setup Guide

A complete checklist for getting YegEdge from "fresh folder" to "Claude Code is happily building."

## 1. Prerequisites (one-time, ~30 minutes)

### 1.1 Install Claude Code

If you haven't already:

```bash
# macOS / Linux
curl -fsSL https://claude.ai/install.sh | bash

# Or via npm
npm install -g @anthropic-ai/claude-code
```

Verify: `claude --version`

### 1.2 Authenticate Claude Code with your Max subscription

```bash
claude
```

On first run it'll open a browser; log in with your Anthropic account (the one with Claude Max). Approve the OAuth grant. Done — Claude Code will now use your Max subscription for everything you do *inside Claude Code*.

### 1.3 Install Python 3.11+

```bash
# Check
python3 --version

# If you need to install on macOS:
brew install python@3.11

# On Linux (Ubuntu/Debian):
sudo apt install python3.11 python3.11-venv
```

### 1.4 Get an Anthropic API key (separate from Max — for the agent runtime)

- Go to https://console.anthropic.com/
- Settings → API Keys → Create Key → name it `trading-agent`
- Copy the key starting with `sk-ant-...`
- Add ~$10–$20 of credit (Billing → Add credit). This is months of runtime for the agent.

**Important:** This key is for the *trading agent's runtime* (Python script calling Claude). Your Max subscription covers Claude Code (where you build the code). They are intentionally separate.

### 1.5 Open an Upstox account (if you don't have one)

- Sign up at https://upstox.com/
- Complete KYC (Aadhaar + PAN + bank).
- Once approved (~24 hours), enable the Developer API: Profile → API → Activate.
- Create an app: Developer Portal → Apps → Create App.
- Note down `api_key` and `api_secret`.
- Whitelist your static IP (we'll get a VPS later; for development you can whitelist your home IP temporarily).

**Optional alternatives:** Dhan API (algo-focused, free) or Zerodha Kite Connect (₹2,000/month, best docs). The codebase has adapters for all three; default is Upstox.

### 1.6 (Optional now, required for live) Get a VPS

- DigitalOcean Bangalore region — ~₹500/month, 1 vCPU, 1GB RAM is enough.
- Hetzner Frankfurt — ~₹600/month, better hardware, slightly higher latency to NSE.
- AWS Mumbai — ~₹1,500/month, lowest latency, most setup.

Note the static IP and whitelist it with your broker.

## 2. Project setup (~5 minutes)

### 2.1 Move/copy this folder where you want it

This folder (`trading-agent/`) is the entire codebase scaffold. Move it anywhere — `~/code/trading-agent/`, `~/Documents/projects/trading-agent/`, whatever. Then:

```bash
cd /path/to/trading-agent
```

### 2.2 Initialize git

```bash
git init
git add .
git commit -m "Initial commit: project scaffold + CLAUDE.md"
```

(Optional: push to a private GitHub repo for backup.)

### 2.3 Create the .env file

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```
ANTHROPIC_API_KEY=sk-ant-...                # from console.anthropic.com
UPSTOX_API_KEY=...                          # from Upstox developer portal
UPSTOX_API_SECRET=...
UPSTOX_REDIRECT_URI=http://localhost:3000   # default; change if needed
TELEGRAM_BOT_TOKEN=...                      # optional, for alerts
TELEGRAM_CHAT_ID=...                        # optional
```

### 2.4 Create a virtual environment and install dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### 2.5 Verify the structure

```bash
tree -L 3 -I '__pycache__|.venv|.git'
```

You should see:

```
.
├── CLAUDE.md
├── README.md
├── SETUP.md
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
├── pyproject.toml
├── .claude/
│   ├── commands/
│   └── agents/
├── config/
│   ├── risk_rules.yaml
│   ├── universe.yaml
│   └── strategies.yaml
├── docs/
│   ├── architecture.md
│   ├── build_handoff.md
│   ├── live_readiness.md
│   └── sebi_compliance.md
└── tests/
```

## 3. Open in Claude Code (the moment of truth)

```bash
cd /path/to/trading-agent
claude
```

On first launch in this directory, Claude Code will:
- Read `CLAUDE.md` automatically — that's the project's contract with you and with itself.
- Detect the `.claude/` directory and load the custom slash commands and subagents.
- Show available MCP servers (none configured yet — see below if you want to add any).

**First prompt to send Claude Code:**

```
Read CLAUDE.md and docs/build_handoff.md. Confirm you understand the project rules, current build phase, and binding constraints. Then propose a plan for Phase 1 (Data Pipeline) using plan mode. Do not write code yet.
```

That gets the session aligned and gives you a chance to review the plan before any code is written.

## 4. Useful Claude Code commands and shortcuts

Inside a Claude Code session:

- `/help` — list all available slash commands.
- `/plan` — enter plan mode (no edits until you approve).
- `/clear` — clear conversation history (good when switching topics).
- `/compact` — compact the conversation to save context.
- `/cost` — show your Max plan usage so far.
- `/morning-brief` — custom command we defined; runs a morning market brief.
- `/risk-audit` — custom command; walks the risk rules.
- `/paper-status` — custom command; summarizes paper trading status.
- `/sebi-check` — custom command; flags potential SEBI regressions.

Subagents (spawned via `Task` tool inside a session):

- `code-reviewer` — mandatory before merging changes to `agent/risk/` or `agent/execution/`.
- `backtest-validator` — read a backtest result and flag overfit/lookahead/unrealistic-fill signs.
- `sebi-compliance-auditor` — check changes against SEBI April 2026 framework.

## 5. MCP connectors (optional, recommended)

Claude Code's built-in tools (file edit, bash, web search) cover most needs. Add MCP servers only when they add real workflow value. For this project:

### 5.1 GitHub MCP (recommended)

If you'll be using a GitHub repo:

```bash
claude mcp add github -- npx -y @modelcontextprotocol/server-github
```

You'll be prompted for a GitHub personal access token. Useful for PR review, issue management, browsing related repos.

### 5.2 Filesystem MCP (rarely needed)

Built-in file tools usually suffice. Add only if you want Claude Code to access folders outside the project root.

### 5.3 PostgreSQL/SQLite MCP (later, when journal database exists)

Once Phase 1 produces a SQLite journal:

```bash
claude mcp add sqlite -- npx -y @modelcontextprotocol/server-sqlite --db-path ./data/journal.db
```

This lets Claude Code directly query the trade journal in natural language during reviews.

### 5.4 Brave Search MCP (already covered)

Claude Code has a built-in `WebSearch` tool — you don't need a separate search MCP.

### 5.5 What we do NOT need MCP for

- **Upstox / Dhan / Kite broker APIs** — there are no official MCP servers for these and we don't need them. The agent calls broker APIs directly via Python in the runtime; Claude Code's bash tool is enough for development.
- **Anthropic API** — the runtime calls it via the SDK; Claude Code uses its own auth.
- **Telegram** — outbound alerts only; the agent sends them via Python.

## 6. First-week workflow

### Day 1
- Get all of section 1 and 2 done.
- Open in Claude Code. First prompt as above.
- Review the Phase 1 plan it proposes. Iterate until you're happy with it.
- Approve. Let it build the Upstox adapter, cache, and validators.

### Day 2–3
- Run `python -m agent.cli refresh` for the first time.
- Eyeball the data. Pick three random dates, three random symbols, and confirm the bars match what NSE/TradingView shows.
- Have Claude Code write tests for the data layer. Run them.

### Day 4–5
- Build the feature library (indicators + regime detector).
- Unit-test every indicator against a known reference (e.g., compare your RSI output for RELIANCE.NS on a specific date against TradingView's).

### Day 6–7
- Build the first strategy (trend-following).
- Run it as a *dry run* — no signals stored, no orders — just to see what it produces on the last 30 days.

### Week 2+
- Backtester. This is the longest phase. Walk-forward, full Indian cost model, Monte Carlo. Cleanest gate of the entire project.

### Phase gates
- Don't proceed to the next phase until the current one's deliverables are tested and committed.
- Use the `code-reviewer` subagent before merging anything to `agent/risk/` or `agent/execution/`.

## 7. When you get stuck

- Ask Claude Code: "Read CLAUDE.md and `docs/architecture.md`. I'm stuck on X. Walk me through it."
- For SEBI questions: use `/sebi-check` or read `docs/sebi_compliance.md`.
- For "is this overfitting?": spawn the `backtest-validator` subagent.
- For "is my risk rule actually firing in test?": use `/risk-audit`.
- For broader strategy/architecture questions: come back to Cowork (it has the research document loaded and is better for back-and-forth).

## 8. Things to remember

- **You do TOTP login each morning** before the agent runs. ~30 seconds.
- **Live trading flag stays `False`** until you've done 60 paper sessions and passed the `docs/live_readiness.md` checklist.
- **The agent journals everything.** Treat the journal as the system's memory; review it weekly.
- **The dashboard is the morning check.** If anything is red, *don't* trade that day.
- **Talk to a CA** before going live. Speculative-income tax treatment is real.

That's the setup. Once Phase 1 is built and you have a clean data pipeline, the rest is iteration.
