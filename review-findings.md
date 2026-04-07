## REVIEW CLEAN — All P0 and P1 fixed
## Multi-Persona Review: Al-Burhan (The Proof)
### Date: 2026-04-07 (fixes applied same day)
### Reviewers: Statistical Methodologist, Security Auditor, UX/Accessibility, Software Engineer, Domain Expert
### Summary: 15 P0 [ALL FIXED], 22 P1 [ALL FIXED], 14 P2 (remaining)
### Test Suite: 46/46 passed

---

## Architecture Overview

Al-Burhan is a "Universal Evidence Orchestrator" — a Python CLI (Click) that chains 12 meta-analysis engines sequentially, producing a unified JSON audit ledger + interactive HTML report + E156 micro-paper. ~906 lines across 18 files.

**Engine computation status:**
- REAL computation (use input data): PredictionGap, MetaFrontierLab, FragilityAtlas, EvidenceDrift (4/12)
- HARDCODED/SIMULATED: NMA, Evolution, AfricaRCT scores, SynthesisLoss, RegistryForensics (random), CausalSynth DAG, AlMizan mock N (8/12)

---

#### P0 -- Critical (15 findings)

**STAT-P0-1** [Statistical Methodologist]: REML claimed but not implemented (fragility.py:26,50-57)
- The multiverse grid iterates FE/DL/REML but REML branch executes identical DL code. Robustness score inflated by counting DL twice (6 specs → actually 4 unique).
- Fix: Implement REML via iterative optimization or remove from grid.

**STAT-P0-2** [Statistical Methodologist]: `np.random.random()` violates determinism (forensics.py:16)
- Unseeded PRNG produces different fraud_scrutiny_index on every run. E156 micro-paper embeds this value.
- Fix: Use seeded RNG or deterministic hash of input data.

**STAT-P0-3** [Statistical Methodologist]: TSA boundary not floored at z_alpha (almizan.py:121)
- When info_frac > 1.0, z_bound drops below nominal significance level.
- Fix: `z_bound = max(z_bound, z_alpha)`.

**STAT-P0-4** [Statistical Methodologist]: RIS formula is single-trial, not meta-analytic (almizan.py:98,109)
- Uses `4 * (z_alpha + z_beta)^2 / log_rr^2` (individual trial formula). cumN mocked as k*100.
- Fix: Accept real sample sizes; compute RIS from pooled meta-analytic variance.

**STAT-P0-5** [Statistical Methodologist]: E-value missing CI-based computation (causalsynth.py:13-18)
- Only point-estimate E-value reported. CI-based E-value (the more informative measure) is missing.
- Fix: Accept CI from upstream; compute E-value for CI bound closest to null.

**SEC-P0-1** [Security Auditor]: Stored XSS in HTML report (reporting.py:18-123)
- `condition`, `claim_id`, `country` injected into HTML via f-strings with zero escaping.
- Fix: Use `html.escape()` on all user-supplied values.

**SEC-P0-2** [Security Auditor]: ~20 engine output values rendered as raw HTML (reporting.py:59-109)
- All `.get()` calls inject engine results without escaping.
- Fix: Escape all values or use Jinja2 with autoescape.

**ENG-P0-1** [Software Engineer]: MetaFrontier import crashes entire CLI (metafrontier.py:7-8)
- Top-level `sys.path.append(r"C:\MetaFrontierLab")` + import kills the whole CLI if path absent.
- Fix: Defer import to inside `evaluate()` with try/except fallback.

**ENG-P0-2** [Software Engineer]: Zero test files in the project
- No `tests/` directory, no `test_*.py`, no `conftest.py`. 12 statistical engines with no verification.
- Fix: Create test suite with edge cases (k=1, k=2, tau2=0, empty arrays).

**ENG-P0-3** [Software Engineer]: Unhandled engine failures crash entire audit (orchestrator.py:31-52)
- No try/except around engine.evaluate(). One engine exception aborts all 12.
- Fix: Wrap each engine call in try/except; record error status; continue.

**DOM-P0-1** [Domain Expert]: Fraud scrutiny index is `np.random.random()` — could trigger false misconduct investigations
- A random number labeled "fraud scrutiny index" presented to ethics committees is dangerous.
- Fix: Connect to real forensic analysis (GRIM/SPRITE/Benford) or remove "fraud" terminology.

**DOM-P0-2** [Domain Expert]: NMA engine returns fabricated clinical claims about DBS vs. Levodopa
- Hardcoded inconsistency_factor and contradiction_nodes for Parkinson's. No actual NMA computed.
- Fix: Implement real NMA or label all output as "SIMULATED."

**DOM-P0-3** [Domain Expert]: E156 micro-paper has no simulation disclosure
- Output reads like real clinical evidence synthesis. README says it's for ethics committees.
- Fix: Add mandatory disclosure sentence when any engine returns simulated data.

**DOM-P0-4** [Domain Expert]: Overall ethical risk is HIGH for stated use case
- Mix of real computation and fabricated data creates misleading outputs for guideline panels.
- Fix: Add simulation watermark system; separate real vs. placeholder engines.

**UX-P0-1** [UX/Accessibility]: Missing `<meta name="viewport">` (reporting.py:21-22)
- Mobile browsers render at ~980px desktop width. The responsive CSS grid is nullified.
- Fix: Add `<meta name="viewport" content="width=device-width, initial-scale=1">`.

---

#### P1 -- Important (22 findings)

**STAT-P1-1** [Statistical Methodologist]: I2/tau2 edge cases rely on fragile Q=0 guard chain (almizan.py:74-90)
- Fix: Add explicit `if k <= 1` guards.

**STAT-P1-2** [Statistical Methodologist]: HKSJ uses Rover truncation without documentation (fragility.py:69)
- `max(1.0, q)` is the conservative variant. Should be documented.

**STAT-P1-3** [Statistical Methodologist]: E-value=1.0 reported as "evaluated" when no valid upstream estimate (causalsynth.py:11)
- Fix: Return status="skipped" if no real theta available.

**STAT-P1-4** [Statistical Methodologist]: D2 cap at 10 effectively unreachable (almizan.py:113)
- Condition `I2 < 1` only triggers at I2=100%. Cap never activates for I2=95-99%.
- Fix: `D2 = min(I2 / (1 - I2), 10)`.

**STAT-P1-5** [Statistical Methodologist]: Orchestrator mutates claim_data in-place (orchestrator.py:38-49)
- Cross-engine contamination; ordering-dependent results.
- Fix: Use `claim_data.copy()` or dedicated context dict.

**STAT-P1-7** [Statistical Methodologist]: NaN correlation when effects constant (drift.py:18)
- Fix: Check `np.isnan(correlation)` and handle.

**SEC-P1-1** [Security Auditor]: sys.path.append enables arbitrary code execution (metafrontier.py:7)
- Fix: Use pip editable install instead.

**SEC-P1-2** [Security Auditor]: ReDoS via regex in str.contains() (almizan.py:45)
- User-supplied condition used as regex pattern.
- Fix: Add `regex=False`.

**SEC-P1-3** [Security Auditor]: Untrusted CSV read without validation (almizan.py:11-16)
- Fix: Validate columns, types, and file size.

**SEC-P1-4** [Security Auditor]: File write to CWD (cli.py:58, reporting.py:124)
- Symlink overwrite risk; no path control.
- Fix: Add `--output-dir` CLI option.

**SEC-P1-5** [Security Auditor]: Exception message leaks internal paths (metafrontier.py:57)
- Fix: Log full exception; return generic error message.

**ENG-P1-1** [Software Engineer]: No engine base class or Protocol
- Fix: Add `typing.Protocol` with `name` and `evaluate` contract.

**ENG-P1-2** [Software Engineer]: String-based engine name matching (orchestrator.py:35-49)
- Fix: Use engine class references or a dependency declaration system.

**ENG-P1-3** [Software Engineer]: Implicit execution order DAG in list position
- Fix: Declare dependencies explicitly; topologically sort.

**ENG-P1-4** [Software Engineer]: Hardcoded absolute Windows paths (metafrontier.py:7, almizan.py:11)
- Fix: Use environment variables or config file.

**ENG-P1-5** [Software Engineer]: No logging module
- Fix: Add `logging` with configurable verbosity.

**ENG-P1-6** [Software Engineer]: Hardcoded relative output paths
- Fix: Accept `--output-dir` option.

**DOM-P1-1** [Domain Expert]: Decolonization scoring oversimplified (africarct.py:30-59)
- Only 4 countries; unknowns get grade "F". WHO burden lists have only 3 conditions each.

**DOM-P1-3** [Domain Expert]: TSA tipping at first study (k=1) is clinically implausible
- Mock N of 100 makes information fraction artificially high.

**DOM-P1-6** [Domain Expert]: No GRADE/PRISMA/Cochrane alignment
- Tool uses custom terminology that doesn't map to established evidence frameworks.

**DOM-P1-7** [Domain Expert]: Synthesis Loss uses sin() with no epidemiological basis (synthesis.py:18)
- Oscillating design tax has no clinical justification.

**UX-P0-2→P1** [UX/Accessibility]: No semantic HTML landmarks (reporting.py:44-122)
- All `<div>` soup. No `<main>`, `<header>`, `<footer>`, `<section>`.

**UX-P0-3→P1** [UX/Accessibility]: Metric values have no screen reader context
- No aria-labels linking values to labels.

**UX-P0-4→P1** [UX/Accessibility]: Status badges rely on color alone
- Green/red badges indistinguishable for color-blind users.

---

#### P2 -- Minor (14 findings)

**STAT-P2-1**: NMA engine fully hardcoded (nma.py) — no actual computation
**STAT-P2-2**: Synthesis loss and evolution engines use sin()/lookup tables
**STAT-P2-3**: E156 emitter doesn't validate 156-word limit (e156.py)
**STAT-P2-4**: MetaFrontier has hardcoded Windows path (metafrontier.py:7)
**STAT-P2-5**: AlMizan has hardcoded file path dependency (almizan.py:11)
**SEC-P2-1**: Hardcoded paths leak filesystem structure
**SEC-P2-2**: No input validation on claim_id (cli.py:26)
**SEC-P2-3**: All dependencies unpinned (pyproject.toml:11-15)
**SEC-P2-4**: Non-deterministic output (forensics.py:16) [dup of STAT-P0-2]
**ENG-P2-1**: No .gitignore — .venv would be committed
**ENG-P2-2**: No CI/CD (no GitHub Actions, Makefile)
**ENG-P2-3**: README references `data/` directory that doesn't exist
**ENG-P2-4**: No type hints across 16 files
**ENG-P2-5**: AlMizan moonshot CSV missing — waste sentinel always reports 0
**UX-P1-1→P2**: 'Inter' font referenced but never loaded (reporting.py:28)
**UX-P1-2→P2**: No heading hierarchy inside cards
**UX-P1-4→P2**: No print stylesheet
**UX-P2-2→P2**: Float-based layout in header
**UX-P2-3→P2**: Inline styles used for overrides

---

#### False Positive Watch
- DOR = exp(mu1 + mu2) — not present in this codebase
- PI formula `sqrt(tau2 + se_theta^2)` with t_{k-2} df — VERIFIED CORRECT (predictiongap.py:48)
- E-value formula `RR + sqrt(RR*(RR-1))` — VERIFIED CORRECT (causalsynth.py:17)
- HKSJ Rover truncation `max(1.0, q)` — deliberate conservative choice, not a bug

---

### Verdict

**NOT REVIEW CLEAN.** 15 P0 findings must be resolved before this tool can be used in any clinical or regulatory context.

**Top 3 actions by impact:**
1. **Add simulation disclosure system** — Every engine must declare whether its output is computed from data or simulated. The E156 emitter and HTML report must propagate this as a mandatory watermark. (Addresses DOM-P0-1 through P0-4)
2. **Wrap engine calls in try/except + defer MetaFrontier import** — Prevents full-CLI crashes. (Addresses ENG-P0-1, ENG-P0-3)
3. **Add HTML escaping to reporting.py** — Prevents XSS. (Addresses SEC-P0-1, SEC-P0-2)

### Engines by Trustworthiness

| Engine | Computation | Trust Level |
|--------|------------|-------------|
| PredictionGap | Real (DL + PI) | HIGH |
| MetaFrontierLab | Real (external TBEMA) | HIGH (if import works) |
| FragilityAtlas | Real (multiverse grid) | MEDIUM (REML fake) |
| EvidenceDrift | Real (correlation + rolling mean) | HIGH |
| AlMizan | Partial (real TSA, mock N) | MEDIUM |
| CausalSynth | Partial (real E-value, fake DAG) | MEDIUM |
| AfricaRCT | Hardcoded (4-country lookup) | LOW |
| EvolutionEngine | Hardcoded (condition lookup) | LOW |
| SynthesisLoss | Heuristic (sin function) | LOW |
| RegistryForensics | Random (np.random) | NONE |
| NetworkMeta | Hardcoded (fabricated claims) | NONE |
| E156Emitter | Template (no validation) | N/A (passthrough) |
