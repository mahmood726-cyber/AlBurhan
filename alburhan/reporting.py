import os
from datetime import datetime
from html import escape


def generate_html_report(results, claim_id, country, condition, output_path=None):
    """Generate an accessible HTML report with proper escaping."""

    # Escape all user-supplied and engine-derived strings (SEC-P0-1, SEC-P0-2)
    s_condition = escape(str(condition))
    s_claim_id = escape(str(claim_id))
    s_country = escape(str(country))

    pg = results.get('PredictionGap', {}).get('metrics', {})
    am = results.get('Al-Mizan', {})
    mf = results.get('MetaFrontierLab', {})
    fa = results.get('FragilityAtlas', {})
    evo = results.get('TreatmentEvolution', {})
    syn = results.get('SynthesisLoss', {})
    cs = results.get('CausalSynth', {})
    dr = results.get('EvidenceDrift', {})
    rf = results.get('RegistryForensics', {})
    nm = results.get('NetworkMeta', {})
    ar = results.get('AfricaRCT', {})
    bayes = results.get('BayesianMA', {})
    pb = results.get('PubBias', {})
    grade = results.get('GRADE', {})
    e156_data = results.get('E156', {})
    e156_body = escape(str(e156_data.get('body', 'N/A')))

    # Helper to safely escape any value
    def sv(val, fallback='N/A'):
        if val is None:
            return escape(str(fallback))
        return escape(str(val))

    # Determine badge class and icon (UX-P0-4: non-color differentiation)
    fa_class = fa.get('classification', 'N/A')
    if fa_class == 'Robust':
        badge_css = 'badge-robust'
        badge_icon = '&#x2713; '  # checkmark
    else:
        badge_css = 'badge-fragile'
        badge_icon = '&#x26A0; '  # warning triangle

    rf_status = rf.get('forensic_status', 'N/A')
    rf_badge_icon = '&#x26A0; ' if rf_status != 'Nominal' else '&#x2713; '

    # GRADE badge CSS
    grade_certainty = grade.get('certainty', 'N/A') if grade.get('status') == 'evaluated' else 'N/A'
    _grade_css_map = {
        'HIGH': 'badge-grade-high',
        'MODERATE': 'badge-grade-moderate',
        'LOW': 'badge-grade-low',
        'VERY LOW': 'badge-grade-verylow',
    }
    grade_badge_css = _grade_css_map.get(grade_certainty, 'badge-grade-verylow')
    grade_domains = grade.get('domains', {}) if grade.get('status') == 'evaluated' else {}
    grade_total_dg = grade.get('total_downgrade', 'N/A') if grade.get('status') == 'evaluated' else 'N/A'
    _gd_rob = (grade_domains.get('risk_of_bias') or {}).get('downgrade', 'N/A')
    _gd_inc = (grade_domains.get('inconsistency') or {}).get('downgrade', 'N/A')
    _gd_ind = (grade_domains.get('indirectness') or {}).get('downgrade', 'N/A')
    _gd_imp = (grade_domains.get('imprecision') or {}).get('downgrade', 'N/A')
    _gd_pub = (grade_domains.get('publication_bias') or {}).get('downgrade', 'N/A')

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Al-Burhan Master Evidence Proof: {s_condition}</title>
    <style>
        :root {{
            --bg: #05080a; --panel: #0d1117; --accent: #58a6ff; --text: #c9d1d9;
            --danger: #f85149; --success: #3fb950; --warning: #d29922;
        }}
        * {{ box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); padding: 40px 20px; line-height: 1.6; margin: 0; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        header {{ border-bottom: 1px solid #30363d; padding-bottom: 20px; margin-bottom: 30px; display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px; }}
        header .meta {{ text-align: right; color: #8b949e; flex-shrink: 0; }}
        h1 {{ color: var(--accent); margin: 0; font-size: 2.2rem; }}
        h2 {{ color: var(--accent); }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }}
        .card {{ background: var(--panel); border: 1px solid #30363d; border-radius: 12px; padding: 24px; }}
        .card h3 {{ font-size: 0.8rem; text-transform: uppercase; color: #8b949e; letter-spacing: 1px; margin: 0 0 16px 0; }}
        .metric-box {{ text-align: center; margin-bottom: 20px; }}
        .metric-val {{ font-size: 2.2rem; font-weight: 800; color: #fff; }}
        .metric-sublabel {{ font-size: 0.8rem; text-transform: uppercase; color: #8b949e; letter-spacing: 1px; }}
        .detail {{ font-size: 0.9rem; }}
        .status-badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; }}
        .badge-robust {{ background: rgba(63, 185, 80, 0.15); color: var(--success); border: 1px solid var(--success); }}
        .badge-fragile {{ background: rgba(248, 81, 73, 0.15); color: var(--danger); border: 1px solid var(--danger); }}
        .e156-box {{ background: #161b22; border-left: 4px solid var(--accent); padding: 20px; font-style: italic; font-size: 1.05rem; margin-top: 20px; border-radius: 0 12px 12px 0; }}


        .badge-grade-high {{ background: rgba(63, 185, 80, 0.15); color: var(--success); border: 1px solid var(--success); }}
        .badge-grade-moderate {{ background: rgba(210, 153, 34, 0.15); color: var(--warning); border: 1px solid var(--warning); }}
        .badge-grade-low {{ background: rgba(248, 152, 29, 0.15); color: #f8981d; border: 1px solid #f8981d; }}
        .badge-grade-verylow {{ background: rgba(248, 81, 73, 0.15); color: var(--danger); border: 1px solid var(--danger); }}
        footer {{ text-align: center; margin-top: 50px; color: #8b949e; font-size: 0.8rem; border-top: 1px solid #30363d; padding-top: 20px; }}
        a {{ color: var(--accent); }}
        .sr-only {{ position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }}

        @media print {{
            body {{ background: #fff; color: #000; padding: 20px; }}
            .card {{ border-color: #ccc; background: #f9f9f9; }}
            .metric-val {{ color: #000; }}
            h1, h2 {{ color: #003366; }}


            .badge-robust {{ border-color: #2e7d32; color: #2e7d32; background: #e8f5e9; }}
            .badge-fragile {{ border-color: #c62828; color: #c62828; background: #ffebee; }}
            .e156-box {{ background: #f5f5f5; border-left-color: #003366; }}
            footer {{ color: #666; }}
        }}
    </style>
</head>
<body>
    <a href="#main-content" class="sr-only">Skip to content</a>
    <div class="container">
        <header>
            <div>
                <h1>The Proof: {s_condition}</h1>
                <p>Evidence Audit for <strong>{s_claim_id}</strong> | Target Region: <strong>{s_country}</strong></p>
            </div>
            <div class="meta">
                AL-BURHAN ORCHESTRATOR v1.1<br>
                <time datetime="{datetime.now().isoformat()}">{timestamp}</time>
            </div>
        </header>

        <main id="main-content">
        <section aria-label="Evidence metrics">
        <div class="grid">
            <!-- TRUTH CARD -->
            <article class="card">
                <h3>TBEMA Ensemble Estimate</h3>
                <div class="metric-box">
                    <div class="metric-val" aria-label="TBEMA Estimate: {sv(mf.get('estimate'))}">{sv(mf.get('estimate'))}</div>
                    <div class="metric-sublabel">Exact Sparse Log-OR</div>
                </div>
                <div class="detail">
                    <strong>Multiverse Robustness:</strong> {sv(fa.get('robustness_score'))}%
                    <span class="status-badge {badge_css}">{badge_icon}{sv(fa_class)}</span><br>
                    <strong>Causal E-Value:</strong> {sv(cs.get('e_value'))} ({sv(cs.get('confounding_resistance'))})
                </div>
            </article>

            <!-- FORENSIC CARD -->
            <article class="card">
                <h3>Registry Anomaly Profile</h3>
                <div class="metric-box">
                    <div class="metric-val" aria-label="Registry Anomaly Index: {sv(rf.get('registry_anomaly_index'))}">{sv(rf.get('registry_anomaly_index'))}</div>
                    <div class="metric-sublabel">Registry Anomaly Index</div>
                </div>
                <div class="detail">
                    <strong>Hazard Rate:</strong> {sv(rf.get('instantaneous_hazard_rate'))}<br>
                    <strong>Network Inconsistency:</strong> {sv(nm.get('inconsistency_factor'))}<br>
                    <strong>Registry Status:</strong> <span class="status-badge badge-fragile">{rf_badge_icon}{sv(rf_status)}</span>
                </div>
            </article>

            <!-- PIPELINE CARD -->
            <article class="card">
                <h3>Synthesis Efficiency</h3>
                <div class="metric-box">
                    <div class="metric-val" aria-label="Information Loss Ratio: {sv(syn.get('information_loss_ratio'))}">{sv(syn.get('information_loss_ratio'))}</div>
                    <div class="metric-sublabel">Information Loss Ratio</div>
                </div>
                <div class="detail">
                    <strong>Design Effect:</strong> {sv(syn.get('design_effect'))}<br>
                    <strong>Effective N:</strong> {sv(syn.get('effective_n'))}<br>
                    <strong>Scientific Entropy:</strong> {sv(rf.get('scientific_entropy'))}
                </div>
            </article>

            <!-- WASTE CARD -->
            <article class="card">
                <h3>Evidence Waste (Post-Tipping)</h3>
                <div class="metric-box">
                    <div class="metric-val" style="color:var(--danger)" aria-label="Active Trials Randomizing: {sv(am.get('waste_momentum', 0))}">{sv(am.get('waste_momentum', 0))}</div>
                    <div class="metric-sublabel">Active Trials Randomizing</div>
                </div>
                <div class="detail">
                    <strong>Tipping Point:</strong> {sv(am.get('tipping_year'))}<br>
                    <strong>GERI Relevance:</strong> {sv(ar.get('relevance_index'))}<br>
                    <strong>Burden Alignment:</strong> {sv((ar.get('burden_alignment') or {}).get('reason', 'N/A'))}
                </div>
            </article>

            <!-- BAYESIAN EVIDENCE CARD -->
            <article class="card">
                <h3>Bayesian Evidence</h3>
                <div class="metric-box">
                    <div class="metric-val" aria-label="Bayesian Posterior Mean: {sv(bayes.get('posterior_mu'))}">{sv(bayes.get('posterior_mu'))}</div>
                    <div class="metric-sublabel">Posterior Mean</div>
                </div>
                <div class="detail">
                    <strong>95% CrI:</strong> {sv(bayes.get('cri_lo'))} to {sv(bayes.get('cri_hi'))}<br>
                    <strong>Bayes Factor (BF&#x2081;&#x2080;):</strong> {sv(bayes.get('bf10'))}<br>
                    <strong>Evidence:</strong> {sv(bayes.get('evidence_label'))}
                </div>
            </article>

            <!-- PUBLICATION BIAS CARD -->
            <article class="card">
                <h3>Publication Bias</h3>
                <div class="metric-box">
                    <div class="metric-val" aria-label="Trim-Fill Missing Studies: {sv((pb.get('trim_fill') or {}).get('n_missing'))}">{sv((pb.get('trim_fill') or {}).get('n_missing'))}</div>
                    <div class="metric-sublabel">Trim-Fill Missing Studies</div>
                </div>
                <div class="detail">
                    <strong>Egger p-value:</strong> {sv((pb.get('egger') or {}).get('p_value'))}<br>
                    <strong>Fail-safe N:</strong> {sv((pb.get('failsafe_n') or {}).get('failsafe_n'))}<br>
                    <strong>P-curve verdict:</strong> {sv((pb.get('p_curve') or {}).get('skew_direction'))}
                </div>
            </article>

            <!-- GRADE CERTAINTY CARD -->
            <article class="card">
                <h3>GRADE Certainty</h3>
                <div class="metric-box">
                    <div class="metric-val" aria-label="GRADE Certainty: {sv(grade_certainty)}">
                        <span class="status-badge {grade_badge_css}">{sv(grade_certainty)}</span>
                    </div>
                    <div class="metric-sublabel">Overall Certainty of Evidence</div>
                </div>
                <div class="detail">
                    <strong>Risk of Bias:</strong> {sv(_gd_rob)}<br>
                    <strong>Inconsistency:</strong> {sv(_gd_inc)}<br>
                    <strong>Indirectness:</strong> {sv(_gd_ind)}<br>
                    <strong>Imprecision:</strong> {sv(_gd_imp)}<br>
                    <strong>Publication Bias:</strong> {sv(_gd_pub)}<br>
                    <strong>Total Downgrade:</strong> {sv(grade_total_dg)}
                </div>
            </article>
        </div>
        </section>

        <section aria-label="E156 micro-paper">
        <h2>E156 Micro-Paper Summary</h2>
        <blockquote class="e156-box" cite="Al-Burhan Evidence Orchestrator">
            {e156_body}
        </blockquote>
        </section>
        </main>

        <footer>
            Generated by Project Al-Burhan: The Universal Evidence Orchestrator.<br>
            Audited via meta-analysis ensembling, multiverse robustness, and causal sensitivity analysis.
        </footer>
    </div>
</body>
</html>'''

    if output_path is None:
        output_path = "AL_BURHAN_MASTER_REPORT.html"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Master Report generated: {output_path}")
