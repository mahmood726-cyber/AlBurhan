"""
E156 Micro-Paper Emitter — compresses multi-engine audit into 7-sentence format.

All values are derived from upstream engines that compute from real data.
"""


class E156Emitter:
    name = "E156"

    def evaluate(self, claim_data):
        results = claim_data.get('audit_results', {})
        country = claim_data.get('country', 'the target region')
        condition = claim_data.get('condition', 'the condition')

        pg = results.get('PredictionGap', {}).get('metrics', {})
        mf = results.get('MetaFrontierLab', {})
        fa = results.get('FragilityAtlas', {})
        cs = results.get('CausalSynth', {})
        rf = results.get('RegistryForensics', {})
        nm = results.get('NetworkMeta', {})
        syn = results.get('SynthesisLoss', {})

        theta = mf.get('estimate', pg.get('theta', 0.0))
        ci = mf.get('ci', [pg.get('ci_lo', 0.0), pg.get('ci_hi', 0.0)])
        k = pg.get('k', 0)

        # S1: Question
        s1 = (f"In patients with {condition}, does current clinical evidence "
              f"provide a robust basis for treatment compared with standard care?")
        # S2: Scope
        n_influential = nm.get('n_influential', 0) if nm.get('status') == 'evaluated' else 0
        info_loss = syn.get('information_loss_ratio', 0) if syn.get('status') == 'evaluated' else 0
        s2 = (f"A multi-engine audit of {k} trials identified "
              f"{n_influential} influential studies and an information loss ratio of "
              f"{info_loss:.2f}.")
        # S3: Method
        s3 = (f"Reviewers applied random-effects meta-analysis, REML-based multiverse "
              f"robustness testing, leave-one-out influence analysis, and E-value "
              f"causal sensitivity in {country}.")
        # S4: Result
        e_val = cs.get('e_value', 'N/A')
        e_val_ci = cs.get('e_value_ci', 'N/A')
        s4 = (f"The pooled effect was {theta:.2f} (95% CI {ci[0]:.2f} to {ci[1]:.2f}) "
              f"with a point E-value of {e_val} and CI E-value of {e_val_ci}.")
        # S5: Robustness
        classification = fa.get('classification', 'undetermined')
        max_resid = nm.get('max_studentized_residual', 'N/A') if nm.get('status') == 'evaluated' else 'N/A'
        s5 = (f"Multiverse analysis classified this as {classification}, with a maximum "
              f"studentized residual of {max_resid}.")
        # S6: Forensic context
        anomaly_flags = rf.get('anomaly_flags', 0) if rf.get('status') == 'evaluated' else 0
        total_tests = rf.get('total_tests', 0) if rf.get('status') == 'evaluated' else 0
        entropy = rf.get('scientific_entropy', 0) if rf.get('status') == 'evaluated' else 0
        s6 = (f"Registry forensics flagged {anomaly_flags}/{total_tests} anomaly tests "
              f"with a scientific entropy of {entropy}.")
        # S7: Boundary
        s7 = ("Interpretation is limited by the use of aggregate trial-level data and "
              "the scope of available pairwise comparisons.")

        body = " ".join([s1, s2, s3, s4, s5, s6, s7])
        word_count = len(body.split())

        return {
            "status": "emitted",
            "body": body,
            "word_count": word_count,
            "sentence_count": 7,
            "over_limit": word_count > 156,
        }
