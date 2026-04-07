"""
E156 Micro-Paper Emitter — compresses multi-engine audit into 7-sentence format.

All values are derived from upstream engines that compute from real data.
"""

import logging

logger = logging.getLogger(__name__)


class E156Emitter:
    name = "E156"

    def evaluate(self, claim_data):
        results = claim_data.get('audit_results', {})
        logger.info("%s: emitting from %d upstream engines", self.name, len(results))
        country = claim_data.get('country', 'the target region')
        condition = claim_data.get('condition', 'the condition')

        pg = results.get('PredictionGap', {}).get('metrics', {})
        mf = results.get('MetaFrontierLab', {})
        fa = results.get('FragilityAtlas', {})
        cs = results.get('CausalSynth', {})
        rf = results.get('RegistryForensics', {})
        nm = results.get('NetworkMeta', {})
        syn = results.get('SynthesisLoss', {})
        bayes = results.get('BayesianMA', {})
        pb = results.get('PubBias', {})
        grade = results.get('GRADE', {})

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
        s3 = (f"Reviewers applied random-effects meta-analysis, Bayesian hierarchical "
              f"modelling, publication-bias detection, and E-value causal sensitivity "
              f"in {country}.")
        # S4: Result — frequentist CI + Bayesian posterior mean and credible interval
        e_val = cs.get('e_value', 'N/A')
        if bayes.get('status') == 'evaluated':
            post_mean = bayes.get('posterior_mu', 'N/A')
            cri_lo = bayes.get('cri_lo', 'N/A')
            cri_hi = bayes.get('cri_hi', 'N/A')
            s4 = (f"The pooled effect was {theta:.2f} (95% CI {ci[0]:.2f} to {ci[1]:.2f}); "
                  f"Bayesian posterior mean {post_mean} (95% CrI {cri_lo} to {cri_hi}), "
                  f"E-value {e_val}.")
        else:
            e_val_ci = cs.get('e_value_ci', 'N/A')
            s4 = (f"The pooled effect was {theta:.2f} (95% CI {ci[0]:.2f} to {ci[1]:.2f}) "
                  f"with a point E-value of {e_val} and CI E-value of {e_val_ci}.")
        # S5: Robustness — multiverse classification + GRADE certainty
        classification = fa.get('classification', 'undetermined')
        certainty = grade.get('certainty', 'undetermined') if grade.get('status') == 'evaluated' else 'undetermined'
        s5 = (f"Multiverse analysis classified this as {classification}; "
              f"GRADE certainty of evidence was {certainty}.")
        # S6: Context — publication bias trim-fill count
        if pb.get('status') == 'evaluated':
            n_missing = pb.get('trim_fill', {}).get('n_missing', 0)
            pb_verdict = (f"trim-and-fill imputed {n_missing} missing "
                          f"{'study' if n_missing == 1 else 'studies'}")
        else:
            n_missing = 0
            pb_verdict = "publication bias assessment unavailable"
        anomaly_flags = rf.get('anomaly_flags', 0) if rf.get('status') == 'evaluated' else 0
        s6 = (f"Publication bias screening found {pb_verdict}; "
              f"registry forensics flagged {anomaly_flags} anomaly tests.")
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
