import logging
import os
import re
import json
import numpy as np
import click
from pathlib import Path

from alburhan.core.orchestrator import EvidenceOrchestrator
from alburhan.reporting import generate_html_report

# Configure logging (ENG-P1-5)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class AlBurhanEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


@click.group()
def main():
    """Al-Burhan: Universal Evidence Orchestrator"""
    pass


@main.command()
@click.argument('claim_id')
@click.option('--country', default='South Africa')
@click.option('--condition', default='Parkinson Disease')
@click.option('--html', is_flag=True, help='Generate interactive HTML report')
@click.option('--output-dir', default=None, type=click.Path(),
              help='Output directory (default: current directory)')
@click.option('--verbose', '-v', is_flag=True, help='Enable debug logging')
def audit(claim_id, country, condition, html, output_dir, verbose):
    """Run ULTIMATE Evidence Audit."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Input validation (SEC-P2-2)
    if not re.match(r'^[a-zA-Z0-9_-]+$', claim_id):
        click.echo("Error: claim_id must be alphanumeric (hyphens/underscores allowed)")
        return

    # Resolve output directory (SEC-P1-4)
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path = Path('.')

    click.echo(f"Executing Master Audit: {claim_id}")

    mock_claims = {
        "parkinsons_early_signal": {
            "yi": [0.6, 0.5, 0.4, 0.45, 0.55],
            "sei": [0.1, 0.15, 0.1, 0.12, 0.1],
            "years": [2010, 2012, 2015, 2018, 2020],
            "treat_events": [30, 25, 28, 22, 35],
            "treat_total": [100, 100, 100, 100, 100],
            "control_events": [15, 12, 14, 10, 18],
            "control_total": [100, 100, 100, 100, 100],
            "n_per_study": [200, 200, 200, 200, 200]
        }
    }

    claim_data = mock_claims.get(claim_id, {
        "yi": [0.1, 0.2, 0.15], "sei": [0.1, 0.1, 0.1], "years": [2020, 2021, 2022]
    })
    claim_data['country'] = country
    claim_data['condition'] = condition

    orchestrator = EvidenceOrchestrator()
    results = orchestrator.run_audit(claim_data)

    click.echo(f"  - TBEMA Ensemble: {results.get('MetaFrontierLab', {}).get('estimate', 'N/A')}")
    click.echo(f"  - Robustness: {results.get('FragilityAtlas', {}).get('robustness_score', 'N/A')}%")
    click.echo(f"  - Waste detected: {results.get('Al-Mizan', {}).get('waste_momentum', 0)} active trials")

    if html:
        html_path = out_path / f"AL_BURHAN_{claim_id}.html"
        generate_html_report(results, claim_id, country, condition, str(html_path))

    ledger_path = out_path / f"audit_ledger_{claim_id}.json"
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=AlBurhanEncoder)
    click.echo(f"  Ledger saved: {ledger_path}")


@main.command()
@click.argument('condition')
@click.option('--intervention', '-i', default=None, help='Intervention filter')
@click.option('--source', type=click.Choice(['live', 'aact', 'both']), default='live')
@click.option('--max-trials', default=20, type=int)
@click.option('--run-audit', is_flag=True, help='Automatically run full audit on results')
@click.option('--html', is_flag=True, help='Generate HTML report (requires --run-audit)')
@click.option('--output-dir', default=None, type=click.Path())
@click.option('--verbose', '-v', is_flag=True)
def ingest(condition, intervention, source, max_trials, run_audit, html, output_dir, verbose):
    """Ingest real trial data from CT.gov and optionally run audit."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    out_path = Path(output_dir) if output_dir else Path('.')
    out_path.mkdir(parents=True, exist_ok=True)

    claim_data = None

    if source in ('live', 'both'):
        from alburhan.ingest.ctgov import CTGovClient
        click.echo(f"Searching CT.gov for: {condition}...")
        client = CTGovClient()
        claim_data = client.build_claim_data(condition, intervention, max_trials)
        if claim_data.get("status") in ("empty", "error"):
            click.echo(f"  Live: {claim_data.get('message', 'No results')}")
            claim_data = None
        else:
            click.echo(f"  Live: Found {len(claim_data['yi'])} trials with extractable results")

    if source in ('aact', 'both') and claim_data is None:
        from alburhan.ingest.aact import AACTClient
        click.echo(f"Searching local AACT for: {condition}...")
        client = AACTClient()
        claim_data = client.build_claim_data(condition, intervention, max_trials)
        if claim_data.get("status") in ("empty", "error"):
            click.echo(f"  AACT: {claim_data.get('message', 'No results')}")
            claim_data = None
        else:
            click.echo(f"  AACT: Found {len(claim_data['yi'])} trials with extractable results")

    if claim_data is None:
        click.echo("No trial data found. Try different search terms.")
        return

    # Save raw ingested data
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', condition).strip('_').lower()
    raw_path = out_path / f"ingest_{slug}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(claim_data, f, indent=2, cls=AlBurhanEncoder)
    click.echo(f"  Raw data saved: {raw_path}")

    if run_audit:
        click.echo(f"Running full {len(claim_data['yi'])}-study audit...")
        orchestrator = EvidenceOrchestrator()
        results = orchestrator.run_audit(claim_data)

        pg = results.get('PredictionGap', {}).get('metrics', {})
        grade = results.get('GRADE', {})
        click.echo(f"  Pooled effect: {pg.get('theta', 'N/A')}")
        click.echo(f"  GRADE certainty: {grade.get('certainty', 'N/A')}")

        if html:
            html_path = out_path / f"AL_BURHAN_{slug}.html"
            generate_html_report(results, slug, "Global", condition, str(html_path))

        ledger_path = out_path / f"audit_{slug}.json"
        with open(ledger_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, cls=AlBurhanEncoder)
        click.echo(f"  Audit saved: {ledger_path}")


if __name__ == '__main__':
    main()
