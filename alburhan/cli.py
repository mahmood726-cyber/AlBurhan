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


if __name__ == '__main__':
    main()
