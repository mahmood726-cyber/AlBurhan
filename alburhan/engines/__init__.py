from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class EvidenceEngine(Protocol):
    """Contract for all Al-Burhan evidence engines (ENG-P1-1)."""
    name: str

    def evaluate(self, claim_data: Dict[str, Any]) -> Dict[str, Any]: ...
