from dataclasses import dataclass
from typing import Optional

from settings import CONFIDENCE_THRESHOLD, DEFECT_LABELS


@dataclass
class InterpretedResult:
    is_ok:       bool
    label:       str
    confidence:  float
    defect_type: Optional[str] = None  # set by the Gemini typer when a defect is found

    def __str__(self) -> str:
        if self.is_ok:
            return "OK"
        if self.defect_type:
            return f"{self.defect_type.replace('_', ' ').title()}  {self.confidence:.0%}"
        return f"Defect Detected  {self.confidence:.0%}"


def interpret(predictions: list) -> InterpretedResult:
    if not predictions:
        return InterpretedResult(is_ok=True, label="ok", confidence=1.0)

    top   = max(predictions, key=lambda p: float(getattr(p, "score", 0.0)))
    label = getattr(top, "label_name", None) or getattr(top, "label", None) or ""
    score = float(getattr(top, "score", 0.0))

    if label in DEFECT_LABELS and score >= CONFIDENCE_THRESHOLD:
        return InterpretedResult(is_ok=False, label=label, confidence=score)

    return InterpretedResult(is_ok=True, label=label, confidence=score)
