"""Zero-shot defect-type naming via the Mistral API (Pixtral vision model).

This does NOT replace LandingLens — it runs only after LandingLens has already
flagged a defect, to name *which kind* of defect it is. It is zero-shot: the set
of candidate types lives in settings.DEFECT_TYPES and can be edited freely with
no retraining.
"""

import base64
import io

from PIL import Image
from loguru import logger

try:
    # SDK >= 2.x moved the client under mistralai.client; older 1.x exposes it
    # at the top level. Support both.
    try:
        from mistralai import Mistral
    except ImportError:
        from mistralai.client import Mistral
except ImportError:  # SDK not installed at all
    Mistral = None


class DefectTyper:
    def __init__(self, api_key: str, model_name: str, defect_types: list[str]) -> None:
        self._types = defect_types
        self._model_name = model_name
        self._enabled = bool(api_key) and Mistral is not None

        if not api_key:
            logger.warning("MISTRAL_API_KEY not set — defect typing disabled.")
            self._client = None
            return
        if Mistral is None:
            logger.warning("mistralai not installed — defect typing disabled. "
                           "Run: pip install mistralai")
            self._client = None
            return

        self._client = Mistral(api_key=api_key)
        logger.info(f"Mistral defect typer ready (model: {model_name})")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _build_prompt(self) -> str:
        options = ", ".join(self._types)
        return (
            "You are a textile quality-control inspector. The fabric in this image "
            "has already been flagged as defective. Identify the single most likely "
            "type of defect.\n\n"
            f"Respond with EXACTLY ONE of these labels and nothing else: {options}.\n"
            "Do not add any explanation, punctuation, or extra words."
        )

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """PIL image -> base64 JPEG data URI for Mistral's image_url content."""
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _normalize(self, text: str) -> str | None:
        """Map Mistral's free-text reply onto one of the known types."""
        cleaned = text.strip().lower().replace(" ", "_").strip(".,!\"' ")
        if cleaned in self._types:
            return cleaned
        # Fallback: substring match (handles e.g. "the defect is a hole").
        for t in self._types:
            if t in cleaned:
                return t
        return None

    def classify(self, image: Image.Image) -> str | None:
        """Return one of DEFECT_TYPES, or None if disabled/uncertain/failed."""
        if not self._enabled or self._client is None:
            return None
        try:
            response = self._client.chat.complete(
                model=self._model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._build_prompt()},
                            {"type": "image_url", "image_url": self._encode_image(image)},
                        ],
                    }
                ],
                temperature=0.0,
            )
            raw = (response.choices[0].message.content or "").strip()
            logger.debug(f"Mistral defect-type raw reply: {raw!r}")
            defect_type = self._normalize(raw)
            if defect_type is None:
                logger.warning(f"Mistral reply did not match any known type: {raw!r}")
            return defect_type
        except Exception as exc:
            logger.error(f"Mistral defect typing failed: {exc}")
            return None
