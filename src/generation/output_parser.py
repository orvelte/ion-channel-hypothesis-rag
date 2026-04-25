"""
Structured output parser — Stage 6.

The LLM is instructed to return JSON matching HypothesisOutput. A strict Pydantic
parse is attempted first; if it fails, a regex-based JSON extraction is tried on
the raw string. Soft fallback rather than silent drop: the parse failure is recorded
in the evaluation artifact so Check 6 can accurately measure parse success rate.
Two-stage parsing catches the common case where the model wraps JSON in a markdown
code block.
"""

import json
import logging
import re
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class HypothesisOutput(BaseModel):
    hypothesis: str                    # the core mechanistic claim
    supporting_chunk_ids: list[str]    # chunk IDs cited in the hypothesis
    confidence: str                    # "high" | "medium" | "low"
    data_gaps: list[str]               # where context was insufficient
    mechanism_type: Optional[str] = None  # e.g. "channel block", "gating shift"


def parse(raw_response: str) -> tuple[Optional[HypothesisOutput], Optional[str]]:
    """
    Parse LLM response into HypothesisOutput.

    Returns (output, None) on success, or (None, error_reason) on failure.
    Never raises — parse failures are captured for Check 6 evaluation.
    """
    candidates: list[str] = []

    # Stage 1: try the raw string as-is
    candidates.append(raw_response.strip())

    # Stage 2: extract from ```json ... ``` code fence
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_response, re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1))

    # Stage 3: grab outermost {...} block (handles prose-wrapped JSON)
    brace_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            output = HypothesisOutput.model_validate(data)
            return output, None
        except Exception:
            continue

    return None, f"Could not parse JSON from response (len={len(raw_response)})"
