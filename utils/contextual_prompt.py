"""Dynamic contextual prompt helpers for exogenous variables."""
from __future__ import annotations

from typing import List, Mapping, Optional, Sequence, Union

import numpy as np


ContextType = Union[Mapping[str, Union[float, int, Sequence[float], Sequence[int]]], "ContextualSeries"]


class ContextualSeries(dict):
    """Dictionary subclass to annotate contextual data."""


class DynamicContextPromptBuilder:
    """Convert structured context into natural language summaries."""

    def __init__(self, keys: Optional[Sequence[str]] = None) -> None:
        self.keys = [key.lower() for key in keys] if keys else []

    def __call__(self, batch_context: Sequence[ContextType], horizon: int) -> List[str]:
        return [self.build_prompt(context, horizon) for context in batch_context]

    def build_prompt(self, context: ContextType, horizon: int) -> str:
        if context is None:
            return "No additional external context provided."

        if isinstance(context, Mapping):
            return self._build_from_mapping(context, horizon)
        if isinstance(context, ContextualSeries):
            return self._build_from_mapping(context, horizon)
        return str(context)

    def _build_from_mapping(self, mapping: Mapping[str, object], horizon: int) -> str:
        statements: List[str] = []
        lowered = {key.lower(): value for key, value in mapping.items()}

        if self.keys:
            keys = [key for key in self.keys if key in lowered]
        else:
            keys = list(lowered.keys())

        for key in keys:
            value = lowered.get(key)
            statement = self._format_single_key(key, value, horizon)
            if statement:
                statements.append(statement)

        if not statements:
            return "No significant external events reported."
        return " ".join(statements)

    def _format_single_key(self, key: str, value: object, horizon: int) -> str:
        if value is None:
            return ""
        key_lower = key.lower()

        if key_lower in {"temperature", "temp"}:
            temps = self._to_float_array(value)
            if temps.size == 0:
                return ""
            mean_temp = float(np.mean(temps))
            max_temp = float(np.max(temps))
            min_temp = float(np.min(temps))
            severity = "heatwave" if max_temp >= 32 else "cold spell" if min_temp <= -5 else "moderate"
            return (
                f"Temperature forecast spans {temps.size} intervals with mean {mean_temp:.1f}°C. "
                f"Expected peak {max_temp:.1f}°C and low {min_temp:.1f}°C, indicating a {severity}."
            )

        if key_lower in {"humidity"}:
            hum = self._to_float_array(value)
            if hum.size == 0:
                return ""
            avg_hum = float(np.mean(hum))
            return f"Average humidity around {avg_hum:.0f}% over the horizon."

        if key_lower in {"holiday", "is_holiday"}:
            indicator = self._to_float_array(value)
            if indicator.size == 0:
                return ""
            active = bool(np.any(indicator > 0.5))
            return "Public holiday period continues." if active else "No holiday disruptions expected."

        if key_lower in {"event", "note"} and isinstance(value, str):
            return value

        numeric = self._to_float_array(value)
        if numeric.size > 0:
            mean_val = float(np.mean(numeric))
            return f"Average {key_lower} projected at {mean_val:.2f}."
        return f"{key} context provided."

    def _to_float_array(self, value: object) -> np.ndarray:
        if value is None:
            return np.array([], dtype=np.float32)
        if isinstance(value, (list, tuple, np.ndarray)):
            try:
                return np.asarray(value, dtype=np.float32)
            except (TypeError, ValueError):
                return np.array([], dtype=np.float32)
        if isinstance(value, (float, int)):
            return np.array([float(value)], dtype=np.float32)
        return np.array([], dtype=np.float32)

