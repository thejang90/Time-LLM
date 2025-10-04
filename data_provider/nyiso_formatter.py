"""Utilities for preparing NYISO market data with daylight saving adjustments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class FormatterConfig:
    """Configuration options for :class:`NYISODataFormatter`."""

    timezone: str = "America/New_York"
    freq: str = "H"
    fill_strategy: str = "interpolate"
    numeric_columns: Optional[Sequence[str]] = None


class NYISODataFormatter:
    """Pre-process NYISO data while resolving daylight saving irregularities.

    The formatter expects a dataframe containing at least the columns
    ``timestamp``, ``da_load``, ``da_smp`` and ``location``. It reindexes the
    data to an hourly frequency for every location and fills missing hours that
    appear during daylight saving transitions.
    """

    def __init__(self, config: Optional[FormatterConfig] = None):
        self.config = config or FormatterConfig()
        self.locations: List[str] = []
        self._formatted_frame: Optional[pd.DataFrame] = None

    def format(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a cleaned dataframe with aligned hourly samples per location."""

        required = {"timestamp", "da_load", "da_smp", "location"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df = frame.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp", "location"])

        df["location"] = df["location"].astype(str)
        df = self._localize_timestamp(df)
        df = df.sort_values(["location", "timestamp"])

        start = df["timestamp"].min().floor(self.config.freq)
        end = df["timestamp"].max().ceil(self.config.freq)
        complete_index = pd.date_range(start=start, end=end, freq=self.config.freq)

        formatted_blocks: List[pd.DataFrame] = []
        self.locations = sorted(df["location"].unique())

        for location, group in df.groupby("location"):
            formatted_blocks.append(
                self._format_single_location(location, group, complete_index)
            )

        formatted = pd.concat(formatted_blocks, axis=1).sort_index()
        self._formatted_frame = formatted
        return formatted

    def extract_features(
        self, formatted: Optional[pd.DataFrame] = None, variables: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        """Return a feature matrix for the requested variables.

        Parameters
        ----------
        formatted: Optional[pd.DataFrame]
            If ``None``, the dataframe generated during the last call to
            :meth:`format` is used.
        variables: Optional[Sequence[str]]
            Subset of variables to keep (e.g. ``["da_load"]``). If ``None``, the
            numeric columns discovered while formatting are retained.
        """

        if formatted is None:
            if self._formatted_frame is None:
                raise ValueError("No formatted dataframe available. Call format() first.")
            formatted = self._formatted_frame

        if variables is None:
            variables = self.config.numeric_columns
        if variables is None:
            variables = [col[1] for col in formatted.columns.unique(level=1)]

        columns = [col for col in formatted.columns if col[1] in set(variables)]
        data = formatted[columns]
        data.columns = pd.Index([f"{loc}_{var}" for loc, var in columns], name="feature")
        return data

    def build_adjacency(
        self,
        edges: Optional[Iterable[Sequence[str]]] = None,
        fully_connected: bool = True,
        include_self_loops: bool = True,
    ) -> np.ndarray:
        """Construct an adjacency matrix for the known locations."""

        if not self.locations:
            raise ValueError("Locations are unknown. Call format() before building adjacency.")

        from utils.graph import build_adjacency_matrix

        return build_adjacency_matrix(
            nodes=self.locations,
            edges=edges,
            fully_connected=fully_connected,
            include_self_loops=include_self_loops,
        )

    def _format_single_location(
        self, location: str, group: pd.DataFrame, complete_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        numeric_cols = [
            col
            for col in group.columns
            if col not in {"timestamp", "location"} and np.issubdtype(group[col].dtype, np.number)
        ]
        if numeric_cols:
            self.config.numeric_columns = numeric_cols

        location_frame = (
            group.set_index("timestamp")[numeric_cols]
            .groupby(level=0)
            .mean()
            .reindex(complete_index)
        )

        if self.config.fill_strategy == "interpolate":
            location_frame = location_frame.interpolate(method="time", limit_direction="both")
        elif self.config.fill_strategy == "ffill":
            location_frame = location_frame.ffill().bfill()
        elif self.config.fill_strategy == "bfill":
            location_frame = location_frame.bfill().ffill()
        else:
            raise ValueError(f"Unsupported fill strategy: {self.config.fill_strategy}")

        location_frame.columns = pd.MultiIndex.from_product([[location], location_frame.columns])
        return location_frame

    def _localize_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = df["timestamp"]
        try:
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize(
                    self.config.timezone,
                    ambiguous="NaT",
                    nonexistent="shift_forward",
                )
            else:
                ts = ts.dt.tz_convert(self.config.timezone)
        except (TypeError, AttributeError, ValueError):
            ts = ts.dt.tz_localize(None)

        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        df = df.copy()
        df["timestamp"] = ts
        df = df.dropna(subset=["timestamp"])
        return df

