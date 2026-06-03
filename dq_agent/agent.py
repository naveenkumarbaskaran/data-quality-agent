"""DQAgent: Claude-powered data quality analysis agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anthropic

from .profiler import DataProfiler

# ---------------------------------------------------------------------------
# Tool definitions (JSON Schema) for the Anthropic tool-use API
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_csv",
        "description": (
            "Load a CSV or Parquet file from disk and return the first 10 rows "
            "as a JSON-serialisable list of records, plus the column names and "
            "total row count. Supports .csv and .parquet extensions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the CSV/Parquet file.",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "profile_dataframe",
        "description": (
            "Compute a comprehensive statistical profile for every column in the "
            "dataset: null rates, mean, std, min, max, quartiles, skewness, kurtosis, "
            "unique count, and top-10 most frequent values. Works on .csv and .parquet."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the CSV or Parquet file to profile.",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "detect_anomalies",
        "description": (
            "Detect anomalies/outliers in a specific column. "
            "Uses IQR-based outlier detection for numeric columns and "
            "rare-frequency detection (< 1% occurrence) for categorical columns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the CSV or Parquet file.",
                },
                "column": {
                    "type": "string",
                    "description": "Name of the column to check for anomalies.",
                },
            },
            "required": ["path", "column"],
        },
    },
]

# System prompt
SYSTEM_PROMPT = """
You are a data-quality analyst AI. Your job is to produce a thorough,
actionable data quality report for the dataset(s) given to you.

For each dataset you MUST:
1. Call `read_csv` to inspect the raw data.
2. Call `profile_dataframe` to get full statistics.
3. Call `detect_anomalies` for EVERY column that looks suspicious
   (high null rate, extreme values, unexpected categories, etc.).
4. Synthesise the findings into a structured Markdown report that includes:
   - Executive summary
   - Dataset overview (shape, file, dtypes)
   - Per-column statistics (null rate, distribution, top values)
   - Anomaly / outlier findings
   - Schema drift analysis (if a baseline profile is provided)
   - Data quality score (0-100) and rationale
   - Prioritised recommendations

Be precise, cite actual numbers from the tool outputs, and flag any
critical issues prominently.
"""


class DQAgent:
    """Orchestrates Claude to perform automated data quality analysis.

    Parameters
    ----------
    api_key:
        Anthropic API key.  If ``None`` the ``ANTHROPIC_API_KEY`` environment
        variable is used.
    model:
        Claude model to use.  Defaults to ``claude-sonnet-4-6``.
    max_tokens:
        Maximum output tokens per response turn.  Default 8192.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 8192,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = (
            anthropic.Anthropic(api_key=api_key)
            if api_key
            else anthropic.Anthropic()
        )
        self._profiler = DataProfiler()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def analyse(
        self,
        path: str | Path,
        baseline_profile: dict[str, Any] | None = None,
    ) -> str:
        """Run a full data-quality analysis on *path*.

        Parameters
        ----------
        path:
            CSV or Parquet file to analyse.
        baseline_profile:
            Optional profile dict from a previous run (e.g. loaded from JSON).
            When provided, schema-drift analysis is included in the report.

        Returns
        -------
        str
            Markdown-formatted data quality report.
        """
        path = str(Path(path).resolve())
        user_message = f"Please produce a complete data quality report for: {path}"
        if baseline_profile:
            user_message += (
                "\n\nA baseline profile from a previous run is provided below. "
                "Include a schema-drift analysis section in your report.\n\n"
                f"```json\n{json.dumps(baseline_profile, indent=2)}\n```"
            )

        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

        # Agentic loop: keep calling the API until the model stops requesting tools
        while True:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=TOOLS,  # type: ignore[arg-type]
                messages=messages,
            )

            # Append assistant response to history
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # Extract final text response
                return self._extract_text(response)

            if response.stop_reason != "tool_use":
                # Unexpected stop – return whatever text we have
                return self._extract_text(response)

            # Execute tool calls and collect results
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type == "tool_use":
                    result_content = self._dispatch_tool(block.name, block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result_content, default=str),
                        }
                    )

            messages.append({"role": "user", "content": tool_results})

    def save_profile(self, path: str | Path, output: str | Path) -> dict[str, Any]:
        """Compute and save a profile JSON for *path* to *output*.

        The saved JSON can later be passed as *baseline_profile* to
        :meth:`analyse` for schema-drift detection.

        Returns the profile dict.
        """
        profile = self._profiler.profile(path)
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(profile, indent=2, default=str))
        return profile

    # ------------------------------------------------------------------ #
    # Tool dispatch                                                        #
    # ------------------------------------------------------------------ #

    def _dispatch_tool(
        self, name: str, inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Call the appropriate profiler method for *name*."""
        try:
            if name == "read_csv":
                return self._tool_read_csv(inputs["path"])
            if name == "profile_dataframe":
                return self._profiler.profile(inputs["path"])
            if name == "detect_anomalies":
                return self._profiler.detect_anomalies(
                    inputs["path"], inputs["column"]
                )
            return {"error": f"Unknown tool: {name}"}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    @staticmethod
    def _tool_read_csv(path: str) -> dict[str, Any]:
        """Return a preview of the dataset."""
        import pandas as pd  # local import to keep top-level lean

        p = Path(path)
        if not p.exists():
            return {"error": f"File not found: {path}"}
        suffix = p.suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(p)
        elif suffix in (".csv", ".tsv"):
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(p, sep=sep)
        else:
            return {"error": f"Unsupported file type: {suffix}"}

        return {
            "path": str(p.resolve()),
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "preview": df.head(10).to_dict(orient="records"),
        }

    # ------------------------------------------------------------------ #
    # Response helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_text(response: anthropic.types.Message) -> str:
        parts = [
            block.text
            for block in response.content
            if hasattr(block, "text")
        ]
        return "\n\n".join(parts) if parts else "(No text in response)"
