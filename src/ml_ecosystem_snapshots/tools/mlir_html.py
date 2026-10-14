"""HTML and Markdown documentation parser utilities for MLIR dialect scraping."""

from __future__ import annotations

import re
import warnings
from typing import Any, Callable, Dict, List, Optional


def fetch_html(url: str) -> str:
    """Fetch HTML content from a URL.

    Args:
        url: The HTTP or HTTPS URL to fetch.

    Returns:
        The raw HTML body string, or an empty string if fetching fails.
    """
    from ml_ecosystem_snapshots import utils

    res = utils.fetch_url(url)
    if res is None:
        print(f"Error fetching {url}")
        return ""
    return res


def parse_table(html_chunk: str, expected_cols: int) -> List[str]:
    """Parse an HTML table to extract column contents.

    .. deprecated:: 0.2.0
        HTML doc scraping is deprecated in favor of TableGen ODS parsing.

    Args:
        html_chunk: Raw HTML snippet containing the table.
        expected_cols: Expected minimum column count.

    Returns:
        List of extracted column strings.
    """
    warnings.warn(
        "HTML doc scraping is deprecated in favor of TableGen ODS parsing.",
        DeprecationWarning,
        stacklevel=2,
    )
    results = []
    rows = re.findall(r"<tr>(.*?)</tr>", html_chunk, re.DOTALL)
    for row in rows:
        cells = re.findall(r"<td>(.*?)</td>", row, re.DOTALL)
        if len(cells) >= 1:
            name_cell = cells[0]
            clean_name = re.sub(r"<[^>]+>", "", name_cell).strip()
            if clean_name and not clean_name.startswith("&"):
                results.append(clean_name)
    return results


def parse_dialect_page(
    url: str,
    dialect_name: str,
    html_fetcher: Optional[Callable[[str], str]] = None,
) -> List[Dict[str, Any]]:
    """Parse an MLIR dialect doc page for its operations.

    .. deprecated:: 0.2.0
        HTML doc scraping is deprecated in favor of TableGen ODS parsing.

    Args:
        url: The documentation URL to fetch.
        dialect_name: The dialect prefix name.
        html_fetcher: Optional callable to fetch HTML content from URL.

    Returns:
        List of dictionaries detailing parsed dialect operations.
    """
    from ml_ecosystem_snapshots.tools import scrape_mlir

    warnings.warn(
        "HTML doc scraping is deprecated in favor of TableGen ODS parsing.",
        DeprecationWarning,
        stacklevel=2,
    )
    fetcher = html_fetcher or scrape_mlir.fetch_html
    html = fetcher(url)
    if not html:
        return []

    ops = []
    sections = re.split(r'<h3 id="([^"]+)">', html)
    if len(sections) <= 1:
        return []

    for i in range(1, len(sections), 2):
        sec_content = sections[i + 1]

        header_match = re.search(
            r"<code>([a-zA-Z0-9_\.]+)</code>\s*\(([^)]+)\)", sec_content
        )
        if not header_match:
            continue

        api_path = header_match.group(1)
        class_name = header_match.group(2).split("::")[-1]

        operands: List[str] = []
        attributes: List[str] = []

        op_match = re.search(
            r"<h4>Operands:</h4>\s*<table>(.*?)</table>", sec_content, re.DOTALL
        )
        if op_match:
            operands = parse_table(op_match.group(1), 2)

        attr_match = re.search(
            r"<h4>Attributes:</h4>\s*<table>(.*?)</table>",
            sec_content,
            re.DOTALL,
        )
        if attr_match:
            attributes = parse_table(attr_match.group(1), 2)

        desc_match = re.search(r"<p><em>(.*?)</em></p>", sec_content, re.DOTALL)
        description = None
        if desc_match:
            description = re.sub(r"<[^>]+>", "", desc_match.group(1)).strip()

        ops.append(
            {
                "api_path": api_path,
                "dialect": dialect_name,
                "class_name": class_name,
                "operands": operands,
                "attributes": attributes,
                "description": description,
            }
        )

    return ops


def scrape_stablehlo(
    html_fetcher: Optional[Callable[[str], str]] = None,
) -> List[Dict[str, Any]]:
    """Scrape the StableHLO spec markdown to extract operations.

    .. deprecated:: 0.2.0
        Use build_stablehlo_snapshot instead.

    Args:
        html_fetcher: Optional callable to fetch Markdown content from URL.

    Returns:
        List of scraped StableHLO operation dictionaries.
    """
    from ml_ecosystem_snapshots.tools import scrape_mlir

    nl = chr(10)
    warnings.warn(
        "scrape_stablehlo is deprecated. Use build_stablehlo_snapshot instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    url = "https://raw.githubusercontent.com/openxla/stablehlo/main/docs/spec.md"
    fetcher = html_fetcher or scrape_mlir.fetch_html
    md = fetcher(url)
    ops: List[Dict[str, Any]] = []
    if not md:
        return ops

    sections = re.split(nl + r"### (?![#])", md)
    for sec in sections[1:]:
        lines = sec.split(nl)
        op_name = lines[0].strip()
        if not re.match(r"^[a-z0-9_]+$", op_name) or op_name in (
            "functions",
            "types",
            "constants",
        ):
            continue

        operands: List[str] = []
        attributes: List[str] = []
        desc_lines = []
        current_header = ""
        for line in lines[1:]:
            if line.startswith("#### "):
                current_header = line[5:].strip().lower()
            elif current_header == "inputs":
                if line.startswith("|") and not line.startswith("|---"):
                    parts = [p.strip() for p in line.split("|")[1:-1]]
                    if len(parts) >= 2 and parts[0] != "Name" and parts[0] != "Label":
                        p_name = (
                            parts[1].strip("`")
                            if len(parts) >= 3 and parts[0].startswith("(")
                            else parts[0].strip("`")
                        )
                        if p_name:
                            operands.append(p_name)
            elif current_header == "semantics":
                if line.strip() and not line.startswith("#"):
                    desc_lines.append(line.strip())

        description = " ".join(desc_lines) if desc_lines else None
        class_name = "".join(part.capitalize() for part in op_name.split("_")) + "Op"

        ops.append(
            {
                "api_path": f"stablehlo.{op_name}",
                "dialect": "stablehlo",
                "class_name": class_name,
                "operands": operands,
                "attributes": attributes,
                "description": description,
            }
        )

    return ops


__all__ = ["fetch_html", "parse_dialect_page", "parse_table", "scrape_stablehlo"]
