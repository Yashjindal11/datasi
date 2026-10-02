"""Self-contained HTML report. Works offline from a file:// URL: no server, no CDN.

Report data is embedded as JSON (with '<' escaped so values cannot close the script
tag) and rendered client-side using DOM APIs (textContent, never innerHTML), so labels
from the dataset cannot inject markup.
"""

from __future__ import annotations

import html
import json
from importlib import resources

from datasi.detectors.base import registry
from datasi.reports.model import Report


def _asset(name: str) -> str:
    return resources.files("datasi.reports").joinpath("assets", name).read_text(encoding="utf-8")


def embed_json(data: object) -> str:
    text = json.dumps(data, ensure_ascii=False)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def render_html(report: Report) -> str:
    from datasi.core.compare import DriftDetector

    data = report.to_dict()
    docs = {name: cls.info() for name, cls in registry().items()}
    docs["drift"] = DriftDetector.info()
    data["detector_docs"] = docs
    title = f"DataSI report: {report.dataset.name}"
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:\">"
        f"<title>{html.escape(title)}</title>"
        f"<style>{_asset('report.css')}</style></head>"
        '<body><div id="app"><noscript>This report needs JavaScript to render. '
        "The raw data is embedded in the page source as JSON.</noscript></div>"
        f'<script type="application/json" id="datasi-data">{embed_json(data)}</script>'
        f"<script>{_asset('report.js')}</script></body></html>\n"
    )
