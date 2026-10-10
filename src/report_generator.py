"""Generate JSON and optionally HTML interview reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import REPORTS_DIR


def generate_report(report: dict[str, Any], output_path: str | Path = REPORTS_DIR / "interview_report.json") -> dict[str, Any]:
    """Save a JSON report and an HTML representation."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    html_path = output.with_suffix(".html")
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Interview Analysis Report</title>
    <style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:auto}}h1{{color:#2945a5}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border:1px solid #ccc}}</style></head>
    <body><h1>Interview Emotion & Confidence Analysis</h1><p><strong>Video:</strong>{report.get('video','')}</p>
    <p><strong>Behavioral confidence score:</strong>{report.get('behavioral_confidence_score',0)} / 100</p>
    <p><strong>Dominant emotion:</strong>{report.get('dominant_emotion','unknown')}</p>
    <h2>Emotion distribution</h2><table><tr><th>Emotion</th><th>Percent</th></tr>{''.join('<tr><td>'+str(k)+'</td><td>'+str(v)+'</td></tr>' for k,v in report.get('emotion_distribution',{}).items())}</table>
    <h2>Interpretation</h2><p>{report.get('overall_interpretation','')}</p><p><em>Warning: This score is an observable behavioral proxy, not a definitive psychological or clinical assessment.</em></p></body></html>"""
    html_path.write_text(html, encoding="utf-8")
    return {"json": str(output), "html": str(html_path)}
