"""Session-journal helpers independent of Streamlit."""
import csv
import io

from emotion_model import EMOTIONS


def export_csv(history):
    """Escape spreadsheet formula prefixes while preserving the in-memory entry."""
    output = io.StringIO(newline="")
    fields = ["time", "text", "top", "confidence", "uncertain", "crisis", "reading_status", *EMOTIONS]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for entry in history:
        row = dict(entry)
        text = row["text"]
        if text.lstrip().startswith(("=", "+", "-", "@")) or text.startswith(("\t", "\r", "\n")):
            row["text"] = "'" + text
        writer.writerow(row)
    return output.getvalue()
