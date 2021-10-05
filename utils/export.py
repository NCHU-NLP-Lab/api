import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

from docx import Document
from model import QAExportItem


def _export_json(qa_pairs):
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = Path(f"{now}.json")
    with open(filename, "w") as f:
        json.dump([qa_pair.dict() for qa_pair in qa_pairs], f)
    return filename


def _export_txt(qa_pairs):
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = Path(f"{now}.txt")
    with open(filename, "w") as f:
        for qa_pair in qa_pairs:
            f.write(f"{qa_pair.context}\n\n")
            f.write(f"{qa_pair.question}\n\n")
            for option in qa_pair.options:
                if option.is_answer:
                    f.write(f"* {option.option}\n")
                else:
                    f.write(f"- {option.option}\n")
            f.write("\n")
    return filename


def _export_docx(qa_pairs: List[QAExportItem]):
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = Path(f"{now}.docx")

    document = Document()
    for qa_pair in qa_pairs:
        document.add_paragraph(f"{qa_pair.context}\n")
        document.add_paragraph(f"{qa_pair.question}\n")
        for option in qa_pair.options:
            if option.is_answer:
                document.add_paragraph(f"+ {option.option}")
            else:
                document.add_paragraph(f"- {option.option}")

    document.save(filename)

    return filename


def export_file(qa_pairs: List[QAExportItem], format: str):
    if format == "json":
        return _export_json(qa_pairs)
    elif format == "txt":
        return _export_txt(qa_pairs)
    elif format == "docx":
        return _export_docx(qa_pairs)
    else:
        raise ValueError(f"Unsupported format: {format}")


async def delete_later(file_path, wait=120):
    await asyncio.sleep(wait)
    os.remove(file_path)
