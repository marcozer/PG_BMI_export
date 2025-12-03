"""Minimal Excel loader used by dataset if a .xlsx is provided.

Copied from original analysis/lib/data_loader.py to avoid optional Excel deps.
"""

import re
import xml.etree.ElementTree as ET
import zipfile
from functools import lru_cache
from typing import Iterable, Optional

import pandas as pd

_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

_COLUMN_RE = re.compile(r"([A-Z]+)")


def _col_to_index(col_ref: str) -> int:
    idx = 0
    for ch in col_ref:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _iter_rows(sheet_xml: bytes, shared_strings: Iterable[str]):
    shared = list(shared_strings)
    root = ET.fromstring(sheet_xml)
    for row in root.findall(".//main:row", _NS):
        row_dict = {}
        for cell in row.findall("main:c", _NS):
            ref = cell.attrib.get("r", "")
            match = _COLUMN_RE.match(ref)
            if not match:
                continue
            col_idx = _col_to_index(match.group(1))
            cell_type = cell.attrib.get("t")
            value_elem = cell.find("main:v", _NS)
            raw_value = value_elem.text if value_elem is not None else ""
            if cell_type == "s" and raw_value:
                try:
                    raw_value = shared[int(raw_value)]
                except (IndexError, ValueError):
                    raw_value = ""
            row_dict[col_idx] = raw_value
        if row_dict:
            yield row_dict


def _read_shared_strings(zip_file: zipfile.ZipFile) -> Iterable[str]:
    if "xl/sharedStrings.xml" not in zip_file.namelist():
        return []
    tree = ET.fromstring(zip_file.read("xl/sharedStrings.xml"))
    strings = []
    for si in tree.findall("main:si", _NS):
        parts = [t.text or "" for t in si.findall(".//main:t", _NS)]
        strings.append("".join(parts))
    return strings


def _sheet_path_from_name(zip_file: zipfile.ZipFile, sheet_name: str) -> Optional[str]:
    workbook = ET.fromstring(zip_file.read("xl/workbook.xml"))
    for sheet in workbook.findall("main:sheet", _NS):
        if sheet.attrib.get("name") == sheet_name:
            rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            rels = ET.fromstring(zip_file.read("xl/_rels/workbook.xml.rels"))
            for rel in rels.findall("rel:Relationship", _NS):
                if rel.attrib.get("Id") == rel_id:
                    return f"xl/{rel.attrib['Target']}"
    return None


@lru_cache(maxsize=None)
def load_excel_sheet(filepath: str, *, sheet: Optional[str] = None, sheet_index: int = 1) -> pd.DataFrame:
    with zipfile.ZipFile(filepath) as zf:
        shared_strings = _read_shared_strings(zf)
        if sheet:
            sheet_path = _sheet_path_from_name(zf, sheet)
            if sheet_path is None:
                raise ValueError(f"Sheet '{sheet}' not found in {filepath}")
        else:
            sheet_path = f"xl/worksheets/sheet{sheet_index}.xml"
        sheet_xml = zf.read(sheet_path)

    rows = list(_iter_rows(sheet_xml, shared_strings))
    if not rows:
        return pd.DataFrame()

    max_col = max(max(row.keys()) for row in rows)
    data = []
    for row in rows:
        row_list = [""] * (max_col + 1)
        for idx, value in row.items():
            row_list[idx] = value
        data.append(row_list)

    header, *body = data
    header = list(header)
    if len(header) < (max_col + 1):
        header.extend(f"col_{i}" for i in range(len(header), max_col + 1))
    padded_body = [r + [""] * (len(header) - len(r)) for r in body]

    return pd.DataFrame(padded_body, columns=header)

