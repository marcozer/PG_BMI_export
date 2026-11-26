from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data" / "raw" / "pg_afc_sheet1.csv"
PKG = ROOT / "hbsn_revision" / "minimal_R1_from_R0"
OUT_DOCX = PKG / "docx" / "06_HBSN_main_tables_R0_format_BMI3.docx"
OUT_CSV_DIR = PKG / "tables_csv"

BMI_BINS = [0, 25, 30, np.inf]
BMI_LABELS = ["<25", "25-29.9", ">=30"]
LOS_75TH_THRESHOLD = 13
LOS_25TH_THRESHOLD = 7


@dataclass(frozen=True)
class Group:
    label: str
    mask: pd.Series


def p_format(p: float, *, table2: bool = False) -> str:
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "<0.001"
    if table2 and p < 0.01:
        return "<0.01"
    return f"{p:.3f}"


def is_sig(p_text: str) -> bool:
    p_text = str(p_text).strip()
    if not p_text:
        return False
    if p_text.startswith("<"):
        try:
            return float(p_text[1:]) < 0.05
        except ValueError:
            return False
    try:
        return float(p_text) < 0.05
    except ValueError:
        return False


def fmt_n_pct(n: int, denom: int, *, decimals: int = 1) -> str:
    if denom <= 0:
        return ""
    return f"{n} ({100.0 * n / denom:.{decimals}f})"


def fmt_median_iqr(series: pd.Series, *, decimals: int = 1) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return ""
    q1, q3 = s.quantile([0.25, 0.75])
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(s.median())} ({fmt.format(q1)}-{fmt.format(q3)})"


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0.0
    s = series.fillna("").astype(str).str.strip().str.upper()
    return s.isin({"OUI", "YES", "TRUE", "1", "Y", "T", "URG"})


def chi2_p(table: np.ndarray) -> float:
    try:
        _, p, _, _ = stats.chi2_contingency(table)
        return float(p)
    except Exception:
        return float("nan")


def kruskal_p(values_by_group: list[pd.Series]) -> float:
    arrays = [pd.to_numeric(v, errors="coerce").dropna().to_numpy() for v in values_by_group]
    arrays = [a for a in arrays if a.size > 0]
    if len(arrays) < 2:
        return float("nan")
    return float(stats.kruskal(*arrays).pvalue)


def build_groups(df: pd.DataFrame) -> list[Group]:
    bmi = pd.to_numeric(df["IMC"], errors="coerce")
    bmi_cat = pd.cut(bmi, bins=BMI_BINS, labels=BMI_LABELS, right=False, include_lowest=True)
    return [Group(label=lab, mask=(bmi_cat == lab)) for lab in BMI_LABELS]


def compute_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    los = pd.to_numeric(out["LHS"], errors="coerce")
    mortality = to_bool(out["DECES"])
    major_clavien = pd.to_numeric(out["CLAVIEN"], errors="coerce").ge(3)
    popf = out["POPF"].fillna("").astype(str).str.upper().str.strip()
    popf_bc = popf.isin({"B", "C"})
    reop = to_bool(out["REOPERATION"])
    readm = to_bool(out["REHOSPITALISATION"])

    ideal = (
        (~mortality)
        & (~major_clavien)
        & (~popf_bc)
        & (~reop)
        & (~readm)
        & ((los <= LOS_75TH_THRESHOLD) | los.isna())
    )
    out["ideal_outcome_calc"] = ideal
    out["best_performer_calc"] = ideal & (los <= LOS_25TH_THRESHOLD)
    return out


def init_row(label: str, df: pd.DataFrame, groups: list[Group]) -> dict[str, str]:
    row = {"Variable": label, f"Total (n={len(df)})": ""}
    for g in groups:
        row[f"BMI {g.label} kg/m2 (n={int(g.mask.sum())})"] = ""
    row["p-value"] = ""
    return row


def add_section(rows: list[dict[str, str]], label: str, df: pd.DataFrame, groups: list[Group], p_text: str = "") -> None:
    row = init_row(label, df, groups)
    row["p-value"] = p_text
    rows.append(row)


def add_binary_row(
    rows: list[dict[str, str]],
    label: str,
    df: pd.DataFrame,
    groups: list[Group],
    indicator: pd.Series,
    *,
    p_on_row: bool = True,
    decimals: int = 1,
    table2: bool = False,
) -> None:
    ind = indicator.fillna(False).astype(bool)
    row = init_row(label, df, groups)
    row[f"Total (n={len(df)})"] = fmt_n_pct(int(ind.sum()), len(df), decimals=decimals)
    contingency = []
    for g in groups:
        denom = int(g.mask.sum())
        n = int(ind[g.mask].sum())
        row[f"BMI {g.label} kg/m2 (n={denom})"] = fmt_n_pct(n, denom, decimals=decimals)
        contingency.append([n, denom - n])
    row["p-value"] = p_format(chi2_p(np.array(contingency)), table2=table2) if p_on_row else ""
    rows.append(row)


def add_continuous_row(
    rows: list[dict[str, str]],
    label: str,
    df: pd.DataFrame,
    groups: list[Group],
    values: pd.Series,
    *,
    decimals: int = 1,
    table2: bool = False,
) -> None:
    row = init_row(label, df, groups)
    row[f"Total (n={len(df)})"] = fmt_median_iqr(values, decimals=decimals)
    values_by_group = []
    for g in groups:
        denom = int(g.mask.sum())
        vg = values[g.mask]
        row[f"BMI {g.label} kg/m2 (n={denom})"] = fmt_median_iqr(vg, decimals=decimals)
        values_by_group.append(vg)
    row["p-value"] = p_format(kruskal_p(values_by_group), table2=table2)
    rows.append(row)


def build_table1(df: pd.DataFrame, groups: list[Group]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    add_binary_row(rows, "Female gender", df, groups, df["SEXE"].astype(str).str.upper().eq("F"))
    add_continuous_row(rows, "Age (median, IQR)", df, groups, df["AGE"], decimals=1)
    add_binary_row(rows, "ASA 3-4", df, groups, pd.to_numeric(df["ASA"], errors="coerce") >= 3)
    add_binary_row(rows, "MEN1", df, groups, to_bool(df["NEM"]))
    add_binary_row(rows, "Preexisting diabetes", df, groups, to_bool(df["DIABETE"]))
    add_binary_row(rows, "Chronic renal disease", df, groups, to_bool(df["IRC"]))
    add_binary_row(rows, "Cardiac history", df, groups, to_bool(df["CARDIO"]))
    add_binary_row(rows, "Hypertension", df, groups, to_bool(df["HTA"]))
    add_binary_row(rows, "COPD", df, groups, to_bool(df["BPCO"]))
    add_binary_row(rows, "Cirrhosis", df, groups, to_bool(df["FOIE"]))
    add_binary_row(rows, "Chronic pancreatitis", df, groups, to_bool(df["PCC"]))

    add_section(rows, "Lesion location", df, groups)
    add_binary_row(rows, "  Isthmic", df, groups, df["SIEGE_ISTHME"].astype(str).str.upper().eq("OUI"))

    add_binary_row(rows, "Resecable tumor", df, groups, df["RESECABILITE"].astype(str).str.upper().eq("RES"))
    add_continuous_row(rows, "Tumor size (mm)", df, groups, df["TAILLE_PREOP"], decimals=0)
    add_continuous_row(rows, "Pancreatic duct size (mm)", df, groups, df["CPP_DILATE"], decimals=1)
    add_binary_row(rows, "Robotic DP", df, groups, df["ABORD"].astype(str).str.upper().eq("ROBOT"))
    add_binary_row(rows, "Conversion", df, groups, to_bool(df["CONVERSION"]))
    add_binary_row(rows, "Soft pancreatic texture", df, groups, df["CONSISTANCE"].astype(str).str.upper().eq("MOU"))

    splen = df["SPLENECTOMIE"].fillna("").astype(str).str.upper().str.strip()
    splen_any = splen.isin({"OUI", "URG"})
    splen_emerg = splen.eq("URG")
    add_binary_row(rows, "Associated splenectomy", df, groups, splen_any)
    add_binary_row(rows, "  Emergency", df, groups, splen_emerg, p_on_row=False)

    vasc = df["PG_RESECTION_VSX"].fillna("").astype(str).str.upper().str.strip()
    add_binary_row(rows, "Vascular resection", df, groups, ~vasc.isin({"NON", "NAN", ""}))
    add_binary_row(rows, "RAMPS", df, groups, to_bool(df["RAMPS"]))

    if "RMO" in df.columns:
        rmo = df["RMO"].fillna("").astype(str).str.strip().str.upper()
        add_binary_row(rows, "Multiorgan resection", df, groups, ~rmo.isin({"NON", "NAN", ""}))
    else:
        add_binary_row(rows, "Multiorgan resection", df, groups, pd.Series(False, index=df.index))

    add_continuous_row(rows, "Peroperative bleeding (ml)", df, groups, df["SANG"], decimals=0)
    add_continuous_row(rows, "Operative time (min)", df, groups, df["DUREE_OP"], decimals=0)

    add_section(rows, "Type of lesion", df, groups)
    lesion_types = [
        ("  Adenocarcinoma", "ADK"),
        ("  NET", "TNE"),
        ("  IPMN", "TIPMP"),
        ("  STP", "TSPP"),
        ("  Mucinous cyst", "CM"),
    ]
    for label, col in lesion_types:
        add_binary_row(rows, label, df, groups, to_bool(df[col]))

    add_binary_row(rows, "Positive lymph nodes", df, groups, df["N"].astype(str).str.upper().eq("N1"))
    add_binary_row(rows, "R1 margins", df, groups, df["R1"].astype(str).str.upper().eq("R1"))

    return pd.DataFrame(rows)


def build_table2(df: pd.DataFrame, groups: list[Group]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    popf = df["POPF"].fillna("").astype(str).str.upper().str.strip()
    add_binary_row(rows, "POPF", df, groups, ~popf.isin({"", "NON", "NAN"}), table2=True)
    add_binary_row(rows, "  B/C", df, groups, popf.isin({"B", "C"}), table2=True)
    add_binary_row(rows, "PPH", df, groups, to_bool(df["PPH"]), table2=True)
    add_binary_row(rows, "Delayed gastric emptying", df, groups, to_bool(df["DGE"]), table2=True)
    add_binary_row(rows, "Pulmonary complication", df, groups, df["RESPI"].fillna("").astype(str).str.upper().str.strip().ne("NON"), table2=True)
    add_binary_row(rows, "Site infection", df, groups, to_bool(df["SSI"]), table2=True)
    add_binary_row(rows, "Postoperative collection", df, groups, df["COLLECTION"].fillna("").astype(str).str.upper().str.strip().eq("OUI"), table2=True)
    add_binary_row(rows, "Reintervention", df, groups, to_bool(df["REOPERATION"]), table2=True)
    add_continuous_row(rows, "ICU length of stay (days)", df, groups, df["REA_DUREE"], decimals=0, table2=True)
    add_continuous_row(rows, "Total length of hospital stay (days)", df, groups, df["LHS"], decimals=0, table2=True)
    add_binary_row(rows, "Postoperative death", df, groups, to_bool(df["DECES"]), table2=True)
    add_binary_row(rows, "Rehospitalisation", df, groups, to_bool(df["REHOSPITALISATION"]), table2=True)

    clav = pd.to_numeric(df["CLAVIEN"], errors="coerce")
    contingency = []
    for g in groups:
        contingency.append([int((clav[g.mask] == k).sum()) for k in [1, 2, 3, 4]])
    add_section(rows, "Clavien-Dindo", df, groups, p_format(chi2_p(np.array(contingency)), table2=True))
    for k in [1, 2, 3, 4]:
        row = init_row(f"  {k}", df, groups)
        row[f"Total (n={len(df)})"] = fmt_n_pct(int((clav == k).sum()), len(df))
        for g in groups:
            denom = int(g.mask.sum())
            row[f"BMI {g.label} kg/m2 (n={denom})"] = fmt_n_pct(int((clav[g.mask] == k).sum()), denom)
        rows.append(row)

    add_continuous_row(rows, "CCI score (median)", df, groups, df["CCI_COMPLI"], decimals=1, table2=True)
    add_binary_row(rows, "Ideal Outcome", df, groups, df["ideal_outcome_calc"], decimals=0, table2=True)
    add_binary_row(rows, "Best Performers", df, groups, df["best_performer_calc"], decimals=0, table2=True)
    add_binary_row(rows, "Endocrine insufficiency", df, groups, to_bool(df["IPENDO"]), table2=True)
    add_binary_row(rows, "Exocrine insufficiency", df, groups, df["EXOCRINE"].fillna("").astype(str).str.strip().str.upper().eq("OUI"), table2=True)

    return pd.DataFrame(rows)


def set_cell_borders(cell, *, left=False, right=False, bottom=False) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge, enabled in {"left": left, "right": right, "bottom": bottom}.items():
        tag = f"w:{edge}"
        node = borders.find(qn(tag))
        if node is None:
            node = OxmlElement(tag)
            borders.append(node)
        if enabled:
            node.set(qn("w:val"), "single")
            node.set(qn("w:sz"), "6")
            node.set(qn("w:space"), "0")
            node.set(qn("w:color"), "666666")
        else:
            node.set(qn("w:val"), "nil")


def set_cell_margins(cell, top=8, start=50, bottom=8, end=50) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for key, value in {"top": top, "start": start, "bottom": bottom, "end": end}.items():
        node = tc_mar.find(qn(f"w:{key}"))
        if node is None:
            node = OxmlElement(f"w:{key}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def set_table_width(table, widths: list[float]) -> None:
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = Inches(width)
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.first_child_found_in("w:tcW")
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(int(width * 1440)))
            tc_w.set(qn("w:type"), "dxa")


def add_text(cell, text: str, *, bold: bool = False, italic: bool = False, align=None) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1
    if align is not None:
        p.alignment = align
    run = p.add_run(str(text))
    run.font.name = "Arial"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    run.font.size = Pt(8.0)
    run.bold = bold
    run.italic = italic


def table_to_docx(doc: Document, df: pd.DataFrame, title: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(title)
    run.font.name = "Arial"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    run.font.size = Pt(10.5)
    run.bold = True

    columns = list(df.columns)
    table = doc.add_table(rows=len(df) + 1, cols=len(columns))
    table.autofit = False
    widths = [2.60, 1.25, 1.30, 1.30, 1.30, 0.85]
    set_table_width(table, widths)
    set_repeat_header(table.rows[0])

    for ci, col in enumerate(columns):
        cell = table.rows[0].cells[ci]
        header = col.replace("kg/m2", "kg/m²").replace(">=", "≥")
        add_text(cell, header, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT)

    for ri, (_, row) in enumerate(df.iterrows(), start=1):
        p_text = str(row.get("p-value", ""))
        significant = is_sig(p_text)
        section = all(str(row.get(col, "")) == "" for col in columns[1:-1]) and p_text == ""
        for ci, col in enumerate(columns):
            cell = table.rows[ri].cells[ci]
            val = str(row.get(col, ""))
            label_col = ci == 0
            p_col = col == "p-value"
            bold = section or (significant and (label_col or p_col))
            if label_col and val.startswith("  "):
                val = "    " + val.strip()
            add_text(cell, val, bold=bold, align=WD_ALIGN_PARAGRAPH.LEFT)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

    for ri, row in enumerate(table.rows):
        for ci, cell in enumerate(row.cells):
            set_cell_margins(cell)
            set_cell_borders(
                cell,
                left=(ci > 0),
                right=False,
                bottom=(ri == 0),
            )
    doc.add_paragraph()


def add_footnote(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(text)
    run.font.name = "Arial"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    run.font.size = Pt(8.5)
    run.italic = True


def make_document(table1: pd.DataFrame, table2: pd.DataFrame) -> None:
    doc = Document()
    doc.core_properties.author = "Marc-Anthony Chouillard"
    doc.core_properties.last_modified_by = "Marc-Anthony Chouillard"
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Inches(11.69)
    section.page_height = Inches(8.27)
    section.top_margin = Inches(0.55)
    section.bottom_margin = Inches(0.55)
    section.left_margin = Inches(0.55)
    section.right_margin = Inches(0.55)

    table_to_docx(
        doc,
        table1,
        "Table 1. Baseline and operative characteristics among overall population and according to Body Mass Index",
    )
    add_footnote(
        doc,
        "Footnotes: BMI Body Mass Index; ASA American Society of Anaesthesiologists physical status; "
        "MEN Multiple endocrine neoplasia; COPD Chronic obstructive pulmonary disease; DP distal pancreatectomy; "
        "RAMPS Radical antegrade modular pancreatosplenectomy; NET neuroendocrine tumor; "
        "IPMN intraductal papillary mucinous neoplasm; STP Solid pseudopapillary tumor. "
        "Continuous variables were compared using Kruskal-Wallis tests; categorical variables were compared using chi-square tests.",
    )
    doc.add_page_break()

    table_to_docx(
        doc,
        table2,
        "Table 2. Postoperative outcomes after minimally invasive distal pancreatectomy in overall population and according to Body Mass Index",
    )
    add_footnote(
        doc,
        "Footnotes: BMI Body Mass Index; POPF postoperative pancreatic fistula; PPH post-pancreatectomy hemorrhage; "
        "ICU intensive care unit; CCI Comprehensive Complication Index. Continuous variables were compared using "
        "Kruskal-Wallis tests; categorical variables were compared using chi-square tests.",
    )
    OUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT_DOCX)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = compute_outcomes(df)
    groups = build_groups(df)
    table1 = build_table1(df, groups)
    table2 = build_table2(df, groups)

    columns = ["Variable", f"Total (n={len(df)})"] + [
        f"BMI {g.label} kg/m2 (n={int(g.mask.sum())})" for g in groups
    ] + ["p-value"]
    table1 = table1[columns]
    table2 = table2[columns]

    OUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
    table1.to_csv(OUT_CSV_DIR / "main_table1_r0_format_bmi3.csv", index=False)
    table2.to_csv(OUT_CSV_DIR / "main_table2_r0_format_bmi3.csv", index=False)
    make_document(table1, table2)
    print(f"Saved {OUT_DOCX}")
    print(f"Saved {OUT_CSV_DIR / 'main_table1_r0_format_bmi3.csv'}")
    print(f"Saved {OUT_CSV_DIR / 'main_table2_r0_format_bmi3.csv'}")


if __name__ == "__main__":
    main()
