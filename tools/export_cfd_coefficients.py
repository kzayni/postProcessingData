from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "cfd_coefficient_tables"
MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS = {"x": MAIN_NS}


def xml_escape(value: object) -> str:
    text = str(value)
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(node.text or "" for node in item.iter() if node.tag.endswith("}t")) for item in root]


def sheet_values(archive: ZipFile, sheet_num: int) -> dict[str, object]:
    strings = shared_strings(archive)
    root = ET.fromstring(archive.read(f"xl/worksheets/sheet{sheet_num}.xml"))
    result: dict[str, object] = {}
    for cell in root.findall(".//x:sheetData/x:row/x:c", NS):
        ref = cell.attrib["r"]
        value_node = cell.find("x:v", NS)
        if value_node is None:
            inline = cell.find("x:is", NS)
            if inline is not None:
                result[ref] = "".join(node.text or "" for node in inline.iter() if node.tag.endswith("}t"))
            continue
        raw = value_node.text or ""
        if cell.attrib.get("t") == "s":
            result[ref] = strings[int(raw)]
        else:
            try:
                result[ref] = float(raw)
            except ValueError:
                result[ref] = raw
    return result


def participant_name(path: Path) -> str:
    return path.parent.name


def submission_number(path: Path) -> int:
    match = re.search(r"D(\d+)", path.name, re.IGNORECASE)
    return int(match.group(1)) if match else 1


def extract_case(path: Path, sheet_num: int) -> dict[str, list[float | None]]:
    with ZipFile(path) as archive:
        cells = sheet_values(archive, sheet_num)
    row_for_level = {1: 3, 2: 6, 3: 9, 4: 12}
    columns = {"CL": "B", "CD": "C", "CM": "D"}
    return {
        coefficient: [
            cells.get(f"{column}{row_for_level[level]}")
            if isinstance(cells.get(f"{column}{row_for_level[level]}"), float)
            else None
            for level in range(1, 5)
        ]
        for coefficient, column in columns.items()
    }


def cell(ref: str, value: object | None, style: int = 0) -> str:
    if value is None:
        return f'<c r="{ref}" s="{style}"/>'
    if isinstance(value, (int, float)):
        return f'<c r="{ref}" s="{style}" t="n"><v>{value:.15g}</v></c>'
    return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t>{xml_escape(value)}</t></is></c>'


def worksheet_xml(case_name: str, coefficient: str, rows: list[tuple[str, list[float | None]]]) -> str:
    last_row = len(rows) + 3
    data_rows = [
        '<row r="1" ht="28" customHeight="1">'
        + cell("A1", f"{case_name} — {coefficient} Grid Convergence", 1)
        + "</row>",
        '<row r="2" ht="8" customHeight="1"/>',
        '<row r="3" ht="22" customHeight="1">'
        + "".join(cell(f"{col}3", value, 2) for col, value in zip("ABCDE", ["Participant", "L1", "L2", "L3", "L4"]))
        + "</row>",
    ]
    for index, (participant, values) in enumerate(rows, start=4):
        style = 4 if index % 2 == 0 else 5
        parts = [cell(f"A{index}", participant, style)]
        parts.extend(cell(f"{col}{index}", value, 6 if index % 2 == 0 else 7) for col, value in zip("BCDE", values))
        data_rows.append(f'<row r="{index}" ht="20" customHeight="1">' + "".join(parts) + "</row>")
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="{MAIN_NS}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <dimension ref="A1:E{last_row}"/>
  <sheetViews><sheetView showGridLines="0" workbookViewId="0"><pane ySplit="3" topLeftCell="A4" activePane="bottomLeft" state="frozen"/><selection pane="bottomLeft" activeCell="A4" sqref="A4"/></sheetView></sheetViews>
  <sheetFormatPr defaultRowHeight="18"/>
  <cols><col min="1" max="1" width="34" customWidth="1"/><col min="2" max="5" width="15" customWidth="1"/></cols>
  <sheetData>{''.join(data_rows)}</sheetData>
  <autoFilter ref="A3:E{last_row}"/>
  <mergeCells count="1"><mergeCell ref="A1:E1"/></mergeCells>
  <pageMargins left="0.3" right="0.3" top="0.5" bottom="0.5" header="0.2" footer="0.2"/>
  <pageSetup orientation="landscape" fitToWidth="1" fitToHeight="1"/>
</worksheet>'''


def styles_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <numFmts count="1"><numFmt numFmtId="164" formatCode="0.00000000;-0.00000000;0.00000000"/></numFmts>
  <fonts count="3">
    <font><sz val="10"/><color rgb="FF243447"/><name val="Aptos"/><family val="2"/></font>
    <font><b/><sz val="16"/><color rgb="FFFFFFFF"/><name val="Aptos Display"/><family val="2"/></font>
    <font><b/><sz val="10"/><color rgb="FFFFFFFF"/><name val="Aptos"/><family val="2"/></font>
  </fonts>
  <fills count="5">
    <fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF17365D"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF3F6F99"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFF1F5F9"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2"><border/><border><left style="thin"><color rgb="FFD6DEE8"/></left><right style="thin"><color rgb="FFD6DEE8"/></right><top style="thin"><color rgb="FFD6DEE8"/></top><bottom style="thin"><color rgb="FFD6DEE8"/></bottom><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="8">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
    <xf numFmtId="0" fontId="2" fillId="3" borderId="1" xfId="0" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="0" fillId="4" borderId="1" xfId="0" applyAlignment="1"><alignment horizontal="left" vertical="center"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyAlignment="1"><alignment horizontal="left" vertical="center"/></xf>
    <xf numFmtId="164" fontId="0" fillId="4" borderId="1" xfId="0" applyNumberFormat="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>
    <xf numFmtId="164" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''


def write_workbook(path: Path, case_name: str, dataset: dict[str, list[tuple[str, list[float | None]]]]) -> None:
    sheets = ["CL", "CD", "CM"]
    content_types = ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1, 4))
    workbook_sheets = ''.join(f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>' for i, name in enumerate(sheets, 1))
    rels = ''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, 4))
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>{content_types}<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/><Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/></Types>''')
        archive.writestr("_rels/.rels", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/></Relationships>''')
        archive.writestr("xl/workbook.xml", f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="{MAIN_NS}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><bookViews><workbookView activeTab="0"/></bookViews><sheets>{workbook_sheets}</sheets><calcPr calcId="191029" fullCalcOnLoad="1"/></workbook>''')
        archive.writestr("xl/_rels/workbook.xml.rels", f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rels}<Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>''')
        archive.writestr("xl/styles.xml", styles_xml())
        for index, coefficient in enumerate(sheets, 1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", worksheet_xml(case_name, coefficient, dataset[coefficient]))
        archive.writestr("docProps/core.xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>CFD coefficient grid convergence summary</dc:title><dc:creator>OpenAI</dc:creator></cp:coreProperties>''')
        archive.writestr("docProps/app.xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>Microsoft Excel</Application></Properties>''')


def main() -> None:
    candidates = sorted(p for p in ROOT.glob("*/gridConvergence*.xlsx") if not p.name.startswith("~$"))
    latest_by_participant: dict[str, Path] = {}
    for path in candidates:
        key = participant_name(path)
        if key not in latest_by_participant or submission_number(path) > submission_number(latest_by_participant[key]):
            latest_by_participant[key] = path
    paths = [latest_by_participant[key] for key in sorted(latest_by_participant)]
    cases = {"NACA0012": 2, "ONERA M6": 3}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for case_name, sheet_num in cases.items():
        tables = {coefficient: [] for coefficient in ("CL", "CD", "CM")}
        for path in paths:
            extracted = extract_case(path, sheet_num)
            participant = participant_name(path)
            # Boeing supplied its L2 dataset in the D02 convergence workbook,
            # with the coefficients entered in that file's first level block.
            if participant == "013_BOEING_GLENNICE" and case_name == "NACA0012" and submission_number(path) == 2:
                d01_path = path.with_name("gridConvergence_D01_V1.xlsx")
                d01 = extract_case(d01_path, sheet_num) if d01_path.exists() else {key: [None] * 4 for key in extracted}
                for coefficient in extracted:
                    d02_l2 = extracted[coefficient][0]
                    extracted[coefficient] = [d01[coefficient][0], d02_l2, None, None]
            for coefficient, values in extracted.items():
                tables[coefficient].append((participant, values))
        filename = case_name.lower().replace(" ", "_") + "_cfd_coefficients.xlsx"
        write_workbook(OUTPUT / filename, case_name, tables)
        print(OUTPUT / filename)


if __name__ == "__main__":
    main()
