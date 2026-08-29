from pathlib import Path
from zipfile import ZipFile
import re
import sys
import xml.etree.ElementTree as ET

NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def shared_strings(archive: ZipFile):
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(node.text or "" for node in item.iter() if node.tag.endswith("}t")) for item in root]


def dump(path: Path):
    with ZipFile(path) as archive:
        strings = shared_strings(archive)
        for sheet_num in (2, 3):
            print(f"\n--- sheet{sheet_num} ---")
            root = ET.fromstring(archive.read(f"xl/worksheets/sheet{sheet_num}.xml"))
            for row in root.findall(".//x:sheetData/x:row", NS):
                values = []
                for cell in row.findall("x:c", NS):
                    ref = cell.attrib["r"]
                    value_node = cell.find("x:v", NS)
                    if value_node is None:
                        continue
                    value = value_node.text or ""
                    if cell.attrib.get("t") == "s":
                        value = strings[int(value)]
                    values.append(f"{ref}={value}")
                if values:
                    print(" | ".join(values))


dump(Path(sys.argv[1]))
