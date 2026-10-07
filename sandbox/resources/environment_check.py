#!/opt/agent-venv/bin/python
"""Offline functional checks for the sandbox image (ported from finautorubric)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile


TOKEN = "ENV_CHECK_314159"
DIRECT_PACKAGES = (
    "pip", "requests", "beautifulsoup4", "lxml", "numpy", "pandas", "numpy-financial",
    "pypdf", "PyPDF2", "pdfminer.six", "pdfplumber", "Pillow", "PyMuPDF",
)


class CheckFailure(RuntimeError):
    pass


def run(argv: list[str], *, env: dict[str, str] | None = None) -> str:
    process = subprocess.run(argv, text=True, capture_output=True, env=env, check=False)
    if process.returncode:
        detail = (process.stderr or process.stdout).strip()[-800:]
        error = CheckFailure(f"{argv[0]} exited {process.returncode}: {detail}")
        error.exit_code = process.returncode
        raise error
    return process.stdout


def synthetic_pdf(path: Path) -> None:
    stream = f"BT /F1 18 Tf 72 720 Td ({TOKEN}) Tj ET\n".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    data = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    data.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n".encode()
    )
    path.write_bytes(data)


def cli_and_shell(root: Path) -> dict:
    sample = root / "sample.json"
    sample.write_text('{"a":1,"b":[2,3]}\n')
    archive = root / "sample.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("sample.txt", TOKEN)
    run(["jq", "-e", ".a == 1 and .b[1] == 3", str(sample)])
    run(["ruby", "-rjson", "-e", "abort unless JSON.parse(File.read(ARGV[0]))['b'][1] == 3", str(sample)])
    if "JSON" not in run(["file", str(sample)]) and "text" not in run(["file", str(sample)]):
        raise CheckFailure("file did not recognize the JSON fixture")
    if TOKEN not in run(["strings", str(archive)]):
        raise CheckFailure("strings did not preserve the fixture token")
    if not run(["xxd", "-p", str(sample)]).strip():
        raise CheckFailure("xxd returned no hex output")
    if run(["unzip", "-p", str(archive), "sample.txt"]).strip() != TOKEN:
        raise CheckFailure("unzip content mismatch")
    run(["ps", "-p", str(os.getpid())])
    run(["pgrep", "-f", "environment_check.py"])
    run(["/bin/bash", "-lc", "jq -n -e '1 == 1' >/dev/null && ruby -e 'exit 0'"])
    return {"commands": ["jq", "ruby", "file", "strings", "xxd", "unzip", "ps", "pgrep"]}


def interpreters(_: Path) -> dict:
    prefix = Path("/opt/agent-venv").resolve()
    if Path(sys.prefix).resolve() != prefix:
        raise CheckFailure(f"unexpected interpreter prefix: {sys.prefix}")
    paths = {}
    for name in ("python", "python3", "pip", "pip3"):
        path = shutil.which(name)
        if path is None:
            raise CheckFailure(f"{name} is not on PATH")
        paths[name] = path
    probe = "import pathlib,sys; assert pathlib.Path(sys.prefix).resolve() == pathlib.Path('/opt/agent-venv')"
    run(["python", "-c", probe])
    run(["python3", "-c", probe])
    run(["/bin/bash", "-lc", "python -c \"" + probe + "\" && python3 -c \"" + probe + "\" && pip --version && pip3 --version"])
    run(["python", "-m", "pip", "--version"])
    marker = prefix / ".runtime-write-probe"
    try:
        marker.write_text("must not be writable")
    except OSError:
        pass
    else:
        marker.unlink(missing_ok=True)
        raise CheckFailure("base venv is writable at runtime")
    return {"executable": sys.executable, "prefix": sys.prefix, "paths": paths}


def python_dependencies(_: Path) -> dict:
    import bs4
    import fitz
    import lxml.etree
    import numpy
    import numpy_financial
    import pandas
    import pdfminer.high_level
    import pdfplumber
    import PIL.Image
    import PIL._imaging
    import pymupdf
    import PyPDF2
    import pypdf
    import requests

    del bs4, fitz, lxml, numpy, numpy_financial, pandas, pdfminer, pdfplumber
    del PIL, pymupdf, PyPDF2, pypdf, requests
    run(["python", "-m", "pip", "check"])
    return {name: importlib.metadata.version(name) for name in DIRECT_PACKAGES}


def pdf_text(root: Path) -> dict:
    from pdfminer.high_level import extract_text
    import pdfplumber
    import PyPDF2
    import pypdf

    pdf = root / "synthetic.pdf"
    synthetic_pdf(pdf)
    extracted = root / "pdftotext.txt"
    run(["pdftotext", str(pdf), str(extracted)])
    texts = {
        "pdftotext": extracted.read_text(errors="replace"),
        "pypdf": pypdf.PdfReader(str(pdf)).pages[0].extract_text(),
        "PyPDF2": PyPDF2.PdfReader(str(pdf)).pages[0].extract_text(),
        "pdfminer": extract_text(str(pdf)),
    }
    with pdfplumber.open(pdf) as document:
        texts["pdfplumber"] = document.pages[0].extract_text() or ""
    missing = [name for name, text in texts.items() if TOKEN not in text]
    if missing:
        raise CheckFailure("PDF token missing from: " + ", ".join(missing))
    run(["pdfinfo", str(pdf)])
    return {"extractors": sorted(texts), "token_sha256": hashlib.sha256(TOKEN.encode()).hexdigest()}


def pdf_render(root: Path) -> dict:
    from PIL import Image
    import pymupdf

    pdf = root / "synthetic.pdf"
    if not pdf.exists():
        synthetic_pdf(pdf)
    prefix = root / "poppler-page"
    run(["pdftoppm", "-f", "1", "-singlefile", "-png", str(pdf), str(prefix)])
    with Image.open(prefix.with_suffix(".png")) as image:
        poppler_size = image.size
        image.verify()
    with pymupdf.open(pdf) as document:
        pixmap = document[0].get_pixmap()
        native_size = (pixmap.width, pixmap.height)
        rendered = root / "pymupdf-page.png"
        pixmap.save(rendered)
    with Image.open(rendered) as image:
        image.verify()
    if min(*poppler_size, *native_size) < 100:
        raise CheckFailure("rendered page is unexpectedly small")
    return {"pdftoppm_pixels": poppler_size, "pymupdf_pixels": native_size}


def html_xml(_: Path) -> dict:
    from bs4 import BeautifulSoup
    from lxml import etree
    import pandas as pd

    html = "<html><table><tr><th>value</th></tr><tr><td>17</td></tr><tr><td>25</td></tr></table></html>"
    xml = "<root><value>42</value></root>"
    if BeautifulSoup(html, "html.parser").find("td").text != "17":
        raise CheckFailure("BeautifulSoup parse mismatch")
    if etree.fromstring(xml.encode()).findtext("value") != "42":
        raise CheckFailure("lxml parse mismatch")
    table = pd.read_html(io.StringIO(html))[0]
    if int(table["value"].sum()) != 42:
        raise CheckFailure("pandas HTML table mismatch")
    return {"table_sum": 42, "xml_value": 42}


def financial(_: Path) -> dict:
    import numpy_financial

    value = float(numpy_financial.irr([-100.0, 110.0]))
    if abs(value - 0.1) > 1e-9:
        raise CheckFailure(f"unexpected IRR: {value}")
    return {"irr": value}


def scratch(root: Path, expected_tmp: Path) -> dict:
    expected = str(expected_tmp)
    if os.environ.get("TMPDIR") != expected:
        raise CheckFailure("TMPDIR policy mismatch")
    if Path(tempfile.gettempdir()) != expected_tmp:
        raise CheckFailure("Python tempfile does not use TMPDIR")
    if os.getuid() != 1000:
        raise CheckFailure(f"sandbox UID is {os.getuid()}, expected 1000")
    tmp_mode = Path("/tmp").stat().st_mode
    if not tmp_mode & stat.S_ISVTX:
        raise CheckFailure("/tmp has no sticky bit")
    expected_tmp.mkdir(parents=True, exist_ok=True)
    probes = [Path("/tmp") / f"direct-{os.getpid()}.bin", expected_tmp / "scoped.bin"]
    for index, probe in enumerate(probes):
        payload = (f"scratch-{index}-" + TOKEN).encode()
        probe.write_bytes(payload)
        if probe.read_bytes() != payload:
            raise CheckFailure("scratch content mismatch")
        probe.unlink()
    disk = os.statvfs(expected_tmp)
    return {
        "root": expected,
        "mount": "anonymous_volume",
        "free_bytes": disk.f_bavail * disk.f_frsize,
        "free_inodes": disk.f_favail,
        "tmp_mode": oct(stat.S_IMODE(tmp_mode)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-tmp", type=Path, required=True)
    args = parser.parse_args()
    args.expected_tmp.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="environment-check-", dir=args.expected_tmp))
    checks = []
    functions = (
        ("cli_shell", cli_and_shell),
        ("interpreter", interpreters),
        ("python_dependencies", python_dependencies),
        ("pdf_text", pdf_text),
        ("pdf_render", pdf_render),
        ("html_xml", html_xml),
        ("financial", financial),
        ("scratch", lambda directory: scratch(directory, args.expected_tmp)),
    )
    try:
        for check_id, function in functions:
            try:
                detail = function(root)
                checks.append({"check_id": check_id, "passed": True, "detail": detail})
            except Exception as exc:
                checks.append({
                    "check_id": check_id,
                    "passed": False,
                    "dependency": type(exc).__name__,
                    "exit_code": getattr(exc, "exit_code", None),
                    "stderr": str(exc)[-1000:],
                })
    finally:
        shutil.rmtree(root, ignore_errors=True)
    report = {
        "schema_version": "sandbox.environment-check/v1",
        "ready": all(item["passed"] for item in checks),
        "architecture": os.uname().machine,
        "python": sys.version.split()[0],
        "checks": checks,
        "model_calls": 0,
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
