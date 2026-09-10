from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import shutil
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
PDF_ROOT = ROOT / "Papers_pdf"
TEX_ROOT = ROOT / "Papers_Tex_Source"
USER_AGENT = "Core499ArxivDownloader/1.0 (+https://arxiv.org)"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_ID_RE = re.compile(
    r"(?P<id>(?:[a-z\-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?)",
    flags=re.I,
)


def read_manifest(path: Path) -> list[str]:
    entries: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


def parse_arxiv_id(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Empty arXiv identifier")

    url_match = re.search(r"arxiv\.org/(?:abs|pdf|e-print)/([^?\s#]+)", value, flags=re.I)
    if url_match:
        value = url_match.group(1)

    value = value.removesuffix(".pdf").strip("/")
    match = ARXIV_ID_RE.search(value)
    if not match:
        raise ValueError(f"Could not parse an arXiv ID from: {value}")
    return match.group("id")


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request) as response:
        return response.read()


def fetch_metadata(arxiv_id: str) -> dict[str, object]:
    api_url = f"https://export.arxiv.org/api/query?id_list={quote(arxiv_id)}"
    data = fetch_bytes(api_url)
    root = ET.fromstring(data)
    entry = root.find("atom:entry", ATOM_NS)
    if entry is None:
        raise RuntimeError(f"No arXiv metadata returned for {arxiv_id}")

    title = " ".join((entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").split())
    summary = " ".join((entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").split())
    entry_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS) or ""
    authors = [
        " ".join((author.findtext("atom:name", default="", namespaces=ATOM_NS) or "").split())
        for author in entry.findall("atom:author", ATOM_NS)
    ]
    resolved_id = entry_id.rstrip("/").split("/")[-1] if entry_id else arxiv_id
    return {
        "requested_id": arxiv_id,
        "arxiv_id": resolved_id,
        "title": title or arxiv_id,
        "summary": summary,
        "authors": authors,
        "pdf_url": f"https://arxiv.org/pdf/{resolved_id}.pdf",
        "source_url": f"https://arxiv.org/e-print/{resolved_id}",
        "abstract_url": f"https://arxiv.org/abs/{resolved_id}",
    }


def normalize_title_for_path(title: str) -> str:
    text = title.upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def discover_existing_prefixes() -> set[int]:
    prefixes: set[int] = set()
    for base in (PDF_ROOT, TEX_ROOT):
        if not base.exists():
            continue
        for path in base.iterdir():
            match = re.match(r"^(\d+)\s", path.name)
            if match:
                prefixes.add(int(match.group(1)))
    return prefixes


def find_existing_entry(arxiv_id: str) -> tuple[Path | None, Path | None]:
    pdf_match = next(iter(PDF_ROOT.glob(f"*{arxiv_id}*.pdf")), None)
    tex_match = next(iter(TEX_ROOT.glob(f"*{arxiv_id}*")), None)
    return pdf_match, tex_match


def next_prefix(start_index: int | None = None) -> int:
    used = discover_existing_prefixes()
    if start_index is not None:
        current = start_index
    else:
        current = 1
    while current in used:
        current += 1
    return current


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def flatten_single_top_level_directory(target_dir: Path) -> None:
    children = [path for path in target_dir.iterdir() if path.name != "00README.json"]
    dirs = [path for path in children if path.is_dir()]
    files = [path for path in children if path.is_file()]
    if files or len(dirs) != 1:
        return

    inner_dir = dirs[0]
    for child in list(inner_dir.iterdir()):
        destination = target_dir / child.name
        if destination.exists():
            continue
        child.rename(destination)
    inner_dir.rmdir()


def extract_source_archive(raw: bytes, target_dir: Path) -> None:
    safe_mkdir(target_dir)

    archive = io.BytesIO(raw)
    if zipfile.is_zipfile(archive):
        archive.seek(0)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target_dir)
        flatten_single_top_level_directory(target_dir)
        return

    archive.seek(0)
    try:
        with tarfile.open(fileobj=archive, mode="r:*") as tf:
            tf.extractall(target_dir, filter="data")
        flatten_single_top_level_directory(target_dir)
        return
    except tarfile.TarError:
        pass

    if raw[:2] == b"\x1f\x8b":
        decompressed = gzip.decompress(raw)
        tar_candidate = io.BytesIO(decompressed)
        try:
            with tarfile.open(fileobj=tar_candidate, mode="r:") as tf:
                tf.extractall(target_dir, filter="data")
            flatten_single_top_level_directory(target_dir)
            return
        except tarfile.TarError:
            (target_dir / "source.tex").write_bytes(decompressed)
            return

    (target_dir / "source.tex").write_bytes(raw)


def write_metadata_file(target_dir: Path, metadata: dict[str, object], pdf_path: Path) -> None:
    payload = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": metadata["title"],
        "arxiv_id": metadata["arxiv_id"],
        "requested_id": metadata["requested_id"],
        "authors": metadata["authors"],
        "abstract_url": metadata["abstract_url"],
        "pdf_url": metadata["pdf_url"],
        "source_url": metadata["source_url"],
        "pdf_filename": pdf_path.name,
    }
    (target_dir / "00README.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def download_one(arxiv_id: str, prefix: int, dry_run: bool = False) -> dict[str, str]:
    metadata = fetch_metadata(arxiv_id)
    resolved_id = str(metadata["arxiv_id"])
    existing_pdf, existing_tex = find_existing_entry(resolved_id)
    if existing_pdf or existing_tex:
        return {
            "status": "skipped",
            "arxiv_id": resolved_id,
            "pdf": str(existing_pdf or ""),
            "tex": str(existing_tex or ""),
            "reason": "already exists",
        }

    title_slug = normalize_title_for_path(str(metadata["title"]))
    prefix_str = f"{prefix:02d}"
    pdf_path = PDF_ROOT / f"{prefix_str}_{title_slug}_{resolved_id}.pdf"
    tex_dir = TEX_ROOT / f"{prefix_str}_{title_slug}_{resolved_id}_TeX_Source"

    if dry_run:
        return {
            "status": "planned",
            "arxiv_id": resolved_id,
            "pdf": str(pdf_path),
            "tex": str(tex_dir),
            "reason": "dry-run",
        }

    safe_mkdir(PDF_ROOT)
    safe_mkdir(TEX_ROOT)
    safe_mkdir(tex_dir)

    pdf_bytes = fetch_bytes(str(metadata["pdf_url"]))
    pdf_path.write_bytes(pdf_bytes)

    source_bytes = fetch_bytes(str(metadata["source_url"]))
    extract_source_archive(source_bytes, tex_dir)
    write_metadata_file(tex_dir, metadata, pdf_path)

    return {
        "status": "downloaded",
        "arxiv_id": resolved_id,
        "pdf": str(pdf_path),
        "tex": str(tex_dir),
        "reason": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download arXiv PDFs and TeX sources into the Core499 paper layout.")
    parser.add_argument("papers", nargs="*", help="arXiv IDs or arXiv URLs")
    parser.add_argument("--manifest", type=Path, help="Text file with one arXiv ID or URL per line")
    parser.add_argument("--start-index", type=int, default=None, help="Optional starting numeric prefix")
    parser.add_argument("--dry-run", action="store_true", help="Plan filenames and folders without downloading")
    args = parser.parse_args()

    entries = list(args.papers)
    if args.manifest:
        entries.extend(read_manifest(args.manifest.resolve()))
    if not entries:
        raise SystemExit("Provide arXiv IDs/URLs or a --manifest file.")

    normalized_ids: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        arxiv_id = parse_arxiv_id(entry)
        if arxiv_id in seen:
            continue
        seen.add(arxiv_id)
        normalized_ids.append(arxiv_id)

    current_prefix = next_prefix(args.start_index)
    for arxiv_id in normalized_ids:
        result = download_one(arxiv_id, current_prefix, dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False))
        if result["status"] != "skipped":
            current_prefix += 1


if __name__ == "__main__":
    main()
