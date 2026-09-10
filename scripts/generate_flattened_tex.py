import glob as globmod
import os
import re
import subprocess
import sys

PAPERS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Papers_Tex_Source")

MAIN_FILE_PATTERNS = [
    "main.tex",
    "acl_latex.tex",
    "arxiv.tex",
    "preprint.tex",
    "root.tex",
]

MAIN_FILE_GLOBS = [
    "iclr*.tex",
    "icml*.tex",
    "ijcai*.tex",
    "neurips*.tex",
    "0.main.tex",
]


def find_main_tex(folder_path):
    for name in MAIN_FILE_PATTERNS:
        if os.path.isfile(os.path.join(folder_path, name)):
            return name

    for pattern in MAIN_FILE_GLOBS:
        matches = globmod.glob(os.path.join(folder_path, pattern))
        if matches:
            matches.sort(key=lambda p: len(os.path.basename(p)))
            return os.path.basename(matches[0])

    tex_files = [f for f in os.listdir(folder_path) if f.endswith(".tex")]
    if len(tex_files) == 1:
        return tex_files[0]

    candidates = []
    for f in tex_files:
        try:
            with open(os.path.join(folder_path, f), encoding="utf-8", errors="ignore") as fh:
                head = fh.read(4000)
            if r"\documentclass" in head:
                candidates.append(f)
        except Exception:
            pass
    if candidates:
        candidates.sort(key=len)
        return candidates[0]

    return None


def detect_bib_style(main_tex_path):
    try:
        with open(main_tex_path, encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
    except Exception:
        return None

    if re.search(r"\\usepackage\[.*backend\s*=\s*biber", content) or r"\addbibresource" in content:
        return "biber"
    if r"\bibliographystyle{" in content or r"\bibliography{" in content:
        return "bibtex"
    return None


def find_bbl_file(folder_path, main_tex):
    stem = os.path.splitext(main_tex)[0]
    bbl = os.path.join(folder_path, stem + ".bbl")
    if os.path.isfile(bbl):
        return bbl
    candidates = globmod.glob(os.path.join(folder_path, "*.bbl"))
    if candidates:
        candidates.sort(key=lambda p: len(os.path.basename(p)))
        return candidates[0]
    return None


def bbl_has_entries(bbl_path):
    try:
        with open(bbl_path, encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
        return r"\bibitem" in content
    except Exception:
        return False


def find_bib_file(folder_path):
    candidates = globmod.glob(os.path.join(folder_path, "*.bib"))
    if candidates:
        candidates.sort(key=lambda p: len(os.path.basename(p)))
        return candidates[0]
    return None


def compile_bib(folder_path, main_tex, style):
    stem = os.path.splitext(main_tex)[0]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

    if style == "biber":
        cmds = [
            ["pdflatex", "-interaction=nonstopmode", main_tex],
            ["biber", stem],
            ["pdflatex", "-interaction=nonstopmode", main_tex],
        ]
    elif style == "bibtex":
        cmds = [
            ["pdflatex", "-interaction=nonstopmode", main_tex],
            ["bibtex", stem],
            ["pdflatex", "-interaction=nonstopmode", main_tex],
        ]
    else:
        return

    for cmd in cmds:
        label = cmd[0]
        print(f"          > {label} ...", flush=True)
        try:
            result = subprocess.run(
                cmd,
                cwd=folder_path,
                timeout=180,
                env=env,
            )
            if result.returncode != 0:
                print(f"          > {label} exited with code {result.returncode}")
        except subprocess.TimeoutExpired:
            print(f"          > {label} timed out after 180s -- skipping rest")
            return
        except FileNotFoundError:
            print(f"          > {label} not found -- skipping rest")
            return


def main():
    if not os.path.isdir(PAPERS_DIR):
        print(f"ERROR: Directory not found: {PAPERS_DIR}")
        sys.exit(1)

    folders = sorted(
        d for d in os.listdir(PAPERS_DIR)
        if os.path.isdir(os.path.join(PAPERS_DIR, d))
    )

    success, skipped, failed = 0, 0, 0

    for folder in folders:
        folder_path = os.path.join(PAPERS_DIR, folder)
        main_tex = find_main_tex(folder_path)

        if main_tex is None:
            print(f"  SKIP  {folder} (no main tex file found)")
            skipped += 1
            continue

        output_file = os.path.join(folder_path, "flattened_main.tex")

        print(f"  RUN   {folder} ({main_tex})")
        try:
            bib_style = detect_bib_style(os.path.join(folder_path, main_tex))
            bbl_file = find_bbl_file(folder_path, main_tex) if bib_style else None

            if bib_style and not bbl_file:
                bib_file = find_bib_file(folder_path)
                if bib_file:
                    print(f"        No .bbl found, compiling ({bib_style})...")
                    compile_bib(folder_path, main_tex, bib_style)
                    bbl_file = find_bbl_file(folder_path, main_tex)
                else:
                    print(f"        WARNING: no .bbl or .bib found -- references will be missing")

            has_valid_bbl = bbl_file and bbl_has_entries(bbl_file)

            latexpand_cmd = ["latexpand", main_tex]
            if has_valid_bbl and bib_style == "biber":
                latexpand_cmd = ["latexpand", "--biber", bbl_file, main_tex]
            elif has_valid_bbl and bib_style == "bibtex":
                latexpand_cmd = ["latexpand", "--expand-bbl", bbl_file, main_tex]

            with open(output_file, "w", encoding="utf-8") as out:
                result = subprocess.run(
                    latexpand_cmd,
                    cwd=folder_path,
                    stdout=out,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            if result.returncode != 0:
                print(f"  FAIL  {folder} — latexpand returned {result.returncode}")
                if result.stderr.strip():
                    print(f"        {result.stderr.strip()[:200]}")
                failed += 1
            else:
                size_kb = os.path.getsize(output_file) / 1024
                ref_note = " (with references)" if has_valid_bbl else ""
                print(f"  OK    {folder} ({size_kb:.1f} KB){ref_note}")
                success += 1
        except Exception as e:
            print(f"  ERROR {folder} — {e}")
            failed += 1

    print(f"\nDone: {success} success, {skipped} skipped, {failed} failed out of {len(folders)} folders")


if __name__ == "__main__":
    main()
