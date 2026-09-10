import os

ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)))

EXTENSIONS = [
    ".aux", ".blg", ".fdb_latexmk", ".fls", ".log", ".out",
    ".synctex.gz", ".toc", ".lof", ".lot",
    ".nav", ".snm", ".vrb", ".bcf", ".run.xml", ".idx",
    ".ilg", ".ind", ".glg", ".glo", ".gls", ".ist",
    ".acn", ".acr", ".alg", ".xdy", ".dvi", ".ps", ".eps",
]


def main():
    count = 0
    for dirpath, _, filenames in os.walk(ROOT):
        if ".git" in dirpath or ".venv" in dirpath:
            continue
        for name in filenames:
            if any(name.endswith(ext) for ext in EXTENSIONS):
                path = os.path.join(dirpath, name)
                os.remove(path)
                count += 1
                print(f"  deleted {os.path.relpath(path, ROOT)}")

    print(f"\nCleaned {count} files")


if __name__ == "__main__":
    main()
