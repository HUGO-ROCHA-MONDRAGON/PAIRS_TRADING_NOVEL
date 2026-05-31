from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--kernel", default="python3")
    args = parser.parse_args()
    nb = nbformat.read(args.notebook, as_version=4)
    client = NotebookClient(nb, timeout=args.timeout, kernel_name=args.kernel)
    client.execute()
    out = args.notebook.with_suffix(".executed.ipynb")
    nbformat.write(nb, out)
    print(out)


if __name__ == "__main__":
    main()
