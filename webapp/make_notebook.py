"""Regenerate ResT_webapp_Colab.ipynb from cleanup.py and app.py.

Run after editing either file so the notebook's %%writefile cells stay in sync:

    python webapp/make_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

WEBAPP_DIR = Path(__file__).resolve().parent
NOTEBOOK = WEBAPP_DIR / "ResT_webapp_Colab.ipynb"
CHECKPOINT_REL = "checkpoints/rest_gpds_32pairs_10epochs_best.pt"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


def writefile(name: str) -> dict:
    body = (WEBAPP_DIR / name).read_text(encoding="utf-8")
    return code(f"%%writefile /content/ResT/webapp/{name}\n{body}")


def build() -> dict:
    cells = [
        markdown("""
# ResT signature scanner & verifier (Colab)

Runtime → Change runtime type → **T4 GPU**, then run all cells. The last cell prints a
`https://….gradio.live` link: open it on your phone (the camera needs HTTPS). The link
lasts until this session stops (max 72 h).

**Mock-test use only**: ~75% accuracy on GPDS, and phone photos are out of domain.
"""),
        code("!nvidia-smi"),
        markdown("## 1. Clone the repo (code only; LFS files are skipped)"),
        code("""
%cd /content
!rm -rf /content/ResT
!GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/buixuanhieu1999/ResT.git /content/ResT
!mkdir -p /content/ResT/webapp
%cd /content/ResT
"""),
        markdown("## 2. Write the web app files\nGenerated from `webapp/` by `make_notebook.py`, so this works even before `webapp/` is pushed."),
        writefile("cleanup.py"),
        writefile("app.py"),
        markdown(f"""
## 3. Get the checkpoint (~388 MB)
Uses `MyDrive/ResT/{CHECKPOINT_REL}` from Google Drive if present (fast), otherwise pulls
just that file with Git LFS.
"""),
        code(f"""
import os, subprocess
from pathlib import Path

drive_ckpt = Path("/content/drive/MyDrive/ResT/{CHECKPOINT_REL}")
repo_ckpt = Path("/content/ResT/{CHECKPOINT_REL}")
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as error:
    print("Drive not mounted:", error)

if drive_ckpt.is_file() and drive_ckpt.stat().st_size > 10_000:
    ckpt = drive_ckpt
    print("Using checkpoint from Google Drive")
else:
    print("Not on Drive; pulling from Git LFS ...")
    subprocess.run("apt-get -qq install -y git-lfs > /dev/null && git lfs install --skip-smudge", shell=True, check=True)
    subprocess.run(["git", "lfs", "pull", "--include", "{CHECKPOINT_REL}"], cwd="/content/ResT", check=True)
    ckpt = repo_ckpt

size = ckpt.stat().st_size
print(f"{{ckpt}}: {{size / 1e6:.1f}} MB")
assert size > 10_000, "Still an LFS pointer file: the LFS download failed (bandwidth quota?)."
os.environ["REST_CKPT"] = str(ckpt)
"""),
        markdown("## 4. Install Gradio"),
        code("!pip install -q gradio"),
        markdown("## 5. Launch\nOpen the `https://….gradio.live` link on your phone. Stop this cell to shut down."),
        code("!cd /content/ResT && python webapp/app.py --share"),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    NOTEBOOK.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {NOTEBOOK}")
