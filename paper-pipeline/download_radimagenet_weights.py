import argparse
import os
import shutil

from huggingface_hub import hf_hub_download

REPO_ID = "Lab-Rasool/RadImageNet"
FILENAME = "InceptionV3.pt"

DEFAULT_DEST = "" # set before running


def download_radimagenet_weights(dest=DEFAULT_DEST):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    cached_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    shutil.copy(cached_path, dest)
    print(f"Saved RadImageNet InceptionV3 weights to {dest}")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default=DEFAULT_DEST,
                         help="Where to save the weights file (default: BASE['rad_model_path'] "
                              "in diff_experiments.py).")
    args = parser.parse_args()
    download_radimagenet_weights(args.dest)
