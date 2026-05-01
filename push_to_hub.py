"""Rename HF repos, push updated model artifacts and Space to Hugging Face Hub."""

import getpass
import subprocess
from huggingface_hub import HfApi

OLD_MODEL_REPO = "OmarGamal48812/flickr-captioning"
OLD_SPACE_REPO = "OmarGamal48812/flickr-captioning-demo"
NEW_MODEL_REPO = "OmarGamal48812/flickr-captioning"
NEW_SPACE_REPO = "OmarGamal48812/flickr-captioning-demo"

MODEL_FILES = [
    ("models/attention_gru_glove.pth", "attention_gru_glove.pth"),
    ("data/processed/vocab.pkl",        "vocab.pkl"),
    ("hf_release/config.json",          "config.json"),
    ("hf_release/metrics_beam5.json",   "metrics_beam5.json"),
    ("hf_release/README.md",            "README.md"),
]

SPACE_FILES = [
    ("hf_space/app.py",              "app.py"),
    ("hf_space/README.md",           "README.md"),
    ("hf_space/requirements.txt",    "requirements.txt"),
    ("hf_space/src/__init__.py",     "src/__init__.py"),
    ("hf_space/src/attention.py",    "src/attention.py"),
    ("hf_space/src/decoder.py",      "src/decoder.py"),
    ("hf_space/src/encoder.py",      "src/encoder.py"),
    ("hf_space/src/inference.py",    "src/inference.py"),
    ("hf_space/src/utils.py",        "src/utils.py"),
    ("hf_space/src/visualize.py",    "src/visualize.py"),
    ("hf_space/src/vocabulary.py",   "src/vocabulary.py"),
]

MODEL_FILE_TO_DELETE = "attention_lstm.pth"

FILES_TO_PATCH = [
    "hf_release/README.md",
    "hf_space/README.md",
    "hf_space/app.py",
    "push_to_hub.py",
    "README.md",
]


def push_files(api, files, repo_id, repo_type, token):
    for local_path, repo_path in files:
        print(f"  {local_path} → {repo_path}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
        )
    print("  All files uploaded.")


def patch_local_files():
    print("\n=== Updating local repo ID references ===")
    for path in FILES_TO_PATCH:
        try:
            with open(path) as f:
                content = f.read()
            new_content = (
                content
                .replace(OLD_MODEL_REPO, NEW_MODEL_REPO)
                .replace(OLD_SPACE_REPO, NEW_SPACE_REPO)
                .replace("flickr-captioning", "flickr-captioning")
                .replace("flickr-captioning-demo", "flickr-captioning-demo")
            )
            if new_content != content:
                with open(path, "w") as f:
                    f.write(new_content)
                print(f"  Patched {path}")
        except FileNotFoundError:
            print(f"  Skipped {path} (not found)")


def main():
    token = getpass.getpass("Hugging Face token (write scope): ")
    api = HfApi()

    # --- Step 1: Rename repos ---
    print(f"\n=== Renaming model repo: {OLD_MODEL_REPO} → {NEW_MODEL_REPO} ===")
    try:
        api.move_repo(
            from_id=OLD_MODEL_REPO,
            to_id=NEW_MODEL_REPO,
            repo_type="model",
            token=token,
        )
        print("  OK")
    except Exception as e:
        print(f"  Skipped ({e})")

    print(f"\n=== Renaming Space: {OLD_SPACE_REPO} → {NEW_SPACE_REPO} ===")
    try:
        api.move_repo(
            from_id=OLD_SPACE_REPO,
            to_id=NEW_SPACE_REPO,
            repo_type="space",
            token=token,
        )
        print("  OK")
    except Exception as e:
        print(f"  Skipped ({e})")

    # --- Step 2: Patch local files ---
    patch_local_files()

    # --- Step 3: Push model files ---
    print(f"\n=== Pushing model repo: {NEW_MODEL_REPO} ===")
    push_files(api, MODEL_FILES, NEW_MODEL_REPO, "model", token)

    print(f"\nDeleting {MODEL_FILE_TO_DELETE} from model repo ...")
    try:
        api.delete_file(
            path_in_repo=MODEL_FILE_TO_DELETE,
            repo_id=NEW_MODEL_REPO,
            repo_type="model",
            token=token,
        )
        print("  OK")
    except Exception as e:
        print(f"  Skipped ({e})")

    # --- Step 4: Push Space files ---
    print(f"\n=== Pushing Space: {NEW_SPACE_REPO} ===")
    push_files(api, SPACE_FILES, NEW_SPACE_REPO, "space", token)

    # --- Step 5: Commit patched local files to git ---
    print("\n=== Committing updated repo ID references to git ===")
    subprocess.run(["git", "add"] + FILES_TO_PATCH, check=True)
    subprocess.run([
        "git", "commit", "-m",
        "chore: rename HF repos from flickr8k to flickr-captioning"
    ], check=True)
    subprocess.run(["git", "push", "origin", "main"], check=True)

    print("\nAll done.")


if __name__ == "__main__":
    main()
