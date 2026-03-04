
from huggingface_hub import HfApi, upload_folder
import os
import argparse

def create_and_upload(repo_id: str,
                      local_path: str,
                      repo_type: str = "model",
                      private: bool = False):
    """
    Create a Hugging Face repo (if needed) and upload a local folder.

    Args:
        repo_id: "<namespace>/<name>", e.g. "jiebi/RFC-DR-Align-7B".
        local_path: path to the local folder to upload.
        repo_type: "model" or "dataset".
        private: whether to create a private repo.
        token: HF access token (if not provided, uses env HUGGINGFACE_TOKEN).
    """

    api = HfApi()

    # 1) Create the repo if it does not exist
    #    exist_ok=True makes this idempotent
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )

    # 2) Upload an entire folder (keeps paths relative to local_path)
    #    By default this pushes to the "main" branch and commits in one shot.
    commit_info = upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=local_path,
        #commit_message=f"Upload from {os.path.abspath(local_path)}",
        ignore_patterns=[
            ".git",
            ".gitignore",
            "__pycache__",
            "*.pyc",
            "*.pyo",
            "*.tmp",
            "*.ipynb_checkpoints",
            ".DS_Store",
            "global_step*",  # Training step directories
            "checkpoint-*",  # Other checkpoint directories if needed
            "runs/",  # Tensorboard logs
            "*.log",  # Log files
            "*.swp",  # Vim swap files
            "*~",  # Backup files
            "latest",  # Symlink/pointer to latest checkpoint
            "rng_state*.pth",  # Random number generator state
            "optimizer.pt",  # Optimizer state (large and not needed for inference)
            "scheduler.pt",  # Scheduler state
            "trainer_state.json",  # Training state
        ],
    )

    print("Upload complete.")
    print("Commit info:", commit_info)
    print("Repo URL:", f"https://huggingface.co/{repo_id}" if repo_type == "model"
          else f"https://huggingface.co/datasets/{repo_id}")

def rename_repo(old_repo_id: str, new_repo_id: str, repo_type: str = "model"):
    """
    Rename an existing HuggingFace repository.

    Args:
        old_repo_id: Current repo name, e.g. "jiebi/OldName".
        new_repo_id: New repo name, e.g. "jiebi/NewName".
        repo_type: "model" or "dataset".
    """
    api = HfApi()
    api.move_repo(
        from_id=old_repo_id,
        to_id=new_repo_id,
        repo_type=repo_type
    )
    print(f"Renamed {old_repo_id} -> {new_repo_id}")
    print("New URL:", f"https://huggingface.co/{new_repo_id}" if repo_type == "model"
          else f"https://huggingface.co/datasets/{new_repo_id}")

# -------------------------------
# Usage
# -------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Upload models to Hugging Face Hub")
    parser.add_argument("--name", type=str, required=True, help="Repository ID (e.g., 'jiebi/ModelName')")
    parser.add_argument("--path", type=str, required=True, help="Local path to the folder to upload")
    parser.add_argument("--repo-type", type=str, default="model", help="Repository type: 'model' or 'dataset'")
    parser.add_argument("--private", action="store_true", help="Make the repository private")
    
    args = parser.parse_args()
    
    name = args.name
    path = args.path
    
    print(f"Uploading: {name} from {path}")
    create_and_upload(name, path, repo_type=args.repo_type, private=args.private)