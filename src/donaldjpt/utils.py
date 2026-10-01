# %% Setting up

import shutil
from pathlib import Path

# %% Function to find project root folder


def get_project_root() -> Path:
    # Get the absolute path of the current file
    current_file = Path(__file__).resolve()

    # Walk up the directory tree
    for parent in current_file.parents:
        # If we find a folder named 'src', its parent is the project root
        if parent.name == "src":
            return parent.parent

    # Fallback if 'src' is not found in the path (optional)
    # You could return current_file.parent or raise an error
    raise FileNotFoundError("Project root not found: 'src' directory missing in path.")


# %% Function to remove an existing directory if it exists


def clean_dir_if_exists(path: str | Path) -> None:
    path = Path(path)

    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


# %%
