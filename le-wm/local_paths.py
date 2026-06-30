import os
from pathlib import Path


def configure_output_paths():
    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(os.environ.get("LEWM_OUTPUT_ROOT", repo_root / "outputs"))

    paths = {
        "XDG_CACHE_HOME": output_root / ".cache",
        "XDG_CONFIG_HOME": output_root / ".config",
        "XDG_DATA_HOME": output_root / ".local",
        "PIP_CACHE_DIR": output_root / ".cache" / "pip",
        "UV_CACHE_DIR": output_root / ".cache" / "uv",
        "HF_HOME": output_root / ".cache" / "huggingface",
        "HF_HUB_CACHE": output_root / ".cache" / "huggingface" / "hub",
        "MPLCONFIGDIR": output_root / ".cache" / "matplotlib",
        "TMPDIR": output_root / "tmp",
        "STABLEWM_HOME": output_root / "stable-wm",
    }

    if os.environ.get("LEWM_RESPECT_EXTERNAL_CACHE", "0") == "1":
        for key, path in paths.items():
            os.environ.setdefault(key, str(path))
    else:
        for key, path in paths.items():
            os.environ[key] = str(path)

    for key in paths:
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

    return output_root
