import gzip
import json
import os
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator, Literal, Optional, overload

import numpy as np
from PIL import Image
from pydantic import BaseModel, validate_call

from .dtypes import ImgLike, JsonObject, PathLike, YamlObject


@validate_call
def exists(path: PathLike) -> bool:
    return Path(path).exists()


@validate_call
def is_dir(path: PathLike) -> bool:
    return Path(path).is_dir()


@validate_call
def is_file(path: PathLike) -> bool:
    return Path(path).is_file()


@validate_call
def get_basename(file_path: PathLike) -> str:
    return Path(file_path).name


@validate_call
def get_filename(file_path: PathLike) -> str:
    return Path(file_path).stem


@validate_call
def get_file_ext(file_path: PathLike, no_dot: bool = False) -> str:
    ext = Path(file_path).suffix
    return ext.lstrip(".") if no_dot else ext


@validate_call
def get_file_exts(file_path: PathLike, no_dot: bool = False) -> list[str]:
    exts = Path(file_path).suffixes
    return list(map(lambda x: x.lstrip("."), exts)) if no_dot else exts


@validate_call
def get_file_size(file_path: PathLike) -> int:
    return Path(file_path).stat().st_size


@validate_call
def get_parent(file_path: PathLike) -> Path:
    return Path(file_path).parent.absolute()


@validate_call
def create_dir(
    tgt_dir: PathLike,
    remove_existing: bool = False,
    build_tree: bool = True,
    exist_ok: bool = False,
    mode: int = 511,
) -> bool:
    """
    Create a tree of directory.

    Args:
        tgt_dir (PathLike): Target directory
        remove_existing (bool, optional): Remove existing `tgt_dir` (if any) before creation. Defaults to False.
        build_tree (bool, optional): Create a leaf directory and all intermediate ones. Defaults to True.
        exist_ok (bool, optional): Raise an OSError if `tgt_dir` already exists. Defaults to False.
        mode (int, optional): Set the file mode and access flags. Defaults to 511.

    Returns:
        bool: True if `tgt_dir` is created
    """

    if remove_existing:
        remove_dir(tgt_dir)
    try:
        Path(tgt_dir).mkdir(mode=mode, parents=build_tree, exist_ok=exist_ok)
    except Exception:
        return False
    return True


@validate_call
def remove_dir(tgt_dir: PathLike, only_empty: bool = False) -> bool:
    tgt_dir = Path(tgt_dir)
    if exists(tgt_dir):
        if not is_dir(tgt_dir):
            raise NotADirectoryError(
                f"{str(tgt_dir)} is not a directory. tgt_dir must be a directory"
            )
        tgt_dir.rmdir() if only_empty else shutil.rmtree(tgt_dir)
        return True
    return False


@validate_call
def remove_file(file_path: PathLike) -> bool:
    file_path = Path(file_path)
    if exists(file_path):
        if not is_file(file_path):
            raise OSError(
                f"{str(file_path)} is not a regular file. file_path must be a regular file."
            )
        file_path.unlink()
        return True
    return False


@validate_call
def get_cwd() -> Path:
    """
    Get the current working directory (CWD).

    Returns:
        Path: CWD
    """

    return Path.cwd()


@validate_call
def get_home() -> Path:
    """
    Get the user's home directory.

    Returns:
        Path: User's home directory
    """

    return Path.home()


@validate_call
def iter_files(
    tgt_dir: PathLike,
    exts: Optional[set[str]] = None,
    include_dir: bool = False,
    recursive: bool = False,
) -> Iterator[Path]:
    """
    Get all file paths under a directory (recursively). Similar to the `ls -a`
    (`ls -R` for recursive directory listing) command on Linux.

    Args:
        tgt_dir (PathLike): Target directory.
        exts (Optional[set[str]], optional): If not None, return a file path
            only if its extension (including leading period) is in `exts`.
            Defaults to None.
        include_dir (bool, optional): Return directory paths together with file
            paths. Defaults to False.
        recursive (bool, optional): Recurse into sub-directories. Defaults to
            False.

    Returns:
        Iterator[Path]: File paths under `tgt_dir`.
    """

    exts = set(map(lambda x: x.lower(), exts)) if exts is not None else exts
    for child in (tgt_dir := Path(tgt_dir)).iterdir():
        if is_file(child):
            if exts is not None:
                ext = get_file_ext(child)
                if ext.lower() in exts:
                    yield child
            else:
                yield child
        elif is_dir(child):
            if include_dir:
                yield child
            if recursive:
                yield from iter_files(
                    child,
                    exts=exts,
                    include_dir=include_dir,
                    recursive=recursive,
                )


@validate_call
def find_files(target_dir, pattern: str = "*.*") -> Iterator[Path]:
    return Path(target_dir).rglob(pattern)


@validate_call
def load_text_file(path: str) -> str:
    with open(path, encoding="utf-8") as file:
        return file.read()


@validate_call
def read_file(
    file_path: PathLike,
    remove_spaces: bool = False,
    remove_empty: bool = False,
) -> Iterator[str]:
    """
    Read lines from a file (with formatting).

    Args:
        file_path (PathLike): Target file
        remove_spaces (bool, optional): Remove leading and trailing whitespaces. Defaults to False.
        remove_empty (bool, optional): Omit empty lines. Defaults to False.

    Returns:
        Iterator[str]: Lines in a file
    """

    with Path(file_path).open() as f:
        for line in f:
            line = line.rstrip("\n")
            line = line.strip() if remove_spaces else line
            if remove_empty and line == "":
                continue
            yield line


@validate_call
def copy_file(src: PathLike, dst: PathLike) -> None:
    """
    Copy a file.

    Args:
        src (PathLike): Source path. Must be a file.
        dst (PathLike): Destination path. It can be a file or directory.
    """

    shutil.copy(src, dst)


@validate_call
def _replace_env_var(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _replace_env_var(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_replace_env_var(item) for item in data]
    elif isinstance(data, str) and data.startswith("$"):
        return os.environ.get(data[1:].split()[0], None)
    return data


@overload
@validate_call
def load_json(path: PathLike, is_jsonl: Literal[False] = False) -> JsonObject: ...


@overload
@validate_call
def load_json(path: PathLike, is_jsonl: Literal[True] = True) -> list[JsonObject]: ...


@validate_call
def load_json(path: PathLike, is_jsonl: bool = False) -> JsonObject | list[JsonObject]:
    from io import TextIOWrapper

    def try_load(f: TextIOWrapper) -> Any | list[Any]:
        return (
            [json.loads(_line) for line in f if (_line := line.strip()) != ""]
            if is_jsonl
            else json.load(f)
        )

    if str(path).lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return try_load(f)
    else:
        with open(path, encoding="utf-8") as f:
            return try_load(f)


class DefaultJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump_json()
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)


@validate_call
def save_json(
    data: Any,
    path: PathLike,
    indent: Optional[int] = None,
    encoder: Optional[type[json.JSONEncoder]] = DefaultJsonEncoder,
) -> bool:
    ext = get_file_ext(path, no_dot=True).lower()
    kwargs = {"ensure_ascii": False, "indent": indent, "cls": encoder}
    match ext:
        case "gz":
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(data, f, **kwargs)
        case _:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, **kwargs)
    return exists(path)


@validate_call
def load_yaml(
    path: PathLike, safe: bool = True, replace_env_var: bool = False
) -> YamlObject:
    import yaml

    if str(path).lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = yaml.safe_load(f) if safe else yaml.unsafe_load(f)
    else:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) if safe else yaml.unsafe_load(f)
    return _replace_env_var(data) if replace_env_var else data


@validate_call
def save_yaml(data: Any, path: PathLike) -> bool:
    import yaml

    if str(path).lower().endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    else:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    return exists(path)


@validate_call
def get_available_path(path: PathLike) -> Path:
    path: Path = Path(path)
    if not path.exists():
        return path
    if path.is_dir():
        base, ext = path, ""
    else:
        base, ext = path.with_suffix(""), path.suffix
    i = 2
    new_path = Path(f"{base}_{i}{ext}")
    while new_path.exists():
        i += 1
        new_path = Path(f"{base}_{i}{ext}")
    return new_path


@validate_call(config=dict(arbitrary_types_allowed=True))
def load_image_to_pillow(img: ImgLike) -> Image.Image:
    if isinstance(img, Image.Image):
        img = img
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    elif isinstance(img, str):
        if img.startswith("http://") or img.startswith("https://"):
            import requests

            img = Image.open(requests.get(img, stream=True).raw)
        else:
            img = Image.open(img)
    elif isinstance(img, Path):
        img = Image.open(img)
    else:
        raise ValueError(f'Unsupported image type "{type(img)}"')
    return img.convert("RGB")
