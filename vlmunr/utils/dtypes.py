from pathlib import Path
from typing import Annotated, Any, Dict, TypeVar, Union

import numpy as np
from PIL import Image
from pydantic import StringConstraints

UNSET = object()  # Sentinel for unset values

PathLike = Union[Path, str]
Url = str
ImgLike = Image.Image | np.ndarray | Url | bytes | PathLike

# Type for dictionary with string as key
T = TypeVar("T")
SDict = Dict[str, T]

JsonObject = Any
YamlObject = Any
NonEmptyStr = Annotated[str, StringConstraints(strict=True, min_length=1)]
