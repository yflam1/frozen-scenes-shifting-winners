from typing import Optional

import torch
from pydantic import BaseModel


class VlmResult(BaseModel, arbitrary_types_allowed=True):
    scores: Optional[torch.Tensor] = None  # [len(texts), len(images)]
