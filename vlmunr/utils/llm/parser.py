import json
import re
from json.decoder import JSONDecodeError
from typing import Any, Optional

import json_repair
from pydantic import validate_call


class JsonParser(object):
    @validate_call
    def __call__(self, text: str) -> Optional[dict[str, Any]]:
        extracted = self.extract(text)
        if extracted is None:
            return None
        try:
            return json.loads(extracted)
        except JSONDecodeError:  # Complete but invalid structure
            repaired = json_repair.loads(extracted, skip_json_loads=True)
            if len(repaired) > 0:
                return repaired
            raise

    @staticmethod
    @validate_call
    def extract(text: str) -> Optional[str]:
        # Remove any C-style // comments that are not inside a string
        text = re.sub(
            r'("(?:\\.|[^"\\])*")|(\s*\/\/.*)',
            lambda m: m.group(1) if m.group(1) is not None else "",
            text,
        )

        # Use a robust brace-counting algorithm instead of a greedy regex
        try:
            start_index = text.index("{")
        except ValueError:
            return None  # No opening brace found
        brace_count = 1
        in_string = False
        for i in range(start_index + 1, len(text)):
            char = text[i]
            if char == '"' and text[i - 1] != "\\":
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
            if brace_count == 0:  # We've found the matching closing brace
                return text[start_index : i + 1]
        return None  # Return None if the loop finishes (unmatched braces)
