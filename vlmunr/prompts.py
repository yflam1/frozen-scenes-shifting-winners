from collections import defaultdict
from typing import Iterable, Iterator, Optional


class TokenGraph(object):
    def __init__(self, start: str = "<START>", end: str = "<END>") -> str:
        self.start = start
        self.end = end
        self._adj: defaultdict[str, list[str]] = defaultdict(list)

    def __call__(
        self,
        shortest_first: bool = True,
        replacements: Optional[dict[str, str]] = None,
    ) -> Iterator[str]:
        """
        Enumerate sentences (paths from `self.start` to `self.end`).

        Args:
            shortest_first: BFS if True (short to long), DFS otherwise.
            replacements: A dictionary of words to replace in the output
                sentences.
        """

        _replacements = {} if replacements is None else replacements
        worklist: list[tuple[str, list[str]]] = [(self.start, [])]
        while len(worklist) > 0:
            node, path = worklist.pop(0 if shortest_first else -1)
            for nxt in self._adj[node]:
                if nxt == self.end:
                    out = " ".join(path)
                    for k, v in _replacements.items():
                        out = out.replace(k, v)
                    yield out
                else:
                    worklist.append((nxt, path + [nxt]))

    def add_edge(self, u: str, v: str) -> None:
        self._adj[u].append(v)

    def add_edges(self, edges: Iterable[tuple[str, str]]) -> None:
        for u, v in edges:
            self.add_edge(u, v)


def generate_vlm_prompts(query: Optional[str] = None) -> Iterator[str]:
    tg = TokenGraph(start="<EMPTY>", end="<END>")
    tg.add_edges(
        [
            ("<EMPTY>", "<QUERY>"),
            ("<EMPTY>", "a[1]"),
            ("<EMPTY>", "an"),
            ("<EMPTY>", "image"),
            ("<EMPTY>", "photo"),
            ("<EMPTY>", "picture"),
            ("<EMPTY>", "rendering"),
            ("<EMPTY>", "top-down"),
            ("<EMPTY>", "bird-eye"),
            ("an", "image"),
            ("a[1]", "photo"),
            ("a[1]", "picture"),
            ("a[1]", "rendering"),
            ("a[1]", "top-down"),
            ("a[1]", "bird-eye"),
            ("top-down", "image"),
            ("top-down", "photo"),
            ("top-down", "picture"),
            ("top-down", "rendering"),
            ("bird-eye", "image"),
            ("bird-eye", "photo"),
            ("bird-eye", "picture"),
            ("bird-eye", "rendering"),
            ("top-down", "view"),
            ("bird-eye", "view"),
            ("image", "about"),
            ("image", "depicting"),
            ("image", "of"),
            ("photo", "about"),
            ("photo", "depicting"),
            ("photo", "of"),
            ("picture", "about"),
            ("picture", "depicting"),
            ("picture", "of"),
            ("rendering", "about"),
            ("rendering", "depicting"),
            ("rendering", "of"),
            ("view", "about"),
            ("view", "depicting"),
            ("view", "of"),
            ("about", "<QUERY>"),
            ("depicting", "<QUERY>"),
            ("of", "<QUERY>"),
            ("about", "scene:"),
            ("depicting", "scene:"),
            ("of", "scene:"),
            ("about", "a[2]"),
            ("depicting", "a[2]"),
            ("of", "a[2]"),
            ("a[2]", "scene:"),
            ("a[2]", "virtual"),
            ("virtual", "scene:"),
            ("scene:", "<QUERY>"),
            ("<QUERY>", "<END>"),
        ]
    )
    for sentence in tg(replacements={"a[1]": "a", "a[2]": "a"}):
        sentence = sentence[0].upper() + sentence[1:]
        if query is not None:
            sentence = sentence.replace("<QUERY>", query)
        yield sentence


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", type=str, default=None)
    args = parser.parse_args()
    for i, p in enumerate(generate_vlm_prompts(query=args.query)):
        print(f"{i:03d}. {p}")
