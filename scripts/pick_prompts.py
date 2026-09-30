import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from os import listdir
from os.path import isdir, isfile, join
from typing import Annotated

import inflect
from pydantic import Field
from tqdm import tqdm

from vlmunr.utils.dtypes import SDict
from vlmunr.utils.file import save_json
from vlmunr.utils.llm import ZERO_SHOT_COT_REASONING as ZSCR
from vlmunr.utils.llm import JsonResponseModel, Llm

SRC = join(os.path.dirname(os.path.abspath(__file__)), "..", "prompts", "sources")

# Holodeck: 52 scene types across 5 MIT-Scenes categories (from the Holodeck paper).
HOLODECK_SCENE_TYPES = [
    "deli",
    "bakery",
    "grocery store",
    "clothing store",
    "toy store",
    "bookstore",
    "bedroom",
    "dining room",
    "living room",
    "kitchen",
    "bathroom",
    "children's room",
    "museum",
    "locker room",
    "art gallery",
    "waiting room",
    "reception area",
    "lobby",
    "gym",
    "casino",
    "arcade",
    "game room",
    "playroom",
    "music room",
    "office",
    "meeting room",
    "conference room",
    "classroom",
    "computer lab",
    "library",
    "laundromat",
    "hair salon",
    "spa",
    "dental office",
    "warehouse",
    "workshop",
    "restaurant kitchen",
    "bar",
    "cafe",
    "food court",
    "wine cellar",
    "pantry",
    "green house",
    "garage",
    "attic",
    "basement",
    "studio apartment",
    "dorm room",
    "hotel room",
    "nursery",
    "home theater",
    "sunroom",
]

p = inflect.engine()


class Response(JsonResponseModel):
    text: str
    probability: Annotated[float, Field(ge=0.0, le=1.0)]


class Responses(JsonResponseModel):
    reasoning: str
    responses: list[Response]


def gen_paraphrases(
    sbert,
    prompt: str,
    model_name: str,
    max_retries: Annotated[int, Field(ge=0)] = 10,
    max_tokens: Annotated[int, Field(ge=1)] = 8192,
    timeout: Annotated[float, Field(gt=0.0)] = 300.0,
    base_url: str | None = None,
    api_key: str | None = None,
) -> list[SDict[str | float]]:
    """
    [Verbalized Sampling](https://arxiv.org/abs/2510.01171)
    """
    assert sbert is not None
    prompt = f"""\
## Original Input Prompt

Generate a syntactically different but **semantically identical** variant of the following text by simply replacing words with their synonyms and/or restructuring sentences (but do NOT remove any content):
{prompt}

## Task

Generate 10 responses to the original input prompt using chain-of-thought reasoning.
First, provide a single "reasoning" field, detailing your step-by-step thought process.
Then, return the output in JSON format with the field "responses" (a list of dictionaries).
Each dictionary must include:
- "text": the response string only (no explanation or extra text).
- "probability": the estimated probability from 0.0 to 1.0 of this response given the original input prompt (relative to the full distribution).

You must respond using only JSON without preamble, postamble, and additional commentary.
Output template:
{Responses.to_str(reasoning=ZSCR)}"""

    llm = Llm(
        model_name,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        api_key=api_key,
        base_url=base_url,
    )
    llm_output = llm(prompt, Responses, verbose=False)
    assert llm_output.response is not None
    paraphrases: list[SDict[str | float]] = []
    for response in llm_output.response.responses:
        score = sbert(sbert.encode([prompt]), sbert.encode([response.text]))
        paraphrases.append({"paraphrase": response.text, "similarity": score.item()})
    return paraphrases


def load_sceneeval():
    path = join(SRC, "sceneeval.csv")
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for r in rows:
        desc = r["Description"].strip()
        if not desc:
            continue
        out.append(
            {
                "source": "sceneeval",
                "prompt": desc,
                "difficulty": r.get("Difficulty", "").strip(),
            }
        )
    return out


def load_instructscene():
    base_dir = join(SRC, "instructscene")
    out = []
    for file in listdir(base_dir):
        p = join(base_dir, file)
        if not isfile(p) or not file.endswith(".txt"):
            continue
        room = file[:-4]
        with open(p) as handle:
            for line in handle:
                s = line.strip()
                if not s:
                    continue
                out.append({"source": "instructscene", "prompt": s, "room": room})
    return out


def load_holodeck():
    return [{"source": "holodeck", "prompt": p.a(st)} for st in HOLODECK_SCENE_TYPES]


def load_genxr():
    base_dir = join(SRC, "genxr")
    out = []
    for bt in listdir(base_dir):
        subdir = join(base_dir, bt)
        if not isdir(subdir):
            continue
        for file in listdir(subdir):
            if not file.endswith(".json"):
                continue
            with open(join(subdir, file)) as f:
                d = json.load(f)
            prompt = d.get("prompts", {}).get("humanized", "").strip()
            if not prompt:
                continue
            out.append({"source": "genxr", "prompt": prompt, "type": bt})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--sceneeval", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--paraphrase", action="store_true")
    ap.add_argument("--llm-model", type=str, default=None)
    ap.add_argument("--llm-max-retries", type=int, default=10)
    ap.add_argument("--llm-max-tokens", type=int, default=8192)
    ap.add_argument("--llm-timeout", type=float, default=300.0)
    ap.add_argument("--llm-base-url", type=str, default=None)
    ap.add_argument("--llm-api-key", type=str, default=None)
    args = ap.parse_args()
    if not 0.0 <= args.sceneeval <= 1.0:
        raise ValueError(f"--sceneeval must be in [0,1], got {args.sceneeval}")
    if args.paraphrase and not args.llm_model:
        raise ValueError("--llm-model is required when --paraphrase is set")
    sbert = None
    if args.paraphrase:
        from vlmunr.models.sbert import SentenceBert

        sbert = SentenceBert()

    pools = {
        "sceneeval": load_sceneeval(),
        "instructscene": load_instructscene(),
        "holodeck": load_holodeck(),
        "genxr": load_genxr(),
    }
    count = 0
    for k, v in pools.items():
        n = len(v)
        count += n
        print(f"Source '{k}': {n}")
    if count < args.n:
        raise RuntimeError(f"Not enough prompts ({count} < {args.n})")

    rng = random.Random(args.seed)

    def largest_remainder(total, k):
        """Split `total` into exactly `k` non-negative ints summing to `total`.

        Hamilton / largest-remainder method: each slot gets `total // k`, and
        the leftover `total % k` units are handed out one each to the first
        slots. Deterministic, so results are reproducible for a given seed.
        """
        if k <= 0:
            return []
        base, rem = divmod(total, k)
        quotas = [base] * k
        for i in range(rem):
            quotas[i] += 1
        return quotas

    def stratified_pick(items, key, total, overflow_value=None):
        """Pick `total` items, spread evenly across the distinct values of
        `key` (discovered from the data, never hardcoded).

        - Categories with fewer items than their even share contribute all they
          have; the leftover portion is filled first from rows whose `key`
          equals `overflow_value` (if given), then from leftover rows in any
          category.
        - All picks are deduped against the shared `seen` set.
        Returns the picked rows (may be fewer than `total` if the source is
        exhausted).
        """
        groups = defaultdict(list)
        overflow = []
        for r in items:
            if r["prompt"] in seen:
                continue
            kv = r.get(key)
            if overflow_value is not None and kv == overflow_value:
                overflow.append(r)
            else:
                groups[kv].append(r)
        for g in groups.values():
            rng.shuffle(g)
        rng.shuffle(overflow)

        cats = list(groups.keys())
        quotas = largest_remainder(total, len(cats)) if cats else []
        picked = []
        leftover = []
        for cat, q in zip(cats, quotas):
            g = groups[cat]
            take = min(len(g), q)
            for r in g[:take]:
                seen.add(r["prompt"])
            picked += g[:take]
            leftover += g[take:]

        remainder = total - len(picked)
        for r in overflow + leftover:
            if remainder <= 0:
                break
            if r["prompt"] in seen:
                continue
            seen.add(r["prompt"])
            picked.append(r)
            remainder -= 1
        return picked

    def plain_pick(items, total):
        """Sample `total` distinct items with no stratification (deduped)."""
        cand = [r for r in items if r["prompt"] not in seen]
        rng.shuffle(cand)
        chosen = cand[:total]
        for r in chosen:
            seen.add(r["prompt"])
        return chosen

    seen = set()
    picked = []

    # --- SceneEval: exact proportion, stratified evenly across difficulties ---
    se = pools["sceneeval"]
    n_sceneeval = max(0, min(round(args.n * args.sceneeval), args.n))
    if n_sceneeval > len(se):
        raise RuntimeError(
            f"--sceneeval proportion requires {n_sceneeval} sceneeval prompts "
            f"but only {len(se)} are available"
        )
    se_picked = stratified_pick(se, "difficulty", n_sceneeval, overflow_value="")
    picked += se_picked

    # --- Remaining proportion: even split across the other three sources ---
    n_other = args.n - len(se_picked)
    other_sources = ["instructscene", "holodeck", "genxr"]
    strat_key = {"instructscene": "room", "genxr": "type"}  # holodeck: plain sample
    quotas = largest_remainder(n_other, len(other_sources))

    non_se_leftovers = []
    for src, q in zip(other_sources, quotas):
        pool = pools[src]
        if src in strat_key:
            picked += stratified_pick(pool, strat_key[src], q)
        else:
            picked += plain_pick(pool, q)
        non_se_leftovers += [r for r in pool if r["prompt"] not in seen]

    # Shortfall top-up: redistribute across the other non-sceneeval sources
    # first, then fall back to sceneeval leftovers only if those are exhausted.
    se_leftovers = [r for r in se if r["prompt"] not in seen]
    rng.shuffle(non_se_leftovers)
    rng.shuffle(se_leftovers)
    for r in non_se_leftovers + se_leftovers:
        if len(picked) >= args.n:
            break
        if r["prompt"] in seen:
            continue
        seen.add(r["prompt"])
        picked.append(r)

    if len(picked) < args.n:
        raise RuntimeError(
            f"Not enough prompts: requested {args.n}, only {len(picked)} "
            f"available after stratification"
        )
    picked = picked[: args.n]
    rng.shuffle(picked)

    # Group by source, preserving the seeded shuffle order within each source.
    grouped: dict[str, list[dict]] = {}
    for r in picked:
        grouped.setdefault(r["source"], []).append(r)

    out: dict[str, list[dict]] = {}
    for src, rows in tqdm(grouped.items(), desc="Paraphrases"):
        entries = []
        for r in tqdm(rows, desc=src, leave=False):
            prompt = r["prompt"].replace("\n", " ").strip()
            entry: dict = {"prompt": prompt, "paraphrases": []}
            if args.paraphrase:
                entry["paraphrases"] = gen_paraphrases(
                    sbert,
                    prompt,
                    args.llm_model,
                    max_retries=args.llm_max_retries,
                    max_tokens=args.llm_max_tokens,
                    timeout=args.llm_timeout,
                    base_url=args.llm_base_url,
                    api_key=args.llm_api_key,
                )
            entries.append(entry)
        out[src] = entries

    save_json(out, args.out, indent=2)

    print(f"Wrote {len(picked)} prompts -> {args.out}")
    print("By source:", Counter(r["source"] for r in picked))
    print(
        "sceneeval by difficulty:",
        Counter(r["difficulty"] for r in picked if r["source"] == "sceneeval"),
    )
    print(
        "instructscene by room:",
        Counter(r["room"] for r in picked if r["source"] == "instructscene"),
    )
    print(
        "genxr by type:", Counter(r["type"] for r in picked if r["source"] == "genxr")
    )


if __name__ == "__main__":
    main()
