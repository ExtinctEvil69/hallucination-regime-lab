"""Seeded fictional dossiers and six-turn question plans.

Two domains share one generator:
  * ``registry`` — the blueprint's abstract fictional entities and codes;
  * ``clinical`` — the use case: fictional patient charts (medication,
    allergy, ward and lab-panel codes). All names and codes are invented.

The *visible* record (dossier rows + questions + pre-answer covariates) and
the *gold* record (authoritative answers) are returned separately, so gold
answers never cross the detector boundary.

Pre-answer covariates (computable live from visible information only):
  evidence      1 if the referenced entity's queried field row is visible
  ambiguity     number of entities matching the reference in the question
  distractor    rows of the queried field whose code shares a 3-char prefix
                with another visible row of that field (near-duplicate load)
  context_len   approximate token count of the conversation so far
  turn          1-based turn index
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass

import numpy as np

from ..util import derive_seed

DOMAINS = {
    "registry": {
        "noun": "entity",
        "first": ["Arven", "Belisk", "Corra", "Dovran", "Elsin", "Fareth", "Gaelin", "Hollis",
                  "Isvet", "Jorran", "Kestra", "Lowen", "Marrik", "Nesta", "Orrin", "Pellan"],
        "last": ["Quill", "Raskel", "Solvane", "Tarrow", "Ulden", "Varrow", "Wexley", "Yarden"],
        "fields": ["access code", "vault code", "relay code", "permit code", "transit code"],
        "attr": "registered",
        "system": ("You are a registry assistant. Answer using only the dossier below. "
                   "Reply with only the requested code (format like AB3-41) or the single word "
                   "UNKNOWN if the dossier does not contain it. No explanation."),
        "title": "Registry dossier (fictional records)",
    },
    "clinical": {
        "noun": "patient",
        "first": ["Aldra", "Bennick", "Cyrene", "Dallow", "Emeth", "Fenna", "Garrow", "Hesper",
                  "Ilka", "Jessamy", "Korvin", "Liesl", "Maelor", "Nyssa", "Osric", "Pervin"],
        "last": ["Ashdown", "Brennick", "Calloway", "Draycott", "Elsworth", "Fairbank", "Greaves", "Halloran"],
        "fields": ["medication code", "allergy code", "ward code", "lab panel code", "imaging code"],
        "attr": "born",
        "system": ("You are a clinical records assistant supporting medication reconciliation. "
                   "Answer using only the patient chart below; never guess. Reply with only the "
                   "requested code (format like AB3-41) or the single word UNKNOWN if the chart "
                   "does not contain it. No explanation."),
        "title": "Ward chart (fictional patients — synthetic test data)",
    },
}
CODE_RE = re.compile(r"\b[A-Z]{2}\d-\d{2}\b")
WORD_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")
QTYPES = ("lookup", "prior_ref", "ambiguous", "missing")
COVARIATES = ("evidence", "ambiguity", "distractor", "context_len", "turn")
PRIMARY_COVARIATES = ("evidence", "ambiguity", "distractor", "turn")   # preregistered subset


def approx_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


def _code(rng, prefix=None):
    if prefix is None:
        prefix = "".join(rng.choice(list(string.ascii_uppercase), 2)) + str(rng.integers(0, 10))
    return f"{prefix}-{rng.integers(10, 100)}"


@dataclass
class Entity:
    first: str
    last: str
    year: int

    @property
    def full(self):
        return f"{self.first} {self.last}"


DESIGNS = {
    # blueprint v1 design (4 entities x 4 fields)
    "v1": {"n_entities": 4, "n_fields": 4, "max_dup": 2, "hidden": (1, 3), "n_turns": 6},
    # pilot-frozen design (harder; chosen after M5 pilot showed ~2% H at v1)
    "v2": {"n_entities": 6, "n_fields": 5, "max_dup": 5, "hidden": (2, 6), "n_turns": 6},
}


def make_dossier(dossier_id: int, domain: str = "registry", base_seed: int = 2026, design: str = "v2"):
    """Return (visible, gold) dicts for one dossier. Deterministic in (id, domain, seed, design)."""
    cfg = DOMAINS[domain]
    dz = DESIGNS[design]
    n_entities, n_fields, n_turns = dz["n_entities"], dz["n_fields"], dz["n_turns"]
    rng = np.random.default_rng(derive_seed(base_seed, domain, dossier_id, design))
    # entities: one shared surname pair creates referential ambiguity
    lasts = list(rng.choice(cfg["last"], n_entities - 1, replace=False))
    lasts.insert(int(rng.integers(0, n_entities)), lasts[int(rng.integers(0, len(lasts)))])
    firsts = rng.choice(cfg["first"], n_entities, replace=False)
    ents = [Entity(str(f), str(l), int(rng.integers(1940, 2005))) for f, l in zip(firsts, lasts)]
    # same surname must imply different year, otherwise "surname (year)" is unresolvable.
    # Deterministic bump (consumes no RNG draws, so unaffected dossiers are unchanged).
    for i in range(n_entities):
        while any(ents[j].last == ents[i].last and ents[j].year == ents[i].year for j in range(i)):
            ents[i].year += 1
    fields = cfg["fields"][:n_fields]
    codes = {}
    used = set()
    for e in range(n_entities):
        for f in range(n_fields):
            c = _code(rng)
            while c in used:
                c = _code(rng)
            used.add(c); codes[(e, f)] = c
    # controlled near-duplicate distractors: copy a prefix within a field column
    n_dup = int(rng.integers(0, dz["max_dup"] + 1))
    for _ in range(n_dup):
        f = int(rng.integers(0, n_fields)); a, b = rng.choice(n_entities, 2, replace=False)
        c = _code(rng, prefix=codes[(int(a), f)][:3])
        while c in used:
            c = _code(rng, prefix=codes[(int(a), f)][:3])
        used.add(c); codes[(int(b), f)] = c
    # experimentally assigned missingness: 1-3 rows hidden
    keys = list(codes)
    hidden = {keys[i] for i in rng.choice(len(keys), int(rng.integers(dz["hidden"][0], dz["hidden"][1] + 1)), replace=False)}
    rows = [(e, f) for e in range(n_entities) for f in range(n_fields) if (e, f) not in hidden]
    order = rng.permutation(len(rows))
    rows = [rows[i] for i in order]
    row_text = [f"{ents[e].full} ({cfg['attr']} {ents[e].year}) | {fields[f]}: {codes[(e, f)]}"
                for e, f in rows]
    system = cfg["system"] + "\n\n" + cfg["title"] + ":\n" + "\n".join(f"- {r}" for r in row_text)

    # question plan
    turns, gold = [], []
    prev_e = None
    for t in range(n_turns):
        if t == 0:
            qt = str(rng.choice(["lookup", "ambiguous"], p=[.7, .3]))
        else:
            qt = str(rng.choice(QTYPES, p=[.35, .25, .2, .2]))
        if qt == "missing":
            e, f = list(hidden)[int(rng.integers(0, len(hidden)))]
        elif qt == "prior_ref":
            e, f = prev_e, int(rng.integers(0, n_fields))
        else:
            if qt == "ambiguous":
                dup = [i for i in range(n_entities) if lasts.count(ents[i].last) > 1]
                e = int(rng.choice(dup))
            else:
                e = int(rng.integers(0, n_entities))
            vis_f = [ff for ff in range(n_fields) if (e, ff) not in hidden]
            f = int(rng.choice(vis_f)) if vis_f else int(rng.integers(0, n_fields))
        ent = ents[e]
        if qt == "prior_ref":
            q = f"What is that {cfg['noun']}'s {fields[f]}?"; amb = 1
        elif qt == "ambiguous":
            q = f"What is the {fields[f]} for {ent.last} ({cfg['attr']} {ent.year})?"
            amb = sum(x.last == ent.last for x in ents)
        else:
            q = f"What is the {fields[f]} for {ent.full}?"; amb = 1
        visible = (e, f) not in hidden
        col = [codes[(x, f)][:3] for x in range(n_entities) if (x, f) not in hidden]
        dist = sum(col.count(p) > 1 for p in col)
        turns.append({"turn": t + 1, "qtype": qt, "question": q, "field": fields[f],
                      "covariates": {"evidence": int(visible), "ambiguity": int(amb),
                                     "distractor": int(dist), "turn": t + 1}})
        gold.append({"turn": t + 1, "entity": e, "field_index": f,
                     "answer": codes[(e, f)] if visible else "UNKNOWN",
                     "hidden_code": None if visible else codes[(e, f)]})
        prev_e = e
    visible_rec = {"dossier_id": dossier_id, "domain": domain, "design": design, "system": system, "rows": row_text,
                   "turns": turns, "entities": [e.full for e in ents]}
    gold_rec = {"dossier_id": dossier_id, "domain": domain, "turns": gold,
                "all_codes": {f"{e}:{f}": c for (e, f), c in codes.items()},
                "hidden": [f"{e}:{f}" for e, f in sorted(hidden)]}
    return visible_rec, gold_rec


def assign_splits(ids, base_seed=2026, frac=(0.6, 0.2, 0.2)):
    """Dossier-level split ownership decided before fitting (300/100/100 for 500)."""
    ids = sorted(ids)
    rng = np.random.default_rng(derive_seed(base_seed, "splits", len(ids)))
    perm = list(rng.permutation(ids))
    n_tr, n_va = int(round(frac[0] * len(ids))), int(round(frac[1] * len(ids)))
    out = {}
    for i, d in enumerate(perm):
        out[int(d)] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return out
