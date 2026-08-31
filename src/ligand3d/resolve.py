"""Turn what someone types into a structure they can edit.

Drawing a fused polycyclic from scratch is tedious and error-prone, and most
molecules people want already have a name. This takes a name, a SMILES, an
InChI, or a PubChem CID and returns something the sketcher can load as a
starting scaffold.

Three routes, tried in order, because they fail in different places:

**Parsing** handles SMILES and InChI. Instant, offline, exact.

**OPSIN** handles systematic IUPAC nomenclature offline — `3-Cyano-7-
ethoxycoumarin`, `2-amino-2-methylpropan-1-ol`, `quinuclidin-3-one`. It is a
grammar for the naming rules, not a database, so it parses names nobody has
ever catalogued and knows nothing about `aspirin`.

**PubChem** handles exactly what OPSIN cannot: trivial names, trade names,
registry numbers, and anything else that is a lookup rather than a derivation.
It needs the network.

Together they cover most of what gets typed. Neither is a fallback for the
other — they are complements, and which one answered is reported, because
"OPSIN derived this from the name you typed" and "PubChem had a record under
that name" are different kinds of claim about whether you got what you meant.

Note on InChI: it is a structure *serialization*, not a name resolver. An InChI
string is accepted here, but InChI cannot turn `aspirin` into a structure —
that is what the other two routes are for.
"""

from __future__ import annotations

import pathlib

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .errors import Ligand3DError

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_WEB = "https://pubchem.ncbi.nlm.nih.gov/compound"
USER_AGENT = "ligand3d (https://github.com/SethWoodbury/ligand3d)"
DEFAULT_TIMEOUT = 12.0


class ResolveError(Ligand3DError):
    """Could not turn the query into a structure."""


@dataclass
class Resolved:
    """A structure, and an honest account of where it came from."""

    smiles: str
    source: str
    """parsed-smiles | parsed-inchi | opsin | pubchem | template."""
    query: str
    name: str = ""
    formula: str = ""
    cid: int | None = None
    url: str = ""
    synonyms: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)

    @property
    def provenance(self) -> str:
        """One line a person can judge the result by."""
        return {
            "parsed-smiles": "read directly as SMILES",
            "parsed-inchi": "read directly as InChI",
            "opsin": "derived from the name by OPSIN, offline, from IUPAC rules",
            "pubchem": f"looked up in PubChem (CID {self.cid})",
            "template": "a built-in scaffold",
            "peptide": "assembled from the peptide sequence you typed",
            "dna": "assembled from the DNA sequence you typed, 5' to 3'",
            "rna": "assembled from the RNA sequence you typed, 5' to 3'",
        }.get(self.source, self.source)

    def molblock(self) -> str:
        """A 2D molblock, laid out for a sketcher to load."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            raise ResolveError(f"RDKit could not read the result {self.smiles!r}")
        AllChem.Compute2DCoords(mol)
        return Chem.MolToMolBlock(mol)

    def to_json(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "molblock": self.molblock(),
            "source": self.source,
            "kind_label": KIND_LABELS.get(self.source, self.source),
            "provenance": self.provenance,
            "query": self.query,
            "name": self.name,
            "formula": self.formula,
            "cid": self.cid,
            "url": self.url,
            "synonyms": list(self.synonyms),
            "notes": list(self.notes),
        }


# A name is not a SMILES, but some are spelled the same. "CO" is methanol as
# SMILES and carbon monoxide as a name; "No" is nobelium. Anything that looks
# like a word gets sent to the name routes first, and the prefixes below settle
# it outright when the guess would be wrong.
_PREFIXES = ("smiles:", "inchi:", "name:", "cid:")

# Outside brackets, SMILES only ever uses these lowercase letters: b, c, n, o,
# p, s for aromatic atoms, and l and r as the tails of Cl and Br. A lowercase
# run containing anything else is a word, not a structure — which is what
# separates `benzene` from the `ccccc` inside `c1ccccc1`.
_SMILES_LOWER = set("bcnopslr")
_BRACKETS = re.compile(r"\[[^\]]*\]")
_LOWER_RUN = re.compile(r"[a-z]{4,}")
_OBVIOUSLY_TEXT = re.compile(r"[\s,'’]|^\d+[-,]|\bacid\b", re.IGNORECASE)


def looks_like_a_name(text: str) -> bool:
    """True if this should go to the name routes before the SMILES parser.

    Only picks an order — both routes are still tried — but picking well keeps
    a plain SMILES from making a needless trip to PubChem.
    """
    if _OBVIOUSLY_TEXT.search(text):
        return True
    # Bracket atoms are the one place SMILES uses other lowercase letters
    # ([nH], [Se], [se]), so they are removed before looking for word-like runs.
    bare = _BRACKETS.sub("", text)
    return any(
        set(run.group()) - _SMILES_LOWER for run in _LOWER_RUN.finditer(bare)
    )


def _canonical(smiles: str) -> tuple[str, str]:
    """Canonical SMILES and formula, or raise."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ResolveError(f"could not parse {smiles!r} as a structure")
    return Chem.MolToSmiles(mol), rdMolDescriptors.CalcMolFormula(mol)


def from_smiles_text(text: str) -> Resolved | None:
    """Read the query as SMILES, or return None if it is not one."""
    from rdkit import Chem

    from .molecule import rdkit_quiet

    with rdkit_quiet():
        mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    canonical, formula = _canonical(text)
    return Resolved(
        smiles=canonical, source="parsed-smiles", query=text, formula=formula
    )


def from_inchi_text(text: str) -> Resolved | None:
    """Read the query as InChI, or return None if it is not one."""
    if not text.strip().startswith("InChI="):
        return None
    from rdkit import Chem

    from .molecule import rdkit_quiet

    with rdkit_quiet():
        mol = Chem.MolFromInchi(text.strip())
    if mol is None:
        raise ResolveError("that looks like an InChI but RDKit could not read it")
    canonical, formula = _canonical(Chem.MolToSmiles(mol))
    return Resolved(
        smiles=canonical, source="parsed-inchi", query=text, formula=formula
    )


def opsin_available() -> bool:
    """True if the offline name parser can run.

    py2opsin ships the OPSIN jar but shells out to `java`, so both have to be
    present.
    """
    import importlib.util
    import shutil

    return (
        importlib.util.find_spec("py2opsin") is not None
        and shutil.which("java") is not None
    )


def from_opsin(name: str) -> Resolved | None:
    """Derive a structure from a systematic IUPAC name, offline.

    Returns None when OPSIN is absent or cannot parse the name — the second is
    ordinary for a trivial name like `aspirin` and is not an error.
    """
    if not opsin_available():
        return None
    import tempfile
    import warnings

    from py2opsin import py2opsin

    try:
        with warnings.catch_warnings():
            # An unparsable name is a normal outcome here, not something to
            # print at whoever is typing.
            warnings.simplefilter("ignore")
            # py2opsin writes its input file to a *fixed* name relative to the
            # working directory. Left alone that drops a file into whatever
            # directory someone happened to run `fetch` from, and two lookups
            # running at once collide on the same name. A private directory
            # per call fixes both.
            with tempfile.TemporaryDirectory(prefix="ligand3d-opsin-") as scratch:
                smiles = py2opsin(
                    name, tmp_fpath=str(pathlib.Path(scratch) / "input.txt")
                )
    except Exception:  # a missing jar, a broken java, anything
        return None
    if not smiles:
        return None

    canonical, formula = _canonical(smiles)
    return Resolved(
        smiles=canonical, source="opsin", query=name, name=name, formula=formula
    )


def _pubchem_get(path: str, timeout: float) -> Any:
    """One PubChem PUG REST call, returning parsed JSON."""
    url = f"{PUBCHEM_BASE}/{path}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise ResolveError(f"PubChem returned HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise ResolveError(
            f"could not reach PubChem ({exc.reason}). Use a SMILES, or a systematic "
            "name if OPSIN can parse it, to work offline."
        ) from exc
    except (TimeoutError, OSError) as exc:
        raise ResolveError(f"could not reach PubChem: {exc}") from exc


# PubChem renamed these: CanonicalSMILES became ConnectivitySMILES and
# IsomericSMILES became SMILES. Both spellings are read so the lookup keeps
# working whichever vintage answers.
_SMILES_KEYS = ("SMILES", "IsomericSMILES", "ConnectivitySMILES", "CanonicalSMILES")


def from_pubchem(query: str, timeout: float = DEFAULT_TIMEOUT) -> Resolved | None:
    """Look the query up in PubChem by name, or by CID if it is one.

    Returns None when PubChem has no record, and raises when it could not be
    asked — a missing compound and a missing network deserve different answers.
    """
    text = query.strip()
    if text.lower().startswith("cid:"):
        selector = f"compound/cid/{urllib.parse.quote(text[4:].strip())}"
    elif text.isdigit():
        selector = f"compound/cid/{text}"
    else:
        selector = f"compound/name/{urllib.parse.quote(text, safe='')}"

    properties = "MolecularFormula,SMILES,ConnectivitySMILES,IUPACName,Title"
    payload = _pubchem_get(f"{selector}/property/{properties}/JSON", timeout)
    if not payload:
        return None
    try:
        record = payload["PropertyTable"]["Properties"][0]
    except (KeyError, IndexError):
        return None

    smiles = next((record[k] for k in _SMILES_KEYS if record.get(k)), "")
    if not smiles:
        return None

    canonical, formula = _canonical(smiles)
    cid = record.get("CID")
    return Resolved(
        smiles=canonical,
        source="pubchem",
        query=query,
        name=record.get("Title") or record.get("IUPACName") or "",
        formula=record.get("MolecularFormula") or formula,
        cid=cid,
        url=f"{PUBCHEM_WEB}/{cid}" if cid else "",
    )


#: What an import can be read as. "auto" tries the routes in order; the rest
#: say outright, which is the only way to settle a query that two routes could
#: both answer differently — `GGCAT` is a DNA oligo and also a peptide.
KINDS: tuple[str, ...] = (
    "auto", "smiles", "inchi", "name", "cid", "peptide", "dna", "rna",
)

KIND_LABELS: dict[str, str] = {
    "auto": "Auto-detect",
    "smiles": "SMILES",
    "inchi": "InChI",
    "name": "Chemical name",
    "cid": "PubChem CID",
    "peptide": "Peptide sequence",
    "dna": "DNA sequence",
    "rna": "RNA sequence",
}

_SEQUENCE_KINDS = ("peptide", "dna", "rna")


def from_sequence(text: str, kind: str, ph: float | None = None) -> Resolved:
    """Build a peptide, DNA or RNA chain from its sequence."""
    from .biopolymer import SequenceError, build

    try:
        built = build(text, kind, ph=ph)
    except SequenceError as exc:
        raise ResolveError(str(exc)) from exc

    canonical, formula = _canonical(built.smiles)
    return Resolved(
        smiles=canonical,
        source=kind,
        query=text,
        name=built.description,
        formula=formula,
        notes=list(built.notes),
    )


def resolve(
    query: str,
    allow_network: bool = True,
    timeout: float = DEFAULT_TIMEOUT,
    kind: str = "auto",
    ph: float | None = None,
) -> Resolved:
    """Turn a query into a structure, trying the cheapest route that can work.

    `kind` forces a route. Prefixing the query with `smiles:`, `inchi:`,
    `name:`, or `cid:` does the same thing and takes precedence.

    Sequences are never auto-detected. `GGCAT` is a valid DNA oligo and an
    equally valid pentapeptide, and guessing would sometimes silently build the
    wrong polymer — so a sequence has to be asked for.
    """
    text = query.strip()
    if not text:
        raise ResolveError("nothing to look up")
    if kind not in KINDS:
        raise ResolveError(f"unknown import type {kind!r}; use one of {KINDS}")

    lowered = text.lower()
    forced = next((p for p in _PREFIXES if lowered.startswith(p)), None)
    if forced:
        text = text[len(forced):].strip()
    elif kind in _SEQUENCE_KINDS:
        return from_sequence(text, kind, ph=ph)
    elif kind != "auto":
        forced = f"{kind}:"

    if forced == "smiles:":
        found = from_smiles_text(text)
        if found is None:
            raise ResolveError(f"{text!r} is not a valid SMILES")
        return found
    if forced == "inchi:":
        found = from_inchi_text(text if text.startswith("InChI=") else f"InChI={text}")
        if found is None:
            raise ResolveError(f"{text!r} is not a valid InChI")
        return found
    if forced == "cid:":
        found = from_pubchem(f"cid:{text}", timeout) if allow_network else None
        if found is None:
            raise ResolveError(
                f"no PubChem record for CID {text}"
                if allow_network else "CID lookup needs the network"
            )
        return found

    if (found := from_inchi_text(text)) is not None:
        return found

    by_name_first = forced == "name:" or looks_like_a_name(text)
    if not by_name_first and (found := from_smiles_text(text)) is not None:
        return found

    tried: list[str] = []
    if (found := from_opsin(text)) is not None:
        return found
    tried.append(
        "OPSIN could not parse it as a systematic name"
        if opsin_available()
        else "OPSIN is not installed (pip install py2opsin, and you need java)"
    )

    network_error: ResolveError | None = None
    if allow_network:
        try:
            if (found := from_pubchem(text, timeout)) is not None:
                return found
            tried.append("PubChem has no record under that name")
        except ResolveError as exc:
            network_error = exc
            tried.append(str(exc))
    else:
        tried.append("PubChem was not tried (network lookups are off)")

    # Last chance: it may have been a SMILES that merely reads like a word.
    if by_name_first and (found := from_smiles_text(text)) is not None:
        found.notes.append(
            "read as SMILES — it looked like a name, but no name route matched "
            "and it parses as a structure"
        )
        return found

    if network_error is not None and not tried[:-1]:
        raise network_error

    hint = "Try a SMILES, or prefix the query with smiles:, inchi:, name:, or cid:."
    if _could_be_a_sequence(text):
        hint = (
            "This looks like it could be a sequence. Sequences are never guessed, "
            "because the same letters can be a peptide and an oligonucleotide — "
            "choose Peptide, DNA or RNA as the import type to build it as one."
        )
    raise ResolveError(f"could not resolve {query!r}. " + "; ".join(tried) + ". " + hint)


def _could_be_a_sequence(text: str) -> bool:
    """Whether a failed lookup was plausibly someone pasting a sequence.

    Only chooses which advice to print once everything else has failed, so it
    can afford to be generous — but it asks the real question, which is whether
    every letter is actually a residue code.
    """
    bare = re.sub(r"\([^)]*\)|[\s\-*.,:0-9]", "", text).upper()
    if len(bare) < 3 or not bare.isalpha():
        return False
    residues = set("ACDEFGHIKLMNPQRSTVWYUO")  # the peptide alphabet covers ACGTU
    return set(bare) <= residues


# A few scaffolds worth starting from, for when the point is to draw a
# derivative rather than to reproduce a specific compound. Kept short on
# purpose: this is a starting point, not a compound library.
TEMPLATES: tuple[tuple[str, str, str], ...] = (
    ("benzene", "c1ccccc1", "the ring everything else hangs off"),
    ("pyridine", "c1ccncc1", "benzene with one aza substitution"),
    ("piperidine", "C1CCNCC1", "saturated N heterocycle"),
    ("piperazine", "C1CNCCN1", "two nitrogens, para"),
    ("morpholine", "C1COCCN1", "solubilising saturated ring"),
    ("indole", "c1ccc2[nH]ccc2c1", "tryptophan core"),
    ("quinoline", "c1ccc2ncccc2c1", "fused N bicyclic"),
    ("coumarin", "O=c1ccc2ccccc2o1", "benzopyranone; fluorophore scaffold"),
    ("purine", "c1ncc2[nH]cnc2n1", "adenine and guanine core"),
    ("imidazole", "c1c[nH]cn1", "histidine side chain"),
    ("thiophene", "c1ccsc1", "common bioisostere for phenyl"),
    ("cyclohexane", "C1CCCCC1", "chair reference"),
    ("adamantane", "C1C2CC3CC1CC(C2)C3", "rigid cage"),
    ("quinuclidine", "C1CN2CCC1CC2", "bridged bicyclic amine"),
    ("steroid", "C1CC2CCC3C(CCC4CCCCC34)C2C1", "gonane skeleton"),
    ("beta-lactam", "O=C1CCN1", "penicillin core"),
    ("benzimidazole", "c1ccc2[nH]cnc2c1", "fused imidazole"),
    ("naphthalene", "c1ccc2ccccc2c1", "fused aromatic"),
)


def template(name: str) -> Resolved:
    """One of the built-in scaffolds, by name."""
    wanted = name.strip().lower()
    for label, smiles, note in TEMPLATES:
        if label == wanted:
            canonical, formula = _canonical(smiles)
            return Resolved(
                smiles=canonical, source="template", query=name, name=label,
                formula=formula, notes=[note],
            )
    known = ", ".join(label for label, _, _ in TEMPLATES)
    raise ResolveError(f"no template called {name!r}. Available: {known}")


def template_list() -> list[dict[str, str]]:
    """The scaffolds, for a menu."""
    return [
        {"name": label, "smiles": smiles, "note": note}
        for label, smiles, note in TEMPLATES
    ]
