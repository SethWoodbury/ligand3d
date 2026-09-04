# Chemistry

Stereochemistry, protonation, conformers, and what ligand3d refuses to do rather than answer wrongly.

[← back to the README](../README.md)

## Protonation

The default is **what you drew**. If you type a neutral carboxylic acid you get a
neutral carboxylic acid.

```bash
ligand3d build "NCC1(CC(=O)O)CCCCC1"                       # as drawn
ligand3d build "NCC1(CC(=O)O)CCCCC1" --protonate           # dimorphite-dl at pH 7.4
ligand3d build "NCC1(CC(=O)O)CCCCC1" --ph 2.0              # at pH 2
ligand3d build "NCC1(CC(=O)O)CCCCC1" --protonate --enumerate-states  # one file per state
```


## What it refuses to do

Three inputs are rejected rather than answered, because the answer would be wrong
in a way that looks right:

- **Undefined stereochemistry** — unless you pass `--stereo any` or
  `--stereo enumerate`. Constrained centers that only look stereogenic (the
  bridgeheads of 3-quinuclidinone, say) are not an ambiguity and do not trigger this.
- **Disconnected fragments** — a salt or solvate. Distance geometry has no restraints
  between components and stacks them on top of each other, at measured separations of
  0.0 Å. Use `--largest-fragment` to keep the biggest component and drop the counterion.
- **A charged molecule on a potential with no charge channel**, or a zwitterion on one
  with no implicit solvent. Override with `--allow-charge-mismatch` if you mean it.

After minimization it also verifies that stereochemistry, protonation state, and
heavy-atom connectivity all survived — on every conformer, not just the first.


## Conformers — and what the default actually does

**Every build searches.** Asking for one output structure does not mean one guess: a
batch of conformers is generated with ETKDG, minimized with the cheap force field, and
only the best `--confs` are kept. The count scales with rotatable bonds (20 for a rigid
cage, up to 300 for something floppy) and is overridable with `--sample`.

This matters more than it sounds. Minimizing a single ETKDG guess is a *local*
minimization, and for gabapentin the answer moved by **9.6 kcal/mol** depending only on
the random seed:

| | best energy found |
|---|---|
| one guess, 5 different seeds | −7.46, −17.03, −9.49, −8.74, −15.15 |
| searching (the default now) | −17.03, −17.03, −17.06, −17.25, −17.32 |

The cost is about half a second.

```bash
ligand3d build "<smiles>"                    # searches ~60, keeps the best 1
ligand3d build "<smiles>" --confs 20         # searches, keeps 20
ligand3d build "<smiles>" --sample 500 -n 50 # search harder
ligand3d build "<smiles>" --sample 1         # skip the search: one guess, minimized
ligand3d conformers "<smiles>" -n 50 --method crest
```

ETKDG is a genuine stochastic global sampler — independent distance-geometry starts with
torsion preferences from CSD statistics — not a walk from one structure. Survivors are
de-duplicated by symmetry-corrected heavy-atom RMSD and ranked by energy. CREST does far
more (metadynamics at the GFN level, minutes instead of seconds) and is what to reach for
when the answer really matters.

**Chained backends search cheaply and refine narrowly.** `--backend mmff94,gfn2` runs
MMFF94 over the whole sample, prunes to the survivors, and only then runs GFN2 on those:

```
searched 57 conformer(s) via rdkit, keeping the best 2
mmff94 narrowed 57 to 2; refining with gfn2
minimization time: mmff94 0.29s, gfn2 0.45s
```

Running both methods over all 57 would have cost about thirty seconds of GFN2 to rediscover
shapes MMFF94 already found.


## Stereochemistry reporting

```console
$ ligand3d stereo "C[C@@H](O)[C@H](N)C(=O)O"
C4H9NO3  C[C@@H](O)[C@H](N)C(=O)O
  2 stereocenter(s): atom 1 = R, atom 3 = S

$ ligand3d stereo "OC(=O)/C=C\C(=O)O"
C4H4O4  O=C(O)/C=C\C(=O)O
  double bond 3-4: Z (cis)
```

**E/Z and cis/trans are not synonyms**, and ligand3d only claims the second where it is
defensible. E/Z comes from CIP priorities and is always well defined. cis/trans compares
two *reference* substituents, which is only meaningful when it is obvious which two are
meant — that is, when each alkene carbon carries exactly one hydrogen. There, Z is cis and
E is trans, and both labels are printed.

For a tri- or tetrasubstituted alkene "cis to what?" has no single answer, so only E/Z is
reported:

```console
$ ligand3d stereo "CC/C(=C(\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1"
  double bond 2-3: Z (cis/trans does not apply: tetrasubstituted alkene)
```

Tamoxifen is the standard trap: unambiguously (Z) by CIP, and described as "trans" in
older literature with respect to the two phenyls. Both statements are about the same
molecule.


## Looking a molecule up by name

Drawing a fused polycyclic by hand is slow and easy to get subtly wrong, and most molecules
worth building already have a name. `ligand3d fetch` turns a name, SMILES, InChI, or PubChem
CID into a structure; in the sketcher there is an import box above the canvas that drops the
result straight onto it, so it becomes a scaffold you edit rather than a drawing you start
from nothing.

```bash
ligand3d fetch "3-Cyano-7-ethoxycoumarin"       # systematic name, resolved offline
ligand3d fetch aspirin                          # trivial name, via PubChem
ligand3d fetch cid:2244                         # straight to a PubChem record
ligand3d fetch "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
ligand3d fetch PLYNGSG --type peptide           # a sequence
ligand3d fetch GGCAT   --type dna
ligand3d fetch coumarin -o scaffold.sdf         # a 2D file to open in anything
ligand3d fetch --templates                      # the built-in scaffolds
ligand3d build "$(ligand3d fetch aspirin --smiles)" -o aspirin.cif
```

### Three routes, and why it matters which one answered

| route | what it handles | needs |
|---|---|---|
| parsing | SMILES, InChI | nothing |
| **OPSIN** | systematic IUPAC names | `py2opsin` and a `java` |
| **PubChem** | trivial names, trade names, CIDs | the network |
| **sequences** | peptides, DNA, RNA — see below | nothing |

These are complements, not fallbacks for each other, and `fetch` always says which one
answered:

```
$ ligand3d fetch "3-Cyano-7-ethoxycoumarin"
C12H9NO3  CCOc1ccc2cc(C#N)c(=O)oc2c1
  · derived from the name by OPSIN, offline, from IUPAC rules
```

OPSIN is a **grammar for the naming rules**, not a database. It parses names nobody has ever
catalogued, works with no network, and returns exactly what the name says — so it cannot
hand you the wrong compound, but it knows nothing about `aspirin`, because that name is not
derivable from anything.

PubChem is the opposite: a **lookup**, so it covers every trivial and trade name, and it can
quietly give you something you did not mean. Searching the name `CO` returns cobalt, not
carbon monoxide. That is why the matched title is always printed — read it.

For the record, [InChI](https://github.com/IUPAC-InChI/InChI) does not help here. It is a
structure *serialization*: it turns a structure into a canonical string and back. An InChI
string is accepted as input, but InChI cannot turn `aspirin` into a structure. OPSIN and
PubChem are the two things that do.

### Sequences: peptides, DNA and RNA

Typing `PLYNGSG` beats drawing fifty atoms, and an oligonucleotide is not
realistically drawable at all. Pick the import type — **Peptide**, **DNA** or **RNA** —
and type the sequence:

```bash
ligand3d fetch PLYNGSG   --type peptide      # N to C
ligand3d fetch GGCAT     --type dna          # 5' to 3'
ligand3d fetch AUGGC     --type rna
ligand3d fetch "GS(KCX)PL" --type peptide    # carboxylated lysine, mid-chain
ligand3d fetch --residues --type dna         # what codes exist
```

**Sequences are never auto-detected**, and that is deliberate: `GGCAT` is a perfectly
good DNA oligo and an equally good pentapeptide. Guessing would sometimes silently build
the wrong polymer, so the type has to be chosen. Auto-detect still handles everything
else, and says so when a failed lookup looks like a sequence.

RDKit builds the canonical alphabets, and is used directly for them. What it will not do
is anything modified — a phosphoserine, a carboxylated lysine, an inosine — which is most
of the reason to want this. Those come from a residue library here, and **that library is
checked against RDKit rather than trusted**: every canonical residue and chain built by
ligand3d is asserted to be the same molecule RDKit produces. A hand-written SMILES with an
inverted stereocentre gives a plausible-looking peptide of the wrong enantiomer, and a
misdrawn ribose gives a 2'-5' linked RNA; neither is something you would catch by eye, so
the test suite catches them instead. Every nucleoside is additionally checked against its
known molecular formula.

Longer codes go in parentheses, following the **PDB Chemical Component Dictionary**, so
`(SEP)` here is `SEP` in a PDB file. The sketcher lists the alphabet for whichever type is
selected, with the full residue name on hover.

| | codes |
|---|---|
| **peptide** | the 20, plus `U` (selenocysteine) and `O` (pyrrolysine) from the expanded alphabet |
| **PTMs** | `SEP` `TPO` `PTR` phospho-Ser/Thr/Tyr · `KCX` carboxy-Lys · `ALY` acetyl-Lys · `MLZ` `MLY` `M3L` mono/di/tri-methyl-Lys · `TYS` sulfo-Tyr · `HYP` hydroxyproline · `PCA` pyroglutamate · `CSO` `CSD` `OCS` oxidised Cys · `NIY` nitro-Tyr · `CIR` citrulline |
| **ncAAs** | `MSE` `SEC` `ORN` `NLE` `NVA` `ABA` `DAB` `AIB` `SAR` `HCS` `HSE` `PFF` `AZF` `BIF`, plus `DAL` `DVA` `DPR` for D residues |
| **DNA** | `A C G T`, plus `U` `I` (deoxyinosine) `5MC` `8OG` `BRU` |
| **RNA** | `A C G U`, plus `T` `I` (inosine) `PSU` (pseudouridine) `5MC` `6MA` `7MG` |

Anything non-standard is reported rather than absorbed silently:

```
$ ligand3d fetch "GS(KCX)PL" --type peptide
C23H40N6O9  CC(C)C[C@H](NC(=O)[C@@H]1CCCN1C(=O)[C@H](CCCCNC(=O)O)NC(=O)...
  · non-standard residue(s): KCX = N6-carboxylysine
  · built N terminus first, with a free amine and a free acid
```

IUPAC ambiguity codes are refused with the reason: `N` in a DNA sequence means *any base*,
which is a set of sequences rather than a molecule.

**Protonation.** Sequences are built neutral — free amine, free acid, protonated
phosphates — and the pH from the Chemistry tab is applied when you build, not at import.
Doing it in both places would protonate twice. The import notice says which will happen:
with protonation set to *as drawn* it tells you the structure is neutral and how to change
that, and with a pH set it says the termini and ionizable side chains will be adjusted to
that pH. The default is 7.4.

Chains are built with free hydroxyls at both ends and no terminal phosphate, matching what
RDKit's uncapped flavors produce.

### Scaffolds

`ligand3d fetch --templates` lists eighteen starting points — benzene, piperidine,
morpholine, indole, coumarin, purine, adamantane, the gonane skeleton, and so on — also
available from the dropdown next to the import box. They exist for when the point is to draw
a derivative, so the list is deliberately short: a starting point, not a compound library.

Anything imported is **2D on purpose**. It is a drawing to edit, and writing it out with
3D-looking coordinates would invite someone to mistake a layout for a geometry. The
embedding happens later, in `build`, from whatever you edited it into.


## Drawing

`ligand3d sketch` starts a local server, opens a browser, and stays up. Draw a structure,
set the options, press **Build**, read the run log, clear the canvas, and draw the next
one — no reloading between molecules. There is an import box above the canvas (see
[Looking a molecule up by name](#looking-a-molecule-up-by-name)) for starting from a
named compound or a scaffold instead of an empty sheet.

```bash
ligand3d sketch                          # output goes to the current directory
ligand3d sketch -d ~/ligands -b gfn2     # preselect a directory and backend
ligand3d sketch --no-browser             # print the URL (SSH port-forwarding)
```

The page is laid out as editor on the left, work area on the right: settings grouped into
four tabs (Output, Minimize, Chemistry, Rosetta) so the panel stays short, then the energy
graph, then the run log. Both the editor and the log grow to fill the window.

- **Output path and formats**, shown resolved in full before anything is written. The name
  field is a base name, since one build can produce an mmCIF, a PDB, an SDF, a trajectory
  and a params set; tick the formats you want. A directory that does not exist yet is
  flagged and created on build; an unwritable location is an error before you spend a
  minimization on it.
- **Auto-incrementing names** — `sketch0`, `sketch1`, skipping any base name already taken
  in *any* format. If a build would replace existing files you get a dialog listing exactly
  which, and nothing is written until you confirm.
- **Every build option**, including the Rosetta params tab and the trace and trajectory
  toggles. Each maps to a `ligand3d build` flag.
- **An energy graph** when tracing is on: one curve per method, each plotted as the change
  from its own first step (a strain energy and a total electronic energy share no scale),
  with the step count and net change in the legend.
- **A run log** reporting stereocenters with R/S, double bonds with E/Z and cis/trans,
  warnings such as more than one fragment, any error in full, per-method timing, the total
  time, and every file written. There is a Copy button.

### Seeing how your drawing is read

Under the editor is a live panel showing the molecule **as ligand3d parses it**, redrawn
by RDKit with every atom numbered, updating as you draw.

This exists because of one specific failure. When the pipeline says

```
2 stereocenter(s) left undefined: atom 4, atom 6
```

those numbers index the file your sketcher emitted, and there is no way to tell from the
canvas which atoms they are — the same molecule drawn twice can even number differently.
The panel makes the number point at something you can see. Atoms are colour-coded:

- **amber** — needs a configuration from you
- **green** — already specified, with its R/S annotation
- **grey** — looks stereogenic to a graph analysis but is fixed by the ring system, so
  there is nothing to decide

Atom numbering can be toggled off, and the panel follows your light or dark theme.

The messages also say *what kind* of stereochemistry is missing, which is often not what
you would guess. Two flagged atoms on the same ring are not two independent wedges to
draw — they are one cis/trans relationship:

```
atoms 3, 5 sit on the same 4-membered ring, so the ambiguity is whether their
substituents are on the same face (cis) or opposite faces (trans). Put a wedge on
one substituent bond and a wedge or a dash on the other to say which.
```

### Drawing stereochemistry — wedges *and* dashes

Select the wedge-bond tool, then **click the same bond repeatedly to cycle it**: solid
wedge → dashed (hashed) → plain. There is no separate dash tool; the one tool cycles. A
wedge and a dash on the same drawing give opposite configurations, which is verified in
the test suite — molfile bond flag 1 reads back as *R* and flag 6 as *S* for the same 2D
layout.

Whatever you draw is checked: after embedding and again after minimization the CIP labels
are re-perceived from the 3D coordinates and compared against your drawing, so a
stereocenter cannot silently flip.

### The editor

[JSME](https://jsme-editor.github.io/) is fetched once (about 1 MB) into
`~/.cache/ligand3d/` and works offline thereafter. Nothing is sent anywhere: the server
binds `127.0.0.1` only, and it shuts down when you stop it.

If it cannot be fetched the same page shows a paste box instead, accepting a molblock or a
SMILES string. Every other control — settings, run log, overwrite protection — works
unchanged, so an air-gapped machine loses only the drawing canvas.
