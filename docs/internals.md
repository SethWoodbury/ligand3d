# How it works

What the optimizer actually does, measured timings and accuracy, and how to add a method.

[← back to the README](../README.md)

## How a minimization actually works

Two separate things are involved, and it helps to keep them apart.

**The potential** answers one question: given these atomic positions, what is the energy,
and what is the force on each atom? That is all MMFF94, GFN2-xTB, MACE, or AIMNet2 ever
do. They differ enormously in how they answer it, and not at all in what they are asked.

**The optimizer** takes those forces and decides where to move the atoms next. ligand3d
uses **L-BFGS** for every backend except the RDKit force fields.

The loop is just:

```
positions ──▶ potential ──▶ energy + forces ──▶ L-BFGS proposes new positions ──▶ repeat
                                                        │
                                              stop when max force < fmax
```

### L-BFGS, briefly

L-BFGS is a quasi-Newton method. Steepest descent would step straight downhill along the
force, which zig-zags badly in a narrow valley — and a molecule's energy surface is full of
narrow valleys, because stretching a bond costs far more than rotating a torsion. Newton's
method would fix that using the Hessian (all second derivatives), but for a 29-atom
molecule that is an 87×87 matrix that nobody wants to compute or invert every step.

L-BFGS splits the difference: it *approximates* the inverse Hessian from the last handful
of position and gradient changes — the "limited memory" part — and never forms the matrix.
The result converges in tens of steps where steepest descent would take thousands, at
almost no extra cost per step. Every timing in this README is dominated by the potential,
not by L-BFGS.

ligand3d uses `ase.optimize.LBFGS` and stops at `fmax = 0.05 eV/Å` by default. The RDKit
force fields are the exception: they use RDKit's own C++ minimizer, because round-tripping
coordinates through ASE for a four-millisecond job costs more than the job.

### Where ACE comes in — and where it does not

ACE (Atomic Cluster Expansion) is **not** an optimizer and does not drive anything. It is
a way of *describing an atom's environment*: a systematic expansion in body order — how
this atom sits relative to one neighbour, then pairs of neighbours, then triples — built
from a basis of radial functions and spherical harmonics. Its appeal is that it is
systematic: turn up the body order and the expansion gets more expressive in a controlled
way, rather than by adding another ad-hoc term.

**MACE** uses that idea inside a message-passing neural network. Each atom carries a
feature vector; neighbours exchange messages along edges; and the many-body ACE features
are formed by symmetric contraction of tensor products of those messages. The pieces are
visible in the code — `EquivariantProductBasisBlock`, `SymmetricContraction` — and
"equivariant" is the load-bearing word: rotate the molecule and the predicted forces
rotate with it, exactly, by construction rather than by training. A model that has to
*learn* rotational symmetry from data wastes capacity on it and never gets it exactly
right.

So: ACE is the functional form of the energy. L-BFGS moves the atoms. They are unrelated
parts of the machine.

### What the other tiers are doing

- **MMFF94 / UFF** — a hand-fitted sum of springs: bond stretches, angle bends, torsions,
  van der Waals, electrostatics from fixed partial charges. The bond list is fixed at
  setup, which is why these cannot move a proton and why a zwitterion is safe with them.
  No electrons anywhere in the model.
- **GFN-FF** — the same idea, but with parameters generated across the periodic table and
  a topology perceived on the fly.
- **GFN1/GFN2-xTB** — genuinely semi-empirical *quantum* methods. They solve a simplified
  self-consistent tight-binding problem with a minimal basis, so electrons are represented
  and charge can redistribute. That is why they can break and form bonds, why they need a
  total charge, and why they are the only tier here with an implicit solvent model.
- **AIMNet2, MACE, eSEN, UMA** — neural potentials trained to reproduce DFT energies and
  forces. No electrons at run time; they have learned the *result* of the electronic
  structure calculation. Fast, and only as trustworthy as the chemistry in their training
  set.

### MACE-POLAR is a different shape

Standard MACE is strictly local: each atom sees neighbours inside a cutoff, usually 5–6 Å.
That is fine for bonded geometry and terrible for anything where a charge on one end of
the molecule matters to the other end, because Coulomb interactions fall off as 1/r and
simply do not fit inside a cutoff.

MACE-POLAR (`PolarMACE`, a subclass of the standard `ScaleShiftMACE`) adds an explicit
long-range term. It predicts atomic multipoles — charges, and optionally dipoles and
quadrupoles — then evaluates the electrostatics in reciprocal space with a k-space cutoff,
Ewald style. It can also run the field response to self-consistency, iterating until the
induced multipoles stop changing, which is what makes it *polarizable* rather than just
long-range. That machinery lives in the separate `graph_longrange` package
([graph_electrostatics](https://github.com/WillBaldwin0/graph_electrostatics)).

**This is not free**, and it is the honest answer to "does it really run that fast?": it
does not. Measured on this machine, MACE-POLAR is roughly ten times slower per L-BFGS step
than plain MACE-OFF, because every step now includes a reciprocal-space sum and possibly a
self-consistency loop:

| model | per L-BFGS step | full minimization |
|---|---|---|
| `mace-off-small` | 0.075 s | 2.0 s |
| `mace-off` | 0.25 s | 6.8 s |
| `mace-polar-s` | 0.74 s | 23 s |
| `mace-polar` | 1.8 s | 57 s |
| `mace-polar-l` | 3.6 s | 114 s |

An earlier version of this table carried estimates rather than measurements and had
MACE-POLAR at 4/8/16 s — wrong by a factor of six to seven, in the flattering direction.
Everything above was timed.


## Timings

All numbers are a **full minimization of gabapentin (29 atoms) on CPU**, no GPU involved:
a 13th-gen Intel i5-13500 with `OMP_NUM_THREADS=8`, torch's `+cpu` build. Roughly 26–32
L-BFGS steps in every case, so the per-step cost is what actually differs.

| backend | minimization | first-call load | note |
|---|---|---|---|
| `mmff94`, `uff` | 4 ms | – | no model to load |
| `gfnff` | 0.06 s | 1.5 s | xtb optimizes internally |
| `gfn2` | 0.33 s | – | tblite; the best value here |
| `gfn1` | 0.57 s | – | |
| `aimnet2` | 0.62 s | **14 s** | load dominates a single run |
| `mace-off-small` | 2.0 s | 3.6 s | |
| `mace-off` | 6.8 s | 0.5 s | |
| `mace-off-24` | 7.0 s | 0.5 s | |
| `mace-mp` | 7.8 s | 1.1 s | |
| `mace-mh-spice` | 7.9 s | 0.5 s | |
| `mace-mh` | 8.4 s | 0.4 s | |
| `mace-mh-1` | 12 s | 1.5 s | |
| `mace-polar-s` | 23 s | 0.2 s | long-range electrostatics |
| `mace-omol` | 28 s | 5.2 s | charge-aware MACE |
| `mace-off-large` | 31 s | 9.3 s | |
| `mace-polar` | 57 s | 0.2 s | |
| `mace-polar-l` | 114 s | 0.3 s | |

Every entry is now measured. The fairchem models were timed inside the container that
can run them, on the same molecule and the same machine, so they sit on the same scale as
the rest:

| backend | CPU | GPU (RTX PRO 6000) |
|---|---|---|
| `esen-sm-direct` | 2.8 s | 0.54 s |
| `esen` | 7.9 s | 1.1 s |
| `allscaip-direct` | 11 s | 1.1 s |
| `uma-s` | 15 s | 2.6 s |
| `uma-s-1p2p1` | 35 s | not timed on GPU |
| `esen-md-direct` | 16 s | 1.5 s |
| `allscaip` | 23 s | 2.5 s |
| `uma-s-1p2` | 36 s | 3.4 s |
| `uma-m` | 63 s | 14 s |

The old estimates were badly wrong in both directions — `uma-m` was guessed at "minutes"
and takes 63 s on CPU, while `uma-s` was guessed at 12 s and takes 15 s. All nine agree
on the energy to within 0.1 kcal/mol, which is the cross-check that says the numbers came
from the same calculation.

`uma-m` carries one caveat the table cannot: **11 GB of weights needs real memory.** It was
OOM-killed on a 31 GB workstation and ran without trouble on a node with 96 GB.

Two things worth noticing. `aimnet2` spends 14 seconds constructing itself and then
minimizes in 0.6 — for one molecule that is a bad trade, and for a hundred it is
irrelevant, since the calculator is built once and reused. And a conformer search
multiplies the minimization column by the number of conformers, which is exactly why
`--backend mmff94,mace-off` searches with MMFF94 and only refines the survivors: 60
conformers through MACE-OFF would be seven minutes to rediscover what MMFF94 found in a
quarter of a second.


## How accurate is any of this

`ligand3d models -v` prints, for every method, what it reproduces and how close it
reportedly gets. The methods page in the browser shows the same thing. Both are worth
reading before trusting a number, and both come with the same three caveats.

**An error bar means nothing without its reference.** A fitted method cannot be more right
than the thing it was fitted to. MACE-OFF reproduces ωB97M-D3(BJ)/def2-TZVPPD to about
1 kcal/mol — but that is agreement with *that functional*, which is itself off by a few
kcal/mol for some of the chemistry you will point this at. "1 kcal/mol against DFT" and
"1 kcal/mol against CCSD(T)" are different claims.

| method | reproduces | reported error |
|---|---|---|
| `uff` | rules, no single reference | geometries ~0.05 Å; conformer energies often several kcal/mol out |
| `mmff94` | MP2/6-31G* + experiment | bond lengths ~0.01–0.02 Å; conformer energies ~1 kcal/mol where parameterized |
| `gfnff` | GFN2-xTB geometries | heavy-atom RMSD ~0.1–0.3 Å vs DFT |
| `gfn2` | DFT (the GFN2 fit set) | bond lengths ~0.01–0.02 Å; conformer and reaction energies ~2–4 kcal/mol |
| `aimnet2` | ωB97M-D3/def2-TZVPP | ~1 kcal/mol energies, ~1–2 kcal/mol/Å forces |
| `mace-off*` | ωB97M-D3(BJ)/def2-TZVPPD (SPICE) | ~1 kcal/mol energies, ~1–2 kcal/mol/Å forces, neutral organics |
| `mace-omol`, `esen`, `uma*` | ωB97M-V/def2-TZVPD (OMol25) | sub-kcal/mol in domain; covers charged and open-shell |
| `mace-mp` | PBE / r2SCAN periodic DFT | tens of meV/Å on solids; **not fitted for molecular conformers** |
| `mace-polar*` | a polarizable set with explicit long-range terms | no settled benchmark to quote; charge-aware |

**Every one of those figures is in-domain.** They are errors on held-out data drawn from the
same distribution as the training set. That is what papers report and it is the number most
likely to mislead you, because a neural potential handed something unlike its training data
does not refuse and does not widen its error bar — it returns a confident number that can be
wrong by an order of magnitude. This is why ligand3d checks elements and total charge
*before* running a model rather than interpreting the result afterwards.

**Force error and energy error are different things**, and a method can be good at one and
poor at the other. Forces decide whether a minimization lands in the right geometry; energies
decide whether you can trust one conformer ranked above another. For picking a conformer, the
energy column binds.

The practical reading: for a **starting geometry**, everything in that table is fine and
`mmff94` is 4 ms. For **ranking conformers within a few kcal/mol**, you need `gfn2` or a
neural potential whose training set actually contains your chemistry. For anything charged,
you need a method with a charge channel at all — which is a capability question, not an
accuracy one, and ligand3d refuses those pairings rather than answering them.


## Adding a method

A backend declares what it can do and implements one function. The pipeline
reads the declaration and routes around it — the charge-channel refusal, the
zwitterion solvation check, the element check and the chaining all come from
`Capabilities` rather than from anything the backend does.

```python
class MyBackend(ASEBackend):
    caps = Capabilities(
        name="mymethod", kind="mlff",
        takes_charge=True, supports_solvation=False,
        fixed_topology=False, energy_kind="total",
        requires=("mypackage",),
    )

    def make_calculator(self, job):
        import mypackage
        return mypackage.ASECalculator(charge=job.charge)

register("mymethod", MyBackend)
```

That is the whole contract for anything with an ASE calculator, which is most
things. **DFT was added this way as the test of it**: one new module, plus three
one-line edits — adding `"dft"` to a `Literal`, to a module list, and to a sort
order. Charge, spin, solvent and chaining needed no work at all.

A new *checkpoint* for an existing family is smaller still: one `ModelSpec` in
`config.py` naming the file and what it was trained on. A new **classical**
method that is not ASE-shaped implements `minimize()` directly instead —
`rdkit_ff.py` and the GFN-FF path in `xtb.py` both do, because they optimise
internally rather than returning gradients.

Two things a new method should get right, because they are what the rest of the
tool reasons about:

- **`takes_charge`** decides whether a charged molecule is allowed near it. Get
  this wrong and a carboxylate is silently treated as a neutral acid.
- **`fixed_topology`** decides whether implicit solvation is forced for a
  zwitterion. A method that works from positions alone will happily move a
  proton and collapse one.
