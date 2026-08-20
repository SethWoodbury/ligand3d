"""Regression tests for bugs found in review.

Each of these produced a wrong molecule, a crash, or a spurious rejection.
They are grouped by the defect rather than by module so the reason each one
exists stays legible.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from ligand3d.embed import embed, perceive_stereo_3d
from ligand3d.errors import InputError, ProtonationError, StereoError
from ligand3d.minimize import MinimizeResult, register
from ligand3d.minimize.base import Availability, Capabilities
from ligand3d.molecule import from_file, from_smiles, largest_fragment
from ligand3d.pipeline import Settings, build
from ligand3d.protonate import assert_connectivity_intact, assert_protonation_intact
from ligand3d.write import to_pdb_block


class _FakeBackend:
    """Minimal backend for driving the pipeline into specific failure modes."""

    def __init__(self, name: str, action, kind: str = "ff", fixed_topology: bool = True):
        self.caps = Capabilities(
            name=name,
            kind=kind,
            description="test double",
            takes_charge=True,
            fixed_topology=fixed_topology,
        )
        self._action = action

    def available(self) -> Availability:
        return Availability(ok=True)

    def minimize(self, job):
        return self._action(job)


class TestEveryConformerIsChecked:
    """Stereo was verified on the default conformer only.

    An optimizer that inverted conformers 1..n while leaving conformer 0 alone
    produced a file whose first model was right and whose remaining models were
    the enantiomer of the requested molecule, with no error raised.
    """

    def test_inverting_a_later_conformer_is_caught(self):
        def mirror_all_but_first(job):
            if job.conf_id != 0:
                conf = job.mol.GetConformer(job.conf_id)
                for i in range(job.mol.GetNumAtoms()):
                    p = conf.GetAtomPosition(i)
                    conf.SetAtomPosition(i, (-p.x, p.y, p.z))
            # conformer 0 sorts first, so it is the one a single check would see
            return MinimizeResult(
                energy=float(job.conf_id), converged=True, n_steps=1, backend="mirror-late"
            )

        register("mirror-late", lambda: _FakeBackend("mirror-late", mirror_all_but_first))
        with pytest.raises(StereoError, match="stereochemistry changed"):
            build(
                from_smiles("C[C@H](N)C(=O)OCC"),
                Settings(backend="mirror-late", n_confs=6, prune_rms=0.0),
            )


class TestExplicitHydrogenNumbering:
    """Stereo audit and 3D verification used different atom numberings.

    Readers keep explicit hydrogens, so a file listing its hydrogens first
    audited "stereocenter at atom 8" while the 3D check — which works on a
    heavy-atom-only view — reported atom 1. Every such file was rejected as
    having lost its stereochemistry.
    """

    @pytest.fixture
    def hydrogens_first_sdf(self, tmp_path):
        mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](N)C(=O)O"))
        AllChem.Compute2DCoords(mol)
        order = sorted(
            range(mol.GetNumAtoms()),
            key=lambda i: mol.GetAtomWithIdx(i).GetAtomicNum() != 1,
        )
        path = tmp_path / "hydrogens_first.sdf"
        Chem.MolToMolFile(Chem.RenumberAtoms(mol, order), str(path))
        return path

    def test_file_with_leading_hydrogens_is_accepted(self, hydrogens_first_sdf):
        molecule = from_file(hydrogens_first_sdf)
        assert molecule.smiles == "C[C@H](N)C(=O)O"
        assert dict(molecule.stereo.assigned_centers) == {1: "S"}

    def test_it_builds_end_to_end(self, hydrogens_first_sdf, tmp_path):
        outcome = build(from_file(hydrogens_first_sdf), Settings())
        recovered = perceive_stereo_3d(outcome.mol_3d, conf_id=0)
        assert dict(recovered.assigned_centers) == {1: "S"}

    def test_deuterium_is_not_discarded(self):
        """RemoveHs keeps isotope-labelled hydrogens; normalization must not."""
        molecule = from_smiles("[2H]C(Cl)(Br)F")
        assert any(a.GetIsotope() == 2 for a in molecule.mol.GetAtoms())


class TestDisconnectedFragments:
    """ETKDG has no restraints between components and stacks them.

    Measured inter-fragment distance for "CCO.CCO" was 0.0 A, and
    "CC(=O)[O-].[Na+]" previously succeeded silently with the sodium 0.64 A
    from an oxygen.
    """

    @pytest.mark.parametrize("smiles", ["CCO.CCO", "CC(=O)[O-].[Na+]", "[Na+].[Cl-]"])
    def test_multi_fragment_input_is_refused(self, smiles):
        with pytest.raises(InputError, match="disconnected fragments"):
            build(from_smiles(smiles), Settings())

    def test_largest_fragment_opt_in_keeps_the_ligand(self):
        outcome = build(
            from_smiles("CC(=O)[O-].[Na+]"), Settings(largest_fragment=True)
        )
        assert outcome.molecule.smiles == "CC(=O)[O-]"
        assert any("discarded" in note for note in outcome.notes)

    def test_single_fragment_still_builds(self):
        build(from_smiles("CCO"), Settings())

    def test_largest_fragment_helper_picks_by_heavy_atom_count(self):
        kept = largest_fragment(from_smiles("C.CCCCCCCC"))
        assert kept.smiles == "CCCCCCCC"


class TestPartialMinimizationFailure:
    """Conformers that failed to minimize were ranked +inf but still kept.

    The caller then looked up an energy that was never recorded and died with a
    KeyError. gfn2 reaches this for real on close contacts.
    """

    def test_pipeline_survives_some_conformers_failing(self):
        from ligand3d.errors import MinimizationError

        def fail_odd(job):
            if job.conf_id % 2 == 1:
                raise MinimizationError(f"synthetic failure on {job.conf_id}")
            return MinimizeResult(
                energy=float(job.conf_id), converged=True, n_steps=1, backend="flaky"
            )

        register("flaky", lambda: _FakeBackend("flaky", fail_odd))
        outcome = build(
            from_smiles("CCCCCCCC(=O)NC"),
            Settings(backend="flaky", n_confs=8, prune_rms=0.0),
        )
        assert outcome.records
        assert all(record.energy is not None for record in outcome.records)
        assert any("failed to minimize" in note for note in outcome.notes)


class TestAtomNameUniqueness:
    """`" C100"` truncated to `" C10"` and collided with the real C10.

    PDB readers key on atom name within a residue, so duplicates corrupt the
    file while it still looks fine in a text editor.
    """

    @pytest.mark.parametrize("n_carbons", [49, 120])
    def test_long_alkanes_keep_unique_names(self, n_carbons):
        block = to_pdb_block(embed(from_smiles("C" * n_carbons)))
        names = [
            line[12:16] for line in block.splitlines() if line.startswith(("ATOM", "HETATM"))
        ]
        assert len(set(names)) == len(names)

    def test_names_stay_within_four_columns(self):
        block = to_pdb_block(embed(from_smiles("C" * 120)))
        for line in block.splitlines():
            if line.startswith(("ATOM", "HETATM")):
                assert len(line[12:16]) == 4


class TestHeavyElementBondCutoffs:
    """A fixed 1.45 A default was shorter than real X-H bonds.

    Se-H is 1.46 A and As-H 1.52 A, so these molecules were reported as having
    lost a hydrogen that never moved. Cutoffs now come from RDKit's covalent
    radii.
    """

    @pytest.mark.parametrize("smiles", ["C[SeH]", "[AsH3]", "[SnH4]", "C[GeH3]", "C[TeH]"])
    def test_heavy_hydrides_are_not_false_positives(self, smiles):
        mol_3d = embed(from_smiles(smiles))
        assert_protonation_intact(mol_3d, conf_id=0)

    def test_molecular_hydrogen_has_no_heavy_partner_to_lose(self):
        mol_3d = embed(from_smiles("[H][H]"))
        assert_protonation_intact(mol_3d, conf_id=0)

    def test_a_genuinely_moved_proton_is_still_caught(self):
        """The loosened cutoffs must not blind the check."""
        mol_3d = embed(from_smiles("[NH3+]CC1(CC(=O)[O-])CCCCC1"))
        conf = mol_3d.GetConformer(0)
        positions = np.array(conf.GetPositions())
        symbols = [a.GetSymbol() for a in mol_3d.GetAtoms()]
        nitrogen = symbols.index("N")
        oxygen = symbols.index("O")
        proton = next(
            i
            for i, s in enumerate(symbols)
            if s == "H" and np.linalg.norm(positions[i] - positions[nitrogen]) < 1.25
        )
        conf.SetAtomPosition(proton, (positions[oxygen] + np.array([0.98, 0, 0])).tolist())

        with pytest.raises(ProtonationError, match="protonation state changed"):
            assert_protonation_intact(mol_3d, conf_id=0)


class TestConnectivityCheck:
    """Nothing verified heavy-atom bonds, which the spec promised.

    A potential that works from positions alone can break a C-C bond; bond
    orders are fixed in the RDKit molecule, so the only trace is two atoms
    implausibly far apart.
    """

    def test_intact_structure_passes(self):
        assert_connectivity_intact(embed(from_smiles("CC(=O)OC")), conf_id=0)

    def test_broken_bond_between_hydrogenless_atoms_is_caught(self):
        mol_3d = embed(from_smiles("CC(=O)OC"))
        conf = mol_3d.GetConformer(0)
        ester_oxygen = next(
            a.GetIdx()
            for a in mol_3d.GetAtoms()
            if a.GetSymbol() == "O" and a.GetDegree() == 2
        )
        p = conf.GetAtomPosition(ester_oxygen)
        conf.SetAtomPosition(ester_oxygen, (p.x + 6.0, p.y, p.z))

        # The hydrogen check cannot see this: no H moved.
        assert_protonation_intact(mol_3d, conf_id=0)
        with pytest.raises(ProtonationError, match="connectivity changed"):
            assert_connectivity_intact(mol_3d, conf_id=0)

    def test_pipeline_rejects_a_bond_breaking_backend(self):
        def yank_first_atom(job):
            conf = job.mol.GetConformer(job.conf_id)
            p = conf.GetAtomPosition(0)
            conf.SetAtomPosition(0, (p.x + 8.0, p.y, p.z))
            return MinimizeResult(
                energy=-1.0, converged=True, n_steps=1, backend="breaker"
            )

        register(
            "breaker",
            lambda: _FakeBackend("breaker", yank_first_atom, kind="mlff", fixed_topology=False),
        )
        with pytest.raises(ProtonationError):
            build(from_smiles("CCCCO"), Settings(backend="breaker"))


class TestBondStereoNormalization:
    """CIS/TRANS and E/Z are different conventions and must not be compared.

    They are also never mixed in practice, because every audit re-runs
    AssignStereochemistry first — this pins that down.
    """

    @pytest.mark.parametrize(
        "smiles",
        [
            r"C/C=C/C(=O)O",
            r"C/C=C\C(=O)O",
            r"CC/C(=C(\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
            r"OC(=O)/C=C\C(=O)O",
        ],
    )
    def test_double_bond_geometry_round_trips(self, smiles):
        molecule = from_smiles(smiles)
        recovered = perceive_stereo_3d(embed(molecule))

        want = {(a, b): label for a, b, label in molecule.stereo.assigned_bonds}
        got = {(a, b): label for a, b, label in recovered.assigned_bonds}
        assert want
        assert all(label in ("E", "Z") for label in want.values())
        assert want == {key: got.get(key) for key in want}


class TestDryRunWritesNothing:
    """`--dry-run` was still leaving a file behind.

    The default output name carries an extension, and the "an explicit
    extension is a format request" rule then put that format straight back into
    the list the dry run had just emptied. The web path had a test for this;
    the CLI path did not, so only the CLI was broken.
    """

    def test_no_file_appears(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from ligand3d.cli import app

        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(app, ["build", "O=C1CN2CCC1CC2", "--dry-run"])

        assert result.exit_code == 0
        assert list(tmp_path.iterdir()) == []

    def test_still_reports_the_energy_it_computed(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from ligand3d.cli import app

        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(app, ["build", "CCO", "--dry-run", "--no-trace"])

        assert "lowest strain energy" in result.stdout
        assert list(tmp_path.iterdir()) == []

    def test_an_explicit_extension_is_still_honoured_normally(self, tmp_path, monkeypatch):
        # The rule the fix narrowed must keep working when it is not a dry run.
        from typer.testing import CliRunner

        from ligand3d.cli import app

        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(
            app, ["build", "CCO", "-o", str(tmp_path / "x.pdb"), "--no-trace"]
        )

        assert result.exit_code == 0
        assert (tmp_path / "x.pdb").exists()


class TestPhosphateIsNotAStereocenter:
    """NAD+ was refused for undefined stereochemistry at its two phosphorus atoms.

    A phosphate diester P has four different neighbours on paper — two bridging
    oxygens, a double-bonded one and a charged one — so graph analysis calls it
    a stereocenter. It is not: the terminal oxygens are one delocalized system,
    and where the proton sits is a matter of pH. Nothing can be drawn to resolve
    it, so demanding that someone resolve it made the molecule unbuildable.
    """

    NAD = (
        "NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)([O-])OP(=O)(O)OC[C@H]3O"
        "[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1"
    )
    ATP = (
        "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)"
        "[C@@H](O)[C@H]1O"
    )

    def test_nad_is_accepted(self):
        from ligand3d.molecule import require_defined_stereo

        molecule = from_smiles(self.NAD)
        assert molecule.stereo.unassigned_centers  # RDKit still flags them
        require_defined_stereo(molecule)  # but they are not a real ambiguity

    def test_atp_is_accepted(self):
        from ligand3d.molecule import require_defined_stereo

        require_defined_stereo(from_smiles(self.ATP))

    def test_nad_builds(self):
        outcome = build(from_smiles(self.NAD), Settings(backend="mmff94", sample=1))
        assert outcome.mol_3d.GetNumConformers() == 1

    def test_the_phosphorus_atoms_are_named_as_the_reason(self):
        from ligand3d.molecule import resonance_averaged_centers

        molecule = from_smiles(self.NAD)
        assert resonance_averaged_centers(molecule) == list(
            molecule.stereo.unassigned_centers
        )

    def test_the_explanation_says_why_rather_than_going_silent(self):
        from ligand3d.molecule import describe_resonance_centers

        text = describe_resonance_centers(from_smiles(self.NAD))
        assert "atom 13" in text and "atom 17" in text
        assert "delocalized" in text
        assert "not real stereocenters" in text

    def test_nothing_is_left_for_the_user_to_fix(self):
        from ligand3d.molecule import classify_undefined_stereo

        assert classify_undefined_stereo(from_smiles(self.NAD)) == []


class TestRealPhosphorusChiralitySurvives:
    """The fix must not erase stereochemistry that genuinely exists.

    Rp and Sp phosphorothioates are different compounds with different
    biochemistry, and a chiral phosphotriester is a textbook P stereocenter.
    Suppressing those would be a worse bug than the one being fixed.
    """

    @pytest.mark.parametrize(
        "name, smiles",
        [
            ("phosphorothioate diester, =O and S-", "CO[P](=O)([S-])OCC"),
            ("chiral phosphotriester", "CO[P](=O)(OCC)OC(C)C"),
            ("methylphosphonate, one terminal O", "C[P](=O)(OC)OCC"),
            ("sulfoxide", "C[S](=O)CC"),
        ],
    )
    def test_still_refused_as_ambiguous(self, name, smiles):
        from ligand3d.molecule import has_real_stereo_ambiguity, require_defined_stereo

        molecule = from_smiles(smiles)
        assert has_real_stereo_ambiguity(molecule), name
        with pytest.raises(StereoError):
            require_defined_stereo(molecule)

    @pytest.mark.parametrize(
        "smiles", ["CO[P](=O)([O-])OCC", "CO[P](=S)([S-])OCC", "CO[S](=O)[O-]"]
    )
    def test_matched_terminal_pairs_are_averaged(self, smiles):
        """Two of the same element are interchangeable; =O with S- is not."""
        from ligand3d.molecule import is_resonance_averaged_center

        molecule = from_smiles(smiles)
        centre = next(
            a.GetIdx() for a in molecule.mol.GetAtoms() if a.GetSymbol() in ("P", "S")
            and a.GetDegree() > 2
        )
        assert is_resonance_averaged_center(molecule.mol, centre)

    def test_a_defined_phosphorothioate_keeps_its_configuration(self):
        molecule = from_smiles("CO[P@](=O)([S-])OCC")
        assert molecule.stereo.assigned_centers


class TestAvailabilityChecksStayCheap:
    """Asking what a backend can do must not load what it would need to run.

    The MACE-POLAR check did `from mace.modules import extensions`, which pulls
    in torch and the whole MACE stack — 5.7 seconds. It was invisible while
    graph_longrange was missing, because the check above it short-circuited
    first; installing that package put a six-second stall in front of every
    sketcher page load, since the page asks for the backend list before it can
    render.
    """

    def test_no_backend_imports_torch_to_answer(self):
        import subprocess
        import sys

        # A subprocess, because torch is already imported in this one.
        probe = subprocess.run(
            [
                sys.executable, "-c",
                "import sys\n"
                "from ligand3d.minimize import all_backends\n"
                "for b in all_backends():\n"
                "    try: b.available()\n"
                "    except Exception: pass\n"
                "heavy = [m for m in ('torch', 'mace', 'ase') if m in sys.modules]\n"
                "print(','.join(heavy))\n",
            ],
            capture_output=True, text=True, timeout=180,
        )
        assert probe.returncode == 0, probe.stderr[-2000:]
        loaded = probe.stdout.strip()
        assert not loaded, f"availability probing imported {loaded}"

    def test_building_the_catalog_is_fast(self):
        """A wall-clock bound, because that is what the user actually felt."""
        import subprocess
        import sys
        import time

        started = time.perf_counter()
        probe = subprocess.run(
            [sys.executable, "-c",
             "from ligand3d.catalog import build_catalog; build_catalog()"],
            capture_output=True, text=True, timeout=120,
        )
        elapsed = time.perf_counter() - started
        assert probe.returncode == 0, probe.stderr[-2000:]
        # Interpreter start plus RDKit is most of this; the regression was 7s of
        # catalog on top. Generous so a slow shared filesystem cannot flake it.
        assert elapsed < 15, f"catalog took {elapsed:.1f}s"

    def test_polar_is_detected_without_importing_mace(self):
        import importlib.util

        if importlib.util.find_spec("mace") is None:
            pytest.skip("mace not installed")
        import subprocess
        import sys

        probe = subprocess.run(
            [sys.executable, "-c",
             "import sys\n"
             "from ligand3d.minimize.mlff import _mace_has_polar\n"
             "found = _mace_has_polar()\n"
             "print(found, 'torch' in sys.modules, 'mace' in sys.modules)\n"],
            capture_output=True, text=True, timeout=120,
        )
        found, torch_loaded, mace_loaded = probe.stdout.split()
        assert found == "True"
        assert torch_loaded == "False" and mace_loaded == "False"
