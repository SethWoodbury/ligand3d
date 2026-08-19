"""mmCIF output, the energy trace, trajectories, and Rosetta params."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from rdkit import Chem

from ligand3d.embed import embed, embed_multi
from ligand3d.minimize import MinimizeJob, get_backend
from ligand3d.molecule import describe_double_bonds, from_smiles, read_3d
from ligand3d.pipeline import Settings, build, run
from ligand3d.write import ConformerRecord, write_cif, write_pdb, write_sdf, write_trajectory


def minimized(smiles: str, n_confs: int = 1):
    mol = embed_multi(from_smiles(smiles), n_confs=n_confs) if n_confs > 1 else embed(
        from_smiles(smiles)
    )
    for conformer in mol.GetConformers():
        get_backend("mmff94").minimize(MinimizeJob(mol=mol, conf_id=conformer.GetId()))
    return mol


class TestDoubleBondReporting:
    """E/Z is always defined; cis/trans only sometimes is.

    E/Z comes from CIP priorities. cis/trans compares two *reference*
    substituents, which is only meaningful when it is obvious which two are
    meant — that is, when each alkene carbon carries exactly one hydrogen.
    """

    @pytest.mark.parametrize(
        "smiles,cip,cis_trans",
        [
            (r"C/C=C/C", "E", "trans"),
            (r"C/C=C\C", "Z", "cis"),
            (r"OC(=O)/C=C/C(=O)O", "E", "trans"),   # fumaric
            (r"OC(=O)/C=C\C(=O)O", "Z", "cis"),     # maleic
            (r"C/C=C/C(=O)O", "E", "trans"),        # crotonic
        ],
    )
    def test_disubstituted_alkenes_get_both_labels(self, smiles, cip, cis_trans):
        reports = describe_double_bonds(from_smiles(smiles))
        assert len(reports) == 1
        assert reports[0].cip == cip
        assert reports[0].cis_trans == cis_trans

    @pytest.mark.parametrize(
        "smiles,label",
        [
            (r"CC/C(=C(\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1", "tetrasubstituted"),
            (r"C/C(Cl)=C/C", "trisubstituted"),
            (r"C/C=C(\C)C(=O)O", "trisubstituted"),
        ],
    )
    def test_more_substituted_alkenes_get_no_cis_trans(self, smiles, label):
        """Tamoxifen is the standard trap: unambiguously (Z), yet the same
        geometry is described as "trans" in older literature."""
        reports = describe_double_bonds(from_smiles(smiles))
        assert len(reports) == 1
        assert reports[0].cip in ("E", "Z")
        assert reports[0].cis_trans is None
        assert label in reports[0]._why

    def test_multiple_double_bonds_each_reported(self):
        reports = describe_double_bonds(from_smiles(r"C/C=C/C=C\C"))
        assert [r.cip for r in reports] == ["E", "Z"]
        assert [r.cis_trans for r in reports] == ["trans", "cis"]

    def test_no_double_bonds_gives_nothing(self):
        assert describe_double_bonds(from_smiles("CCO")) == []


class TestCifOutput:
    def test_carries_bond_orders_that_pdb_would_lose(self, tmp_path):
        molecule = from_smiles("O=C1CN2CCC1CC2")
        mol = embed(molecule)
        cif = write_cif(tmp_path / "q.cif", mol, resname="QUI")

        assert "_chem_comp_bond" in cif.read_text()
        assert Chem.MolToSmiles(Chem.RemoveHs(read_3d(cif))) == molecule.smiles

    def test_marks_aromatic_bonds(self, tmp_path):
        cif = write_cif(tmp_path / "b.cif", embed(from_smiles("c1ccccc1C(=O)O")))
        rows = [ln for ln in cif.read_text().splitlines() if ln.strip().endswith(" Y")]
        assert len(rows) == 6, "benzene ring has six aromatic bonds"

    def test_keeps_formal_charges(self, tmp_path):
        cif = write_cif(tmp_path / "z.cif", embed(from_smiles("[NH3+]CC(=O)[O-]")))
        back = read_3d(cif)
        charges = [a.GetFormalCharge() for a in back.GetAtoms()]
        assert any(c > 0 for c in charges) and any(c < 0 for c in charges)

    def test_multiple_conformers_become_models(self, tmp_path):
        mol = minimized("CCCCCCCCO", n_confs=4)
        cif = write_cif(tmp_path / "o.cif", mol)
        assert "pdbx_PDB_model_num" in cif.read_text()
        assert read_3d(cif).GetNumConformers() == mol.GetNumConformers()

    def test_records_provenance(self, tmp_path):
        records = [ConformerRecord(conf_id=0, energy=-1.5, backend="mmff94")]
        cif = write_cif(tmp_path / "p.cif", embed(from_smiles("CCO")), records=records)
        text = cif.read_text()
        assert "_ligand3d" in text
        assert "mmff94" in text

    @pytest.mark.parametrize("suffix", ["cif", "pdb", "sdf"])
    def test_every_format_round_trips_conformer_count(self, tmp_path, suffix):
        mol = minimized("CCCCCCCCO", n_confs=3)
        writers = {"cif": write_cif, "pdb": write_pdb, "sdf": write_sdf}
        path = writers[suffix](tmp_path / f"m.{suffix}", mol)
        assert read_3d(path).GetNumConformers() == 3


class TestEnergyTrace:
    """Per-step energies, with deltas, kept separate per method."""

    def test_off_by_default(self):
        mol = embed(from_smiles("CCCCCCCCO"))
        result = get_backend("mmff94").minimize(MinimizeJob(mol=mol, conf_id=0))
        assert result.trace == []
        assert result.frames == []
        assert result.wall_seconds > 0

    @pytest.mark.parametrize("backend_name", ["mmff94", "uff"])
    def test_records_a_monotonic_descent(self, backend_name):
        mol = embed(from_smiles("CCCCCCCCO"))
        result = get_backend(backend_name).minimize(
            MinimizeJob(mol=mol, conf_id=0, trace=True)
        )
        assert len(result.trace) > 2
        energies = [step.energy for step in result.trace]
        # A minimizer must not go uphill.
        assert all(b <= a + 1e-6 for a, b in zip(energies, energies[1:]))
        assert result.trace[0].delta is None
        assert all(step.delta is not None for step in result.trace[1:])

    def test_deltas_match_the_energies(self):
        mol = embed(from_smiles("CCCCCCCCO"))
        trace = get_backend("mmff94").minimize(
            MinimizeJob(mol=mol, conf_id=0, trace=True)
        ).trace
        for previous, step in zip(trace, trace[1:]):
            assert step.delta == pytest.approx(step.energy - previous.energy, abs=1e-9)

    def test_tracing_does_not_change_the_result(self):
        """The geometry must not depend on whether a log was requested.

        Single-stepping restarts RDKit's optimizer each call and so descends less
        efficiently; a final uninterrupted pass makes up the difference.
        """
        for smiles in ("CCCCCCCCO", "O=C1CN2CCC1CC2", "NCC1(CC(=O)O)CCCCC1"):
            energies = []
            for trace in (False, True):
                mol = embed(from_smiles(smiles))
                energies.append(
                    get_backend("mmff94")
                    .minimize(MinimizeJob(mol=mol, conf_id=0, trace=trace))
                    .energy
                )
            assert energies[0] == pytest.approx(energies[1], abs=1e-3), smiles

    def test_stage_is_recorded_so_chained_methods_stay_separable(self):
        mol = embed(from_smiles("CCO"))
        first = get_backend("mmff94").minimize(
            MinimizeJob(mol=mol, conf_id=0, trace=True, stage=0)
        )
        second = get_backend("uff").minimize(
            MinimizeJob(mol=mol, conf_id=0, trace=True, stage=1)
        )
        assert {s.stage for s in first.trace} == {0}
        assert {s.stage for s in second.trace} == {1}
        assert {s.backend for s in second.trace} == {"uff"}

    def test_pipeline_collects_a_trace_across_a_chain(self):
        if not get_backend("gfn2").available():
            pytest.skip("tblite not installed")
        outcome = build(
            from_smiles("CCO"), Settings(backend="mmff94,gfn2", trace=True, max_steps=50)
        )
        stages = {step.stage for step in outcome.trace}
        assert stages == {0, 1}
        backends = {step.backend for step in outcome.trace}
        assert backends == {"mmff94", "gfn2"}
        # Energies from two methods are not comparable, so no delta may bridge
        # the boundary.
        first_of_stage_one = next(s for s in outcome.trace if s.stage == 1)
        assert first_of_stage_one.delta is None

    def test_timing_is_reported(self):
        outcome = build(from_smiles("CCO"), Settings())
        assert outcome.wall_seconds > 0
        assert any("total time" in note for note in outcome.notes)


class TestTrajectory:
    def test_writes_one_model_per_frame(self, tmp_path):
        mol = embed(from_smiles("CCCCCCCCO"))
        result = get_backend("mmff94").minimize(
            MinimizeJob(mol=mol, conf_id=0, trace=True, trajectory=True)
        )
        assert len(result.frames) == len(result.trace)

        path = write_trajectory(
            tmp_path / "t.pdb", mol, result.frames,
            energies=[s.energy for s in result.trace],
            stage_labels=[f"stage {s.stage}: {s.backend}" for s in result.trace],
        )
        text = path.read_text()
        assert text.count("MODEL ") == len(result.frames)
        assert text.count("ENDMDL") == len(result.frames)
        assert "TRAJECTORY" in text
        assert "ENERGY" in text

    def test_frames_actually_differ(self, tmp_path):
        mol = embed(from_smiles("CCCCCCCCO"))
        result = get_backend("mmff94").minimize(
            MinimizeJob(mol=mol, conf_id=0, trajectory=True)
        )
        assert not np.allclose(result.frames[0], result.frames[-1])

    def test_pipeline_writes_the_file(self, tmp_path):
        outcomes = run(
            from_smiles("CCCCCCCCO"),
            Settings(trace=True, trajectory=True),
            output=tmp_path / "o.cif",
        )
        outcome = outcomes[0]
        assert outcome.trajectory_path is not None
        assert outcome.trajectory_path.exists()
        assert outcome.trajectory_path.name == "o_traj.pdb"

    def test_refuses_to_write_nothing(self, tmp_path):
        with pytest.raises(ValueError, match="no frames"):
            write_trajectory(tmp_path / "t.pdb", embed(from_smiles("CCO")), [])


_HAVE_MOLFILE_TO_PARAMS = None


def have_params() -> bool:
    global _HAVE_MOLFILE_TO_PARAMS
    if _HAVE_MOLFILE_TO_PARAMS is None:
        from ligand3d.config import find_molfile_to_params

        _HAVE_MOLFILE_TO_PARAMS = find_molfile_to_params() is not None
    return _HAVE_MOLFILE_TO_PARAMS


needs_rosetta = pytest.mark.skipif(
    not have_params(), reason="Rosetta molfile_to_params.py not found"
)


class TestParamsCodes:
    """Code handling needs no Rosetta install."""

    def test_normalizes_to_three_alphanumerics(self):
        from ligand3d.params import normalize_code

        assert normalize_code("lig") == "LIG"
        assert normalize_code("Z-01") == "Z01"
        assert normalize_code("abcdef") == "ABC"

    def test_rejects_a_code_with_nothing_usable(self):
        from ligand3d.params import ParamsError, normalize_code

        with pytest.raises(ParamsError):
            normalize_code("!!!")

    @needs_rosetta
    def test_detects_an_existing_rosetta_code(self):
        from ligand3d.params import code_conflict

        assert code_conflict("ALA") is not None
        assert code_conflict("TRP") is not None
        assert code_conflict("Z01") is None


@needs_rosetta
class TestParamsGeneration:
    def test_single_conformer(self, tmp_path):
        from ligand3d import params

        mol = minimized("c1ccccc1C(=O)O")
        result = params.generate(mol, code="Z01", out_dir=tmp_path, conformers=False)

        assert result.params.exists()
        text = result.params.read_text()
        assert text.count("aroC") == 6, "benzene carbons should type as aromatic"
        assert "PDB_ROTAMERS" not in text
        # molfile_to_params names the single-conformer structure NAME_0001.pdb,
        # not NAME.pdb; the writer must not assume either.
        assert result.pdb is not None and result.pdb.exists()

    def test_conformers_become_the_rotamer_library(self, tmp_path):
        from ligand3d import params

        mol = minimized("NCC1(CC(=O)O)CCCCC1", n_confs=5)
        n_confs = mol.GetNumConformers()
        result = params.generate(mol, code="Z02", out_dir=tmp_path)

        assert "PDB_ROTAMERS" in result.params.read_text()
        assert result.n_conformers == n_confs
        # Conformer 1 lives in NAME.pdb and has to be prepended, or the library
        # is silently short by one.
        assert params.count_rotamers(result.conformers, mol.GetNumAtoms()) == n_confs
        assert any("prepended" in note for note in result.notes)

    def test_atom_count_matches(self, tmp_path):
        from ligand3d import params

        mol = minimized("O=C1CN2CCC1CC2")
        result = params.generate(mol, code="Z03", out_dir=tmp_path)
        typed = [ln for ln in result.params.read_text().splitlines() if ln.startswith("ATOM ")]
        assert len(typed) == mol.GetNumAtoms()

    def test_refuses_an_existing_code_unless_overridden(self, tmp_path):
        from ligand3d import params

        mol = minimized("CCO")
        with pytest.raises(params.ParamsError, match="already a Rosetta residue type"):
            params.generate(mol, code="ALA", out_dir=tmp_path)

        result = params.generate(
            mol, code="ALA", out_dir=tmp_path, allow_code_conflict=True
        )
        assert result.params.exists()

    def test_refuses_a_single_atom(self, tmp_path):
        from rdkit.Chem import AllChem

        from ligand3d import params

        ion = Chem.MolFromSmiles("[Na+]")
        AllChem.Compute2DCoords(ion)
        with pytest.raises(params.ParamsError, match="single-atom"):
            params.generate(ion, code="Z04", out_dir=tmp_path)

    def test_through_the_pipeline(self, tmp_path):
        outcome = run(
            from_smiles("O=C1CN2CCC1CC2"),
            Settings(params=True, params_code="Z05", n_confs=3),
            output=tmp_path / "q.cif",
        )[0]
        assert outcome.params_result is not None
        assert outcome.params_result.params.exists()
        assert outcome.params_result.params in outcome.written()

    def test_params_from_a_written_file(self, tmp_path):
        """The `params` subcommand path: read a 3D file, then generate."""
        from ligand3d import params

        mol = minimized("NCC1(CC(=O)O)CCCCC1", n_confs=3)
        sdf = write_sdf(tmp_path / "g.sdf", mol)
        reloaded = read_3d(sdf)
        assert reloaded.GetNumConformers() == mol.GetNumConformers()

        result = params.generate(reloaded, code="Z06", out_dir=tmp_path)
        assert result.n_conformers == reloaded.GetNumConformers()


class TestConformerSearch:
    """`--confs 1` used to mean one guess, locally minimized.

    That made the answer depend on the random seed by many kcal/mol for anything
    flexible, so a single requested structure now still comes from a real
    search: `sample` conformers are generated and minimized cheaply, and only
    `n_confs` are kept.
    """

    def test_sampling_scales_with_flexibility(self):
        rigid = Settings().effective_sample(from_smiles("O=C1CN2CCC1CC2"))
        medium = Settings().effective_sample(from_smiles("NCC1(CC(=O)O)CCCCC1"))
        floppy = Settings().effective_sample(from_smiles("CCCCCCCCCCCCO"))
        assert rigid < medium < floppy
        assert rigid >= 20

    def test_an_explicit_sample_wins(self):
        assert Settings(sample=7).effective_sample(from_smiles("CCCCCCCCO")) == 7

    def test_sample_never_falls_below_what_was_asked_for(self):
        assert Settings(sample=2, n_confs=10).effective_sample(from_smiles("CCO")) == 10

    def test_crest_does_its_own_sampling(self):
        settings = Settings(conf_method="crest", n_confs=5)
        assert settings.effective_sample(from_smiles("CCCCCCCCO")) == 5

    def test_a_single_output_is_seed_stable_now(self):
        """The point of searching: the answer stops depending on the seed."""
        energies = [
            build(from_smiles("NCC1(CC(=O)O)CCCCC1"), Settings(seed=seed)).records[0].energy
            for seed in (1, 2, 3)
        ]
        assert max(energies) - min(energies) < 1.0

    def test_sample_one_skips_the_search(self):
        outcome = build(from_smiles("NCC1(CC(=O)O)CCCCC1"), Settings(sample=1))
        assert len(outcome.records) == 1

    def test_only_survivors_reach_the_expensive_backend(self):
        """The cheap method searches; the costly one refines what is left."""
        if not get_backend("gfn2").available():
            pytest.skip("tblite not installed")
        outcome = build(
            from_smiles("NCC1(CC(=O)O)CCCCC1"),
            Settings(backend="mmff94,gfn2", n_confs=2, max_steps=200),
        )
        assert any("narrowed" in note for note in outcome.notes)
        stages = {step.stage: step.backend for step in outcome.trace}
        # Both stages must appear: the traced conformer has to follow the
        # survivors, or the refinement stage records nothing at all.
        assert stages.get(0) == "mmff94"
        assert stages.get(1) == "gfn2"


class TestDryRun:
    def test_no_formats_writes_nothing(self, tmp_path):
        outcomes = run(
            from_smiles("CCO"), Settings(formats=()), output=tmp_path / "x.cif"
        )
        assert outcomes[0].written() == []
        assert list(tmp_path.iterdir()) == []

    def test_the_molecule_is_still_built_and_checked(self, tmp_path):
        outcome = run(
            from_smiles("CCO"), Settings(formats=()), output=tmp_path / "x.cif"
        )[0]
        assert outcome.mol_3d.GetNumConformers() >= 1
        assert outcome.best_energy is not None


class TestTraceDefault:
    def test_tracing_is_on_by_default(self):
        assert Settings().trace is True
        outcome = build(from_smiles("CCO"), Settings())
        assert outcome.trace, "the default build should record a trace"

    def test_it_can_be_turned_off(self):
        assert build(from_smiles("CCO"), Settings(trace=False)).trace == []
