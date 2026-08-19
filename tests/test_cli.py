"""The command-line surface: flags mean what the help says they mean."""

from __future__ import annotations

import importlib.util

import pytest
from typer.testing import CliRunner

from ligand3d.cli import app

runner = CliRunner()

needs_dimorphite = pytest.mark.skipif(
    importlib.util.find_spec("dimorphite_dl") is None, reason="dimorphite-dl not installed"
)

QUINUCLIDINONE = "O=C1CN2CCC1CC2"
GABAPENTIN = "NCC1(CC(=O)O)CCCCC1"


def run(*args):
    return runner.invoke(app, list(args))


class TestBuild:
    def test_builds_and_writes_both_files(self, tmp_path):
        out = tmp_path / "q.pdb"
        result = run("build", QUINUCLIDINONE, "-o", str(out))
        assert result.exit_code == 0, result.output
        assert out.exists()
        assert out.with_suffix(".sdf").exists()

    def test_quiet_prints_only_the_path(self, tmp_path):
        out = tmp_path / "q.pdb"
        result = run("build", QUINUCLIDINONE, "-o", str(out), "-q")
        assert result.exit_code == 0
        assert result.output.strip() == str(out)

    def test_no_sdf_suppresses_the_sidecar(self, tmp_path):
        out = tmp_path / "q.pdb"
        run("build", QUINUCLIDINONE, "-o", str(out), "--no-sdf")
        assert out.exists()
        assert not out.with_suffix(".sdf").exists()

    def test_output_suffix_is_normalized(self, tmp_path):
        result = run("build", QUINUCLIDINONE, "-o", str(tmp_path / "q.xyz"), "-q")
        assert result.exit_code == 0
        assert (tmp_path / "q.pdb").exists()

    def test_bad_smiles_exits_nonzero(self, tmp_path):
        result = run("build", "not a molecule", "-o", str(tmp_path / "x.pdb"))
        assert result.exit_code == 1
        assert "error" in result.output.lower()

    def test_undefined_stereo_exits_nonzero(self, tmp_path):
        result = run("build", "CC(N)C(=O)O", "-o", str(tmp_path / "a.pdb"))
        assert result.exit_code == 1
        assert "stereocenter" in result.output.lower()

    def test_unknown_backend_exits_nonzero(self, tmp_path):
        result = run(
            "build", QUINUCLIDINONE, "-o", str(tmp_path / "q.pdb"), "--backend", "nope"
        )
        assert result.exit_code == 1

    def test_invalid_stereo_mode_is_rejected(self, tmp_path):
        result = run(
            "build", QUINUCLIDINONE, "-o", str(tmp_path / "q.pdb"), "--stereo", "sideways"
        )
        assert result.exit_code == 1


class TestProtonationFlags:
    """--ph takes a value; --protonate is the no-argument shorthand.

    An option that optionally takes a value silently swallows the next argument,
    so `--ph -o out.pdb` would read '-o' as the pH. These two flags keep that
    from being possible.
    """

    @needs_dimorphite
    def test_protonate_is_shorthand_for_ph_7_4(self, tmp_path):
        from rdkit import Chem

        out = tmp_path / "g.pdb"
        assert run("build", GABAPENTIN, "-o", str(out), "--protonate", "-q").exit_code == 0

        built = next(iter(Chem.SDMolSupplier(str(out.with_suffix(".sdf")), removeHs=False)))
        charges = [a.GetFormalCharge() for a in built.GetAtoms()]
        assert any(c > 0 for c in charges) and any(c < 0 for c in charges)

    @needs_dimorphite
    def test_explicit_ph_is_honoured(self, tmp_path):
        from rdkit import Chem

        out = tmp_path / "g.pdb"
        assert run("build", GABAPENTIN, "-o", str(out), "--ph", "2.0", "-q").exit_code == 0

        built = next(iter(Chem.SDMolSupplier(str(out.with_suffix(".sdf")), removeHs=False)))
        assert Chem.GetFormalCharge(built) == 1

    def test_contradictory_ph_and_protonate_are_rejected(self, tmp_path):
        result = run(
            "build", GABAPENTIN, "-o", str(tmp_path / "g.pdb"), "--protonate", "--ph", "2.0"
        )
        assert result.exit_code == 1
        assert "--protonate" in result.output

    def test_enumerate_states_without_a_ph_is_rejected(self, tmp_path):
        result = run(
            "build", GABAPENTIN, "-o", str(tmp_path / "g.pdb"), "--enumerate-states"
        )
        assert result.exit_code == 1
        assert "needs a pH" in result.output

    def test_default_leaves_the_molecule_as_drawn(self, tmp_path):
        from rdkit import Chem

        out = tmp_path / "g.pdb"
        run("build", GABAPENTIN, "-o", str(out), "-q")
        built = next(iter(Chem.SDMolSupplier(str(out.with_suffix(".sdf")), removeHs=False)))
        assert all(a.GetFormalCharge() == 0 for a in built.GetAtoms())


class TestInformationalCommands:
    def test_backends_lists_the_registry(self):
        result = run("backends")
        assert result.exit_code == 0
        for name in ("mmff94", "gfn2", "mace-off", "aimnet2"):
            assert name in result.output

    def test_doctor_runs_and_reports_sections(self):
        result = run("doctor")
        assert result.exit_code == 0
        assert "backends" in result.output
        assert "model weights" in result.output

    def test_doctor_does_not_mangle_install_hints(self):
        """Rich treats [mlff] as a style tag; unescaped it vanishes from the hint."""
        result = run("doctor")
        if "pip install" in result.output:
            assert "ligand3d'" not in result.output.replace("ligand3d[mlff]'", "")

    def test_version(self):
        from ligand3d import __version__

        result = run("version")
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_config_show_without_a_file_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LIGAND3D_CONFIG", str(tmp_path / "nope.toml"))
        assert run("config", "--show").exit_code == 0
