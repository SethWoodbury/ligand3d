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
        out = tmp_path / "q.cif"
        result = run("build", QUINUCLIDINONE, "-o", str(out))
        assert result.exit_code == 0, result.output
        assert out.exists()
        assert out.with_suffix(".sdf").exists()

    def test_quiet_prints_only_the_primary_path(self, tmp_path):
        """mmCIF is the default, so that is the path a script wants back."""
        out = tmp_path / "q.cif"
        result = run("build", QUINUCLIDINONE, "-o", str(out), "-q")
        assert result.exit_code == 0
        assert result.output.strip() == str(out)

    def test_no_sdf_suppresses_the_sidecar(self, tmp_path):
        out = tmp_path / "q.cif"
        run("build", QUINUCLIDINONE, "-o", str(out), "--no-sdf")
        assert out.exists()
        assert not out.with_suffix(".sdf").exists()

    def test_explicit_suffix_selects_that_format(self, tmp_path):
        """`-o thing.pdb` should produce a PDB even though mmCIF is the default."""
        result = run("build", QUINUCLIDINONE, "-o", str(tmp_path / "q.pdb"), "-q")
        assert result.exit_code == 0
        assert (tmp_path / "q.pdb").exists()

    def test_default_format_is_mmcif(self, tmp_path):
        result = run("build", QUINUCLIDINONE, "-o", str(tmp_path / "q"), "-q")
        assert result.exit_code == 0
        assert (tmp_path / "q.cif").exists()
        assert not (tmp_path / "q.pdb").exists()

    def test_format_flag_controls_what_is_written(self, tmp_path):
        run("build", QUINUCLIDINONE, "-o", str(tmp_path / "q"), "-f", "pdb,sdf", "-q")
        assert (tmp_path / "q.pdb").exists()
        assert (tmp_path / "q.sdf").exists()
        assert not (tmp_path / "q.cif").exists()

    def test_unknown_format_is_rejected(self, tmp_path):
        result = run("build", QUINUCLIDINONE, "-o", str(tmp_path / "q"), "-f", "xyz")
        assert result.exit_code == 1
        assert "unknown format" in result.output

    def test_bad_smiles_exits_nonzero(self, tmp_path):
        result = run("build", "not a molecule", "-o", str(tmp_path / "x.cif"))
        assert result.exit_code == 1
        assert "error" in result.output.lower()

    def test_undefined_stereo_exits_nonzero(self, tmp_path):
        result = run("build", "CC(N)C(=O)O", "-o", str(tmp_path / "a.cif"))
        assert result.exit_code == 1
        assert "stereocenter" in result.output.lower()

    def test_unknown_backend_exits_nonzero(self, tmp_path):
        result = run(
            "build", QUINUCLIDINONE, "-o", str(tmp_path / "q.cif"), "--backend", "nope"
        )
        assert result.exit_code == 1

    def test_invalid_stereo_mode_is_rejected(self, tmp_path):
        result = run(
            "build", QUINUCLIDINONE, "-o", str(tmp_path / "q.cif"), "--stereo", "sideways"
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

        out = tmp_path / "g.cif"
        assert run("build", GABAPENTIN, "-o", str(out), "--protonate", "-q").exit_code == 0

        built = next(iter(Chem.SDMolSupplier(str(out.with_suffix(".sdf")), removeHs=False)))
        charges = [a.GetFormalCharge() for a in built.GetAtoms()]
        assert any(c > 0 for c in charges) and any(c < 0 for c in charges)

    @needs_dimorphite
    def test_explicit_ph_is_honoured(self, tmp_path):
        from rdkit import Chem

        out = tmp_path / "g.cif"
        assert run("build", GABAPENTIN, "-o", str(out), "--ph", "2.0", "-q").exit_code == 0

        built = next(iter(Chem.SDMolSupplier(str(out.with_suffix(".sdf")), removeHs=False)))
        assert Chem.GetFormalCharge(built) == 1

    def test_contradictory_ph_and_protonate_are_rejected(self, tmp_path):
        result = run(
            "build", GABAPENTIN, "-o", str(tmp_path / "g.cif"), "--protonate", "--ph", "2.0"
        )
        assert result.exit_code == 1
        assert "--protonate" in result.output

    def test_enumerate_states_without_a_ph_is_rejected(self, tmp_path):
        result = run(
            "build", GABAPENTIN, "-o", str(tmp_path / "g.cif"), "--enumerate-states"
        )
        assert result.exit_code == 1
        assert "needs a pH" in result.output

    def test_default_leaves_the_molecule_as_drawn(self, tmp_path):
        from rdkit import Chem

        out = tmp_path / "g.cif"
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

    def test_doctor_does_not_mangle_bracketed_extras(self):
        """Rich reads `[mace]` as a style tag and eats it unless it is escaped.

        Any hint naming an extra must still show the brackets, otherwise doctor
        tells people to run `pip install 'ligand3d'` — the one command that will
        not fix their problem.
        """
        from ligand3d.minimize import all_backends

        hints = []
        for backend in all_backends():
            hint = getattr(backend, "install_hint", lambda: "")()
            if "ligand3d[" in hint:
                hints.append(hint)
        if not hints:
            pytest.skip("no backend advertises a bracketed extra")

        output = run("doctor").output
        # At least one bracketed extra should appear intact somewhere.
        extras = {h.split("ligand3d[", 1)[1].split("]", 1)[0] for h in hints}
        assert any(f"ligand3d[{name}]" in output for name in extras), (
            f"none of {sorted(extras)} survived Rich markup"
        )

    def test_doctor_lists_model_weights_and_unsupported_models(self):
        output = run("doctor").output
        assert "model weights" in output
        assert "known but not loadable" in output
        assert "mace-polar" in output

    def test_version(self):
        from ligand3d import __version__

        result = run("version")
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_config_show_without_a_file_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LIGAND3D_CONFIG", str(tmp_path / "nope.toml"))
        assert run("config", "--show").exit_code == 0


class TestEveryCommandIsReachableBothWays:
    """`python -m ligand3d.cli` executed app() partway down the file.

    Seven commands were defined below that point and never registered, so they
    were invisible to `-m` while working fine through the installed entry
    point. That split matters here specifically: the container and SLURM paths
    both invoke this module with `-m`, and `slurm-run` only worked because it
    happened to be defined above the block.
    """

    COMMANDS = (
        "build", "backends", "doctor", "config", "sketch", "models", "fetch",
        "solvents", "version", "slurm", "stereo", "embed", "minimize",
        "conformers", "protonate", "params", "convert",
    )

    def _run_module(self, *args):
        import os
        import subprocess
        import sys

        # A wide terminal, because rich wraps help text to fit and a narrow
        # CI terminal would split the command names this test looks for.
        return subprocess.run(
            [sys.executable, "-m", "ligand3d.cli", *args],
            capture_output=True, text=True, timeout=180,
            env={**os.environ, "COLUMNS": "200", "TERM": "dumb"},
        )

    @pytest.mark.parametrize("command", COMMANDS)
    def test_reachable_through_the_app_object(self, command):
        result = run(command, "--help")
        assert result.exit_code == 0, result.output

    @pytest.mark.parametrize("command", COMMANDS)
    def test_reachable_through_python_m(self, command):
        result = self._run_module(command, "--help")
        assert result.returncode == 0, result.stdout + result.stderr

    def test_the_two_entry_points_agree(self):
        """Any future drift shows up here rather than in a container."""
        listed = self._run_module("--help")
        assert listed.returncode == 0
        for command in self.COMMANDS:
            assert command in listed.stdout, f"{command} missing from -m help"

    def test_the_main_guard_is_the_last_thing_in_the_file(self):
        """Anything after it is dead to `-m`, silently."""
        from pathlib import Path

        source = Path(__file__).resolve().parents[1] / "src" / "ligand3d" / "cli.py"
        lines = [line for line in source.read_text().splitlines() if line.strip()]
        guard = max(
            i for i, line in enumerate(lines) if line.startswith('if __name__ ==')
        )
        after = [line for line in lines[guard + 1:] if not line.startswith((" ", "\t", "#"))]
        assert after == [], f"defined after the main guard, invisible to -m: {after}"
