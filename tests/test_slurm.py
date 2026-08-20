"""The SLURM submission path.

Nothing here talks to a scheduler. What is worth testing is the reasoning
around the submission — which container a backend needs, which requests would
waste an allocation, and whether a payload survives the trip to a compute node
— because those are the parts that fail expensively and silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ligand3d.molecule import from_smiles
from ligand3d.pipeline import Settings
from ligand3d.slurm import (
    QUANTUM_CHEM_SIF,
    UMA_SIF,
    SlurmConfig,
    SlurmError,
    _walltime_minutes,
    build_payload,
    container_for,
    needs_gpu,
    render_script,
    submit,
)


class TestWalltime:
    @pytest.mark.parametrize(
        "text, minutes",
        [
            ("00:05:00", 5),
            ("01:00:00", 60),
            ("30", 30),
            ("2:30", 2.5),
            ("1-00:00:00", 1440),
            ("2-12:00:00", 3600),
        ],
    )
    def test_parses_every_shape_slurm_accepts(self, text, minutes):
        assert _walltime_minutes(text) == pytest.approx(minutes)

    def test_rejects_a_job_shorter_than_the_scheduler_allows(self):
        # Not a guess: sbatch refuses 00:02:00 with "The job time limit is too
        # short; run longer jobs!". Catching it here saves a round trip.
        problems = SlurmConfig(walltime="00:02:00").check()
        assert any("minimum" in p for p in problems)

    def test_accepts_five_minutes(self):
        assert SlurmConfig(walltime="00:05:00").check() == []


class TestGres:
    def test_gpu_partition_asks_for_a_gpu(self):
        assert SlurmConfig(partition="gpu", gpu_class="large", gpus=2).gres() == "gpu:large:2"

    def test_cpu_partition_asks_for_none(self):
        assert SlurmConfig(partition="cpu").gres() is None

    def test_rejects_a_gpu_class_that_does_not_schedule(self):
        # The a4000/b4000 names were retired; a job asking for one never starts.
        problems = SlurmConfig(gpu_class="a4000").check()
        assert any("small, large, h200" in p for p in problems)


class TestContainerChoice:
    """The mace/fairchem e3nn split exists inside the containers too."""

    def test_mace_gets_the_quantum_chem_image(self):
        assert container_for("mace-off") == QUANTUM_CHEM_SIF

    def test_fairchem_gets_the_uma_image(self):
        assert container_for("esen") == UMA_SIF

    def test_polar_rides_with_mace(self):
        assert container_for("mace-polar") == QUANTUM_CHEM_SIF

    def test_a_classical_chain_defaults_to_quantum_chem(self):
        assert container_for("mmff94") == QUANTUM_CHEM_SIF

    def test_a_mixed_chain_is_refused_here_not_on_the_node(self):
        with pytest.raises(SlurmError, match="incompatible e3nn"):
            container_for("mace-off,esen")

    def test_a_classical_prefix_does_not_confuse_the_choice(self):
        assert container_for("mmff94,esen") == UMA_SIF


class TestNeedsGpu:
    def test_neural_potentials_want_one(self):
        assert needs_gpu("mace-off") is True

    def test_a_classical_force_field_does_not(self):
        assert needs_gpu("mmff94") is False

    def test_a_chain_ending_in_a_potential_does(self):
        assert needs_gpu("mmff94,mace-off") is True

    def test_an_unknown_name_does_not_raise(self):
        # Validation belongs to the pipeline; this only decides whether to warn.
        assert needs_gpu("not-a-backend") is False


class TestPayload:
    def test_round_trips_the_settings(self):
        settings = Settings(backend="mace-off", n_confs=4, trace=True, formats=("cif", "pdb"))
        payload = build_payload(from_smiles("CCO"), settings, Path("/net/x/out.cif"))
        restored = Settings(**{**payload["settings"], "formats": tuple(payload["settings"]["formats"])})
        assert restored == settings

    def test_is_json_serializable(self):
        payload = build_payload(from_smiles("CCO"), Settings(), Path("/net/x/out.cif"))
        assert json.loads(json.dumps(payload))["output"] == "/net/x/out.cif"

    def test_carries_stereochemistry(self):
        # A molblock rather than a SMILES round trip, so the node builds the
        # isomer that was submitted.
        molecule = from_smiles("C[C@H](N)C(=O)O")
        payload = build_payload(molecule, Settings(), Path("/net/x/out.cif"))
        from ligand3d.molecule import from_molblock

        assert from_molblock(payload["molblock"]).smiles == molecule.smiles

    def test_carries_the_name_so_the_residue_is_not_renamed(self):
        payload = build_payload(from_smiles("CCO", name="ETH"), Settings(), Path("/net/x/o.cif"))
        assert payload["name"] == "ETH"


class TestScript:
    def _script(self, **kwargs):
        config = SlurmConfig(**kwargs)
        return render_script(
            Path("/net/scratch/job"), config, Path("/net/img.sif"),
            Path("/home/u/ligand3d/src"), Path("/net/scratch/job/job.json"),
        )

    def test_binds_the_source_tree_rather_than_installing_it(self):
        script = self._script()
        assert "--env PYTHONPATH=/home/u/ligand3d/src" in script
        assert "pip install" not in script

    def test_asks_for_the_gpu_through_apptainer(self):
        assert "apptainer exec --nv" in self._script(partition="gpu")

    def test_a_cpu_job_does_not_ask_for_a_gpu(self):
        script = self._script(partition="cpu", gpus=0)
        assert "--nv" not in script
        assert "--gres" not in script

    def test_logs_land_on_shared_storage(self):
        script = self._script()
        assert "#SBATCH -o /net/scratch/job/job.out" in script

    def test_stops_at_the_first_failure(self):
        assert "set -euo pipefail" in self._script()

    def test_threads_follow_the_allocation(self):
        assert 'OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"' in self._script()

    def test_carries_the_account(self):
        assert "#SBATCH -A IPD" in self._script(account="IPD")


class TestSubmitGuards:
    """Refusals that happen before anything is queued."""

    def test_refuses_node_local_storage(self, tmp_path):
        # A compute node has its own /tmp. A job whose results go there exits 0
        # and leaves nothing behind, which is the worst possible failure mode.
        payload = build_payload(from_smiles("CCO"), Settings(), Path("/tmp/out.cif"))
        with pytest.raises(SlurmError, match="shared storage"):
            submit(payload, Path("/tmp/l3d-should-not-run"), dry_run=True)

    def test_refuses_a_walltime_the_scheduler_would_reject(self, tmp_path):
        payload = build_payload(from_smiles("CCO"), Settings(), tmp_path / "out.cif")
        with pytest.raises(SlurmError, match="minimum"):
            submit(payload, tmp_path, SlurmConfig(walltime="00:01:00"), dry_run=True)

    def test_dry_run_writes_the_script_without_queueing_it(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ligand3d.slurm._is_shared", lambda p: True)
        payload = build_payload(from_smiles("CCO"), Settings(backend="mace-off"), tmp_path / "o.cif")
        job = submit(payload, tmp_path / "work", dry_run=True)

        assert job.job_id == 0
        assert job.script.exists()
        assert json.loads((tmp_path / "work" / "job.json").read_text())["settings"]["backend"] == "mace-off"

    def test_snapshots_the_source_so_later_edits_cannot_change_the_job(
        self, tmp_path, monkeypatch
    ):
        # A queued job may not start for hours. If it imported the working tree
        # directly, editing the repo meanwhile would silently change what runs.
        monkeypatch.setattr("ligand3d.slurm._is_shared", lambda p: True)
        payload = build_payload(from_smiles("CCO"), Settings(), tmp_path / "o.cif")
        job = submit(payload, tmp_path / "work", dry_run=True)

        snapshot = tmp_path / "work" / "src" / "ligand3d"
        script = job.script.read_text()
        assert (snapshot / "pipeline.py").exists()
        assert f"--env PYTHONPATH={tmp_path / 'work' / 'src'}" in script
        assert not (snapshot / "__pycache__").exists()
        # The job would otherwise compile the snapshot as it imports it.
        assert "PYTHONDONTWRITEBYTECODE=1" in script

    def test_a_mixed_chain_is_refused_before_submission(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ligand3d.slurm._is_shared", lambda p: True)
        payload = build_payload(
            from_smiles("CCO"), Settings(backend="mace-off,esen"), tmp_path / "o.cif"
        )
        with pytest.raises(SlurmError, match="incompatible e3nn"):
            submit(payload, tmp_path / "work", dry_run=True)


class TestNothingElseImportsIt:
    def test_the_feature_is_optional(self):
        """ligand3d must work with no scheduler anywhere in sight.

        The module is imported lazily from exactly one place in the CLI, so a
        machine without SLURM never loads it.
        """
        import subprocess
        import sys

        source = Path(__file__).resolve().parents[1] / "src" / "ligand3d"
        importers = subprocess.run(
            ["grep", "-rln", r"import slurm\|from .slurm\|from ligand3d.slurm", str(source)],
            capture_output=True, text=True,
        ).stdout.split()
        assert [Path(p).name for p in importers] == ["cli.py"]

        # And importing the package does not drag it in.
        check = subprocess.run(
            [sys.executable, "-c", "import ligand3d, sys; sys.exit('ligand3d.slurm' in sys.modules)"],
            capture_output=True, text=True,
        )
        assert check.returncode == 0, "importing ligand3d should not import the slurm module"
