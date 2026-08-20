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
        with pytest.raises(SlurmError, match="not on storage the job can reach"):
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


class TestUntrustedValues:
    """These fields reach a shell script, and one is a web text box.

    A newline ends the `#SBATCH` comment it sits in and leaves whatever
    follows in the script as a command, so the shapes are checked rather than
    trusted.
    """

    @pytest.mark.parametrize(
        "field, value",
        [
            ("walltime", "01:00:00\nrm -rf /"),
            ("walltime", "; touch /tmp/pwned"),
            ("walltime", "not-a-time"),
            ("partition", "gpu\n#SBATCH --uid=0"),
            ("memory", "16G$(whoami)"),
            ("gpu_class", "small`id`"),
            ("job_name", "a b\nevil"),
            ("account", "IPD;evil"),
        ],
    )
    def test_refused_before_a_script_is_written(self, field, value):
        assert SlurmConfig(**{field: value}).check(), f"{field}={value!r} was accepted"

    def test_a_bad_walltime_is_a_refusal_not_a_crash(self, tmp_path, monkeypatch):
        # _walltime_minutes would raise ValueError on this; the caller should
        # see the same clean error it gets for every other bad request.
        monkeypatch.setattr("ligand3d.slurm._is_shared", lambda p: True)
        payload = build_payload(from_smiles("CCO"), Settings(), tmp_path / "o.cif")
        with pytest.raises(SlurmError, match="not a valid SLURM value"):
            submit(payload, tmp_path / "w", SlurmConfig(walltime="soon"), dry_run=True)

    def test_ordinary_values_still_pass(self):
        assert SlurmConfig(
            partition="gpu-bf", gpu_class="h200", memory="32G", walltime="2-00:00:00",
            account="IPD", job_name="l3d-gabapentin",
        ).check() == []


class TestJobName:
    def test_strips_what_slurm_would_not_accept(self):
        from ligand3d.slurm import job_name_for

        assert job_name_for("my ligand\n#SBATCH -x") == "l3d-myligandsbatch-x"

    def test_survives_a_name_with_nothing_usable_in_it(self):
        from ligand3d.slurm import job_name_for

        assert job_name_for("////") == "l3d"

    def test_result_always_passes_validation(self):
        from ligand3d.slurm import job_name_for

        for name in ("LIG", "my ligand", "////", "ünïcödé", "a" * 80):
            assert SlurmConfig(job_name=job_name_for(name)).check() == []


class TestWaiting:
    """A scheduler too busy to answer is not a job that died."""

    def _no_sleeping(self, monkeypatch):
        monkeypatch.setattr("ligand3d.slurm.time.sleep", lambda _s: None)

    def test_returns_the_final_state(self, monkeypatch):
        from ligand3d import slurm

        self._no_sleeping(monkeypatch)
        states = iter(["PENDING", "RUNNING", "COMPLETED"])
        monkeypatch.setattr(slurm, "job_state", lambda _id: next(states))
        assert slurm.wait_for(1) == "COMPLETED"

    def test_rides_out_a_transient_query_failure(self, monkeypatch):
        from ligand3d import slurm

        self._no_sleeping(monkeypatch)
        answers = ["RUNNING", SlurmError("squeue timed out"), "RUNNING", "COMPLETED"]
        calls = iter(answers)

        def flaky(_id):
            value = next(calls)
            if isinstance(value, Exception):
                raise value
            return value

        monkeypatch.setattr(slurm, "job_state", flaky)
        assert slurm.wait_for(1) == "COMPLETED"

    def test_gives_up_when_the_scheduler_never_answers(self, monkeypatch):
        from ligand3d import slurm

        self._no_sleeping(monkeypatch)

        def always_fails(_id):
            raise SlurmError("scheduler is gone")

        monkeypatch.setattr(slurm, "job_state", always_fails)
        with pytest.raises(SlurmError, match="scheduler is gone"):
            slurm.wait_for(1)

    def test_honours_a_timeout(self, monkeypatch):
        from ligand3d import slurm

        self._no_sleeping(monkeypatch)
        clock = iter([0.0, 10.0, 9999.0])
        monkeypatch.setattr(slurm.time, "monotonic", lambda: next(clock))
        monkeypatch.setattr(slurm, "job_state", lambda _id: "PENDING")
        with pytest.raises(SlurmError, match="still PENDING"):
            slurm.wait_for(1, timeout=60)


class TestSnapshotNeverDeletesSomebodysWork:
    """`--slurm-dir` takes an arbitrary directory, and the snapshot lives at
    `<workdir>/src`. Blindly clearing that would recursively delete the source
    tree of any project whose directory was passed by mistake."""

    def test_refuses_to_clear_a_src_that_is_not_a_snapshot(self, tmp_path):
        from ligand3d.slurm import _snapshot_source

        workdir = tmp_path / "someones-project"
        precious = workdir / "src" / "important.py"
        precious.parent.mkdir(parents=True)
        precious.write_text("# a year of work\n")

        with pytest.raises(SlurmError, match="Refusing to delete"):
            _snapshot_source(Path(__file__).resolve().parents[1] / "src", workdir)
        assert precious.read_text() == "# a year of work\n"

    def test_replaces_a_previous_snapshot(self, tmp_path):
        from ligand3d.slurm import _snapshot_source

        root = Path(__file__).resolve().parents[1] / "src"
        workdir = tmp_path / "job"
        _snapshot_source(root, workdir)
        stale = workdir / "src" / "ligand3d" / "gone-in-the-next-one.py"
        stale.write_text("x")

        _snapshot_source(root, workdir)
        assert not stale.exists()
        assert (workdir / "src" / "ligand3d" / "pipeline.py").exists()

    def test_keeps_what_the_job_imports(self, tmp_path):
        # slurm-run imports the result serializer from sketch.session, so the
        # subpackage has to travel even though the job serves no web page.
        from ligand3d.slurm import _snapshot_source

        _snapshot_source(Path(__file__).resolve().parents[1] / "src", tmp_path / "j")
        package = tmp_path / "j" / "src" / "ligand3d"
        assert (package / "sketch" / "session.py").exists()
        assert not (package / "sketch" / "static").exists()


class TestPathsThatWouldBreakTheScript:
    """SLURM reads `#SBATCH` paths literally — no shell, so no quoting."""

    @pytest.mark.parametrize(
        "bad", ["/home/me/my project", "/home/me/a;touch PWNED", "/home/me/x$(id)"]
    )
    def test_refused(self, bad, monkeypatch):
        monkeypatch.setattr("ligand3d.slurm._is_shared", lambda p: True)
        payload = build_payload(from_smiles("CCO"), Settings(), Path(bad) / "o.cif")
        with pytest.raises(SlurmError, match="shell character"):
            submit(payload, Path(bad) / "job", dry_run=True)

    def test_the_shell_body_is_quoted_anyway(self):
        # Defence in depth: the directives are screened, but anything the shell
        # actually executes is quoted so a surprising path cannot split a command.
        script = render_script(
            Path("/net/j"), SlurmConfig(), Path("/net/an image.sif"),
            Path("/net/j/src"), Path("/net/j/job.json"),
        )
        assert "'/net/an image.sif'" in script


class TestSharedStorage:
    def test_matches_path_components_not_string_prefixes(self):
        from ligand3d.slurm import _is_shared

        assert _is_shared(Path("/home/woodbuse/x")) is True
        assert _is_shared(Path("/homeless/x")) is False
        assert _is_shared(Path("/net2/x")) is False

    def test_only_accepts_what_the_job_actually_mounts(self):
        # /projects is shared between hosts but is not bind-mounted, so inside
        # the container it does not exist and PYTHONPATH points at nothing.
        from ligand3d.slurm import STANDARD_BINDS, _is_shared

        assert _is_shared(Path("/projects/me/runs")) is False
        assert set(STANDARD_BINDS) == {"/home", "/net", "/mnt"}


class TestUnaccountedJob:
    """squeue forgets a finished job; without accounting, sacct never knew it."""

    def _no_sleeping(self, monkeypatch):
        monkeypatch.setattr("ligand3d.slurm.time.sleep", lambda _s: None)

    def test_stops_instead_of_watching_forever(self, monkeypatch):
        from ligand3d import slurm

        self._no_sleeping(monkeypatch)
        monkeypatch.setattr(slurm, "job_state", lambda _id: "UNKNOWN")
        assert slurm.wait_for(1) == "UNKNOWN"

    def test_a_flicker_of_unknown_does_not_end_the_wait(self, monkeypatch):
        from ligand3d import slurm

        self._no_sleeping(monkeypatch)
        states = iter(["RUNNING", "UNKNOWN", "RUNNING", "COMPLETED"])
        monkeypatch.setattr(slurm, "job_state", lambda _id: next(states))
        assert slurm.wait_for(1) == "COMPLETED"

    def test_a_failed_query_is_not_treated_as_an_unknown_job(self, monkeypatch):
        # Being unable to ask is not the same as the job having no record; the
        # first is worth retrying twenty times, the second means it is over.
        from ligand3d import slurm

        self._no_sleeping(monkeypatch)
        calls = {"n": 0}

        def flaky(_id):
            calls["n"] += 1
            if calls["n"] <= 10:
                raise SlurmError("squeue timed out")
            return "COMPLETED"

        monkeypatch.setattr(slurm, "job_state", flaky)
        assert slurm.wait_for(1) == "COMPLETED"


class TestFailedSubmissionLeavesNothingBehind:
    def test_the_workdir_is_removed_when_sbatch_refuses(self, tmp_path, monkeypatch):
        import subprocess as sp

        from ligand3d import slurm

        monkeypatch.setattr(slurm, "_is_shared", lambda p: True)
        monkeypatch.setattr(slurm, "slurm_available", lambda: True)
        monkeypatch.setattr(slurm, "apptainer_available", lambda: True)
        monkeypatch.setattr(
            slurm.subprocess, "run",
            lambda *a, **k: sp.CompletedProcess(a[0], 1, "", "Invalid partition"),
        )
        payload = build_payload(from_smiles("CCO"), Settings(), tmp_path / "o.cif")
        workdir = tmp_path / "job"

        with pytest.raises(SlurmError, match="Invalid partition"):
            submit(payload, workdir, SlurmConfig(container=Path(__file__)))
        assert not workdir.exists()

    def test_a_pre_existing_workdir_is_left_alone(self, tmp_path, monkeypatch):
        import subprocess as sp

        from ligand3d import slurm

        monkeypatch.setattr(slurm, "_is_shared", lambda p: True)
        monkeypatch.setattr(slurm, "slurm_available", lambda: True)
        monkeypatch.setattr(slurm, "apptainer_available", lambda: True)
        monkeypatch.setattr(
            slurm.subprocess, "run",
            lambda *a, **k: sp.CompletedProcess(a[0], 1, "", "nope"),
        )
        workdir = tmp_path / "job"
        workdir.mkdir()
        (workdir / "notes.txt").write_text("mine")
        payload = build_payload(from_smiles("CCO"), Settings(), tmp_path / "o.cif")

        with pytest.raises(SlurmError):
            submit(payload, workdir, SlurmConfig(container=Path(__file__)))
        assert (workdir / "notes.txt").read_text() == "mine"
