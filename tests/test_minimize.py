"""Backends: capability gating, energy semantics, and that they actually work."""

from __future__ import annotations

import numpy as np
import pytest

from ligand3d.embed import embed
from ligand3d.errors import BackendMismatch, BackendUnavailable
from ligand3d.minimize import (
    MinimizeJob,
    check_compatible,
    get_backend,
    list_backends,
    parse_chain,
    resolve_name,
)
from ligand3d.minimize.base import ORGANIC_ELEMENTS
from ligand3d.molecule import from_smiles


def available(name: str) -> bool:
    try:
        return bool(get_backend(name).available())
    except Exception:
        return False


needs_xtb = pytest.mark.skipif(not available("gfn2"), reason="tblite not installed")


class TestRegistry:
    def test_builtin_backends_are_registered(self):
        registered = set(list_backends())
        assert {"mmff94", "uff", "gfn2", "gfn1", "gfnff", "aimnet2", "mace-off"} <= registered

    def test_aliases_resolve(self):
        assert resolve_name("mmff") == "mmff94"
        assert resolve_name("gfn2-xtb") == "gfn2"
        assert resolve_name("GFN2") == "gfn2"

    def test_unknown_backend_names_the_alternatives(self):
        with pytest.raises(BackendUnavailable, match="unknown backend"):
            get_backend("dft-please")

    def test_chain_parsing(self):
        assert parse_chain("mmff94,gfn2") == ["mmff94", "gfn2"]
        assert parse_chain(" mmff , gfn2 ") == ["mmff94", "gfn2"]

    def test_empty_chain_rejected(self):
        with pytest.raises(BackendUnavailable):
            parse_chain(" , ")


class TestCapabilityGating:
    """The checks that stop a backend giving a confident wrong answer."""

    def test_charged_molecule_refused_on_chargeless_potential(self):
        caps = get_backend("mace-off").caps
        with pytest.raises(BackendMismatch, match="no charge channel"):
            check_compatible(
                caps, charge=-1, elements=frozenset({6, 8}), solvent=None
            )

    def test_charge_mismatch_can_be_overridden(self):
        caps = get_backend("mace-off").caps
        check_compatible(
            caps,
            charge=-1,
            elements=frozenset({6, 8}),
            solvent=None,
            allow_charge_mismatch=True,
        )

    def test_charge_is_fine_on_a_charge_aware_backend(self):
        check_compatible(
            get_backend("gfn2").caps, charge=-1, elements=frozenset({6, 8}), solvent=None
        )

    def test_classical_force_field_accepts_charge(self):
        """MMFF reads formal charges off the atom typing, so it is not a mismatch."""
        check_compatible(
            get_backend("mmff94").caps, charge=-1, elements=frozenset({6, 8}), solvent=None
        )

    def test_zwitterion_refused_without_solvation(self):
        with pytest.raises(BackendMismatch, match="zwitterion"):
            check_compatible(
                get_backend("mace-off").caps,
                charge=0,
                elements=ORGANIC_ELEMENTS,
                solvent=None,
                is_zwitterion=True,
            )

    def test_zwitterion_allowed_on_fixed_topology_force_field(self):
        """MMFF cannot move a proton: its bond list is fixed."""
        check_compatible(
            get_backend("mmff94").caps,
            charge=0,
            elements=ORGANIC_ELEMENTS,
            solvent=None,
            is_zwitterion=True,
        )

    def test_unsupported_element_is_named(self):
        with pytest.raises(BackendMismatch, match="Ru"):
            check_compatible(
                get_backend("mace-off").caps,
                charge=0,
                elements=frozenset({6, 44}),  # carbon and ruthenium
                solvent=None,
            )

    def test_solvation_refused_where_unsupported(self):
        with pytest.raises(BackendMismatch, match="no implicit solvent"):
            check_compatible(
                get_backend("mace-off").caps,
                charge=0,
                elements=ORGANIC_ELEMENTS,
                solvent="water",
            )


class TestRDKitForceFields:
    @pytest.mark.parametrize("backend_name", ["mmff94", "uff"])
    def test_minimization_lowers_the_energy(self, backend_name):
        molecule = from_smiles("CCCCCCCCO")
        mol_3d = embed(molecule)
        backend = get_backend(backend_name)

        job = MinimizeJob(mol=mol_3d, conf_id=0)
        before = _energy_only(backend, job)
        result = backend.minimize(job)

        assert result.energy <= before + 1e-6
        assert result.backend == backend_name

    def test_geometry_actually_moves(self):
        mol_3d = embed(from_smiles("CCCCCCCCO"))
        start = np.array(mol_3d.GetConformer(0).GetPositions())
        get_backend("mmff94").minimize(MinimizeJob(mol=mol_3d, conf_id=0))
        end = np.array(mol_3d.GetConformer(0).GetPositions())
        assert not np.allclose(start, end)

    def test_force_fields_report_strain_energy(self):
        assert get_backend("mmff94").caps.energy_kind == "strain"

    def test_bond_lengths_stay_physical(self):
        mol_3d = embed(from_smiles("O=C1CN2CCC1CC2"))
        get_backend("mmff94").minimize(MinimizeJob(mol=mol_3d, conf_id=0))
        positions = np.array(mol_3d.GetConformer(0).GetPositions())
        for bond in mol_3d.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            length = float(np.linalg.norm(positions[i] - positions[j]))
            assert 0.7 < length < 2.2, f"bond {i}-{j} is {length:.2f} A"


def _energy_only(backend, job) -> float:
    """Energy of the current geometry, without letting the optimizer move it."""
    import copy

    probe = MinimizeJob(
        mol=copy.deepcopy(job.mol), conf_id=job.conf_id, charge=job.charge, max_steps=0
    )
    return backend.minimize(probe).energy


@needs_xtb
class TestSemiEmpirical:
    def test_gfn2_minimizes(self):
        mol_3d = embed(from_smiles("O=C1CN2CCC1CC2"))
        result = get_backend("gfn2").minimize(MinimizeJob(mol=mol_3d, conf_id=0))
        assert result.converged
        assert result.energy < 0  # total electronic energy
        assert result.energy_kind == "total"

    def test_gfn2_reports_total_energy_not_strain(self):
        """Absolute electronic energies are enormous and negative.

        Reporting them as if they were strain energies would invite comparison
        with an MMFF number, which means nothing.
        """
        mol_3d = embed(from_smiles("CCO"))
        result = get_backend("gfn2").minimize(MinimizeJob(mol=mol_3d, conf_id=0))
        assert result.energy < -1000

    def test_solvation_keeps_a_zwitterion_intact(self):
        """The finding this whole design is built around.

        In gas phase the ammonium proton hops back to the carboxylate and the
        molecule silently stops being a zwitterion. ALPB water prevents it.
        """
        molecule = from_smiles("[NH3+]CC1(CC(=O)[O-])CCCCC1")
        mol_3d = embed(molecule)
        get_backend("gfn2").minimize(
            MinimizeJob(mol=mol_3d, conf_id=0, charge=0, solvent="water")
        )
        assert _n_hydrogens_on_nitrogen(mol_3d) == 3

    def test_gas_phase_collapses_the_same_zwitterion(self):
        """Confirms the guard is protecting against something real, not theory."""
        mol_3d = embed(from_smiles("[NH3+]CC1(CC(=O)[O-])CCCCC1"))
        get_backend("gfn2").minimize(
            MinimizeJob(mol=mol_3d, conf_id=0, charge=0, solvent=None)
        )
        assert _n_hydrogens_on_nitrogen(mol_3d) < 3


def _n_hydrogens_on_nitrogen(mol_3d) -> int:
    positions = np.array(mol_3d.GetConformer(0).GetPositions())
    symbols = [a.GetSymbol() for a in mol_3d.GetAtoms()]
    nitrogen = symbols.index("N")
    return sum(
        1
        for i, s in enumerate(symbols)
        if s == "H" and np.linalg.norm(positions[i] - positions[nitrogen]) < 1.25
    )
