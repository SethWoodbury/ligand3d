"""Turning a typed query into a structure.

The network is not touched unless a test says so. PubChem is a third party and
a test suite that fails when it is slow is a test suite people stop running, so
the online tests are marked and the routing logic is exercised with stubs.
"""

from __future__ import annotations

import pytest

from ligand3d.resolve import (
    TEMPLATES,
    ResolveError,
    Resolved,
    from_inchi_text,
    from_opsin,
    from_smiles_text,
    looks_like_a_name,
    opsin_available,
    resolve,
    template,
    template_list,
)

needs_opsin = pytest.mark.skipif(not opsin_available(), reason="py2opsin or java missing")
needs_network = pytest.mark.network


class TestLooksLikeAName:
    """Deciding which route to try first, not which one is correct."""

    @pytest.mark.parametrize(
        "text", ["aspirin", "3-Cyano-7-ethoxycoumarin", "acetic acid", "quinuclidin-3-one"]
    )
    def test_names(self, text):
        assert looks_like_a_name(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "CCO",
            "c1ccccc1",          # a five-letter lowercase run that is not a word
            "O=C1CN2CCC1CC2",
            "N[C@@H](C)C(=O)O",
            "c1ccc2[nH]ccc2c1",  # bracket atoms use letters SMILES otherwise never does
            "CCOc1ccc2cc(C#N)c(=O)oc2c1",
            "ClCCBr",            # l and r appear only as the tails of Cl and Br
        ],
    )
    def test_smiles(self, text):
        assert looks_like_a_name(text) is False


class TestParsing:
    def test_reads_smiles(self):
        found = from_smiles_text("CCO")
        assert found.source == "parsed-smiles"
        assert found.smiles == "CCO"
        assert found.formula == "C2H6O"

    def test_canonicalizes(self):
        assert from_smiles_text("C(C)O").smiles == "CCO"

    def test_keeps_stereochemistry(self):
        assert "@" in from_smiles_text("N[C@@H](C)C(=O)O").smiles

    def test_a_name_is_not_a_smiles(self):
        assert from_smiles_text("aspirin") is None

    def test_reads_inchi(self):
        found = from_inchi_text("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3")
        assert found.source == "parsed-inchi"
        assert found.smiles == "CCO"

    def test_ignores_what_is_not_an_inchi(self):
        assert from_inchi_text("CCO") is None

    def test_a_broken_inchi_is_an_error_not_a_pass(self):
        # It announced itself as an InChI, so failing to read it is worth saying.
        with pytest.raises(ResolveError):
            from_inchi_text("InChI=1S/this-is-not-real")


@needs_opsin
class TestOpsin:
    """Systematic names, derived offline from the naming rules."""

    def test_parses_the_motivating_example(self):
        from rdkit import Chem

        found = from_opsin("3-Cyano-7-ethoxycoumarin")
        assert found.source == "opsin"
        assert found.formula == "C12H9NO3"
        # The same molecule PubChem returns for that name.
        assert found.smiles == Chem.MolToSmiles(
            Chem.MolFromSmiles("CCOC1=CC2=C(C=C1)C=C(C(=O)O2)C#N")
        )

    @pytest.mark.parametrize(
        "name, formula",
        [
            ("quinuclidin-3-one", "C7H11NO"),
            ("2-amino-2-methylpropan-1-ol", "C4H11NO"),
            ("benzene", "C6H6"),
        ],
    )
    def test_parses_systematic_names(self, name, formula):
        assert from_opsin(name).formula == formula

    def test_returns_none_for_a_trivial_name(self):
        # Not an error: OPSIN is a grammar, not a database, and "aspirin" is
        # not derivable from the naming rules. PubChem is what covers this.
        assert from_opsin("aspirin") is None

    def test_returns_none_rather_than_raising_on_nonsense(self):
        assert from_opsin("notarealcompoundxyz") is None


class TestRouting:
    """Which route answers, without going near the network."""

    def test_smiles_wins_for_something_that_looks_like_smiles(self):
        assert resolve("CCO", allow_network=False).source == "parsed-smiles"

    def test_inchi_is_recognised_first(self):
        found = resolve("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3", allow_network=False)
        assert found.source == "parsed-inchi"

    def test_a_prefix_forces_the_route(self):
        assert resolve("smiles:CO", allow_network=False).source == "parsed-smiles"

    def test_a_bad_forced_smiles_is_refused_without_trying_anything_else(self):
        with pytest.raises(ResolveError, match="not a valid SMILES"):
            resolve("smiles:aspirin", allow_network=False)

    def test_offline_says_so_rather_than_failing_silently(self):
        with pytest.raises(ResolveError, match="network lookups are off"):
            resolve("aspirin", allow_network=False)

    def test_a_word_that_is_also_valid_smiles_still_resolves(self):
        # "Nc1ccccc1" is unambiguous, but a short lowercase run can read as a
        # word. Whatever the ordering guess was, a parseable structure must not
        # be lost.
        found = resolve("CCCC", allow_network=False)
        assert found.smiles == "CCCC"

    def test_empty_is_refused(self):
        with pytest.raises(ResolveError, match="nothing to look up"):
            resolve("   ", allow_network=False)


class TestPubChemStubbed:
    """The PubChem branch, without depending on PubChem being up."""

    def _payload(self, **overrides):
        record = {
            "CID": 2244, "MolecularFormula": "C9H8O4",
            "SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O", "Title": "Aspirin",
        }
        record.update(overrides)
        return {"PropertyTable": {"Properties": [record]}}

    def test_reads_a_record(self, monkeypatch):
        from ligand3d import resolve as mod

        monkeypatch.setattr(mod, "_pubchem_get", lambda p, t: self._payload())
        found = mod.from_pubchem("aspirin")
        assert found.source == "pubchem"
        assert found.cid == 2244
        assert found.name == "Aspirin"
        assert found.url.endswith("/2244")

    def test_accepts_the_older_property_names(self, monkeypatch):
        # PubChem renamed CanonicalSMILES to ConnectivitySMILES and
        # IsomericSMILES to SMILES; both vintages must still be readable.
        from ligand3d import resolve as mod

        payload = self._payload()
        del payload["PropertyTable"]["Properties"][0]["SMILES"]
        payload["PropertyTable"]["Properties"][0]["CanonicalSMILES"] = "CCO"
        monkeypatch.setattr(mod, "_pubchem_get", lambda p, t: payload)
        assert mod.from_pubchem("whatever").smiles == "CCO"

    def test_no_record_is_none_not_an_error(self, monkeypatch):
        from ligand3d import resolve as mod

        monkeypatch.setattr(mod, "_pubchem_get", lambda p, t: None)
        assert mod.from_pubchem("nothing") is None

    def test_a_cid_query_hits_the_cid_endpoint(self, monkeypatch):
        from ligand3d import resolve as mod

        seen = {}

        def spy(path, timeout):
            seen["path"] = path
            return self._payload()

        monkeypatch.setattr(mod, "_pubchem_get", spy)
        mod.from_pubchem("cid:2244")
        assert seen["path"].startswith("compound/cid/2244")

    def test_a_name_is_url_encoded(self, monkeypatch):
        from ligand3d import resolve as mod

        seen = {}

        def spy(path, timeout):
            seen["path"] = path
            return self._payload()

        monkeypatch.setattr(mod, "_pubchem_get", spy)
        mod.from_pubchem("N,N-dimethyl aniline")
        # Only the name segment; the property list after it is full of commas.
        name_segment = seen["path"].split("/")[2]
        assert " " not in name_segment and "," not in name_segment
        assert "%2C" in name_segment and "%20" in name_segment

    def test_an_unreachable_network_is_distinguished_from_a_missing_compound(
        self, monkeypatch
    ):
        from ligand3d import resolve as mod

        def dead(path, timeout):
            raise ResolveError("could not reach PubChem")

        monkeypatch.setattr(mod, "_pubchem_get", dead)
        with pytest.raises(ResolveError, match="could not reach"):
            mod.from_pubchem("aspirin")


class TestTemplates:
    def test_every_scaffold_parses(self):
        from rdkit import Chem

        for name, smiles, note in TEMPLATES:
            assert Chem.MolFromSmiles(smiles) is not None, f"{name} has a bad SMILES"
            assert note, f"{name} has no description"

    def test_names_are_unique(self):
        names = [name for name, _, _ in TEMPLATES]
        assert len(names) == len(set(names))

    def test_fetches_one_by_name(self):
        found = template("coumarin")
        assert found.source == "template"
        assert found.formula == "C9H6O2"

    def test_is_case_insensitive(self):
        assert template("Benzene").smiles == template("benzene").smiles

    def test_an_unknown_scaffold_lists_the_real_ones(self):
        with pytest.raises(ResolveError, match="Available:"):
            template("unobtainium")

    def test_the_menu_matches_the_table(self):
        assert len(template_list()) == len(TEMPLATES)


class TestResolvedOutput:
    def test_makes_a_2d_molblock_a_sketcher_can_load(self):
        from rdkit import Chem

        found = Resolved(smiles="c1ccccc1", source="template", query="benzene")
        block = found.molblock()
        back = Chem.MolFromMolBlock(block)
        assert back is not None
        assert Chem.MolToSmiles(back) == "c1ccccc1"

    def test_the_layout_is_flat(self):
        # It is a drawing to edit, not a conformer. Writing it with 3D-looking
        # coordinates would invite someone to treat a layout as a geometry.
        from rdkit import Chem

        block = Resolved(smiles="C1CCCCC1", source="template", query="x").molblock()
        conf = Chem.MolFromMolBlock(block).GetConformer()
        assert all(abs(conf.GetAtomPosition(i).z) < 1e-6 for i in range(6))

    def test_json_carries_what_the_page_needs(self):
        payload = Resolved(
            smiles="CCO", source="pubchem", query="ethanol", cid=702,
        ).to_json()
        assert payload["molblock"].splitlines()
        assert payload["provenance"] == "looked up in PubChem (CID 702)"

    def test_provenance_names_every_route(self):
        for source in ("parsed-smiles", "parsed-inchi", "opsin", "pubchem", "template"):
            found = Resolved(smiles="CCO", source=source, query="x")
            assert found.provenance and found.provenance != source


@needs_network
class TestAgainstPubChemForReal:
    """Marked so a flaky third party cannot break an ordinary test run.

    Run with `pytest -m network`.
    """

    def test_finds_a_trivial_name(self):
        found = resolve("aspirin")
        assert found.source == "pubchem"
        assert found.formula == "C9H8O4"

    def test_opsin_and_pubchem_agree_on_the_motivating_example(self):
        by_name = resolve("3-Cyano-7-ethoxycoumarin")
        from ligand3d.resolve import from_pubchem

        by_lookup = from_pubchem("3-Cyano-7-ethoxycoumarin")
        assert by_name.smiles == by_lookup.smiles
