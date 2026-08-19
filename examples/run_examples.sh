#!/usr/bin/env bash
# Worked examples. Run from anywhere; output lands in ./ligand3d-examples/.
set -euo pipefail

OUT="${1:-ligand3d-examples}"
mkdir -p "$OUT"
cd "$OUT"

echo "== 1. 3-Quinuclidinone, default MMFF94 (milliseconds) =="
ligand3d build "O=C1CN2CCC1CC2" -o quinuclidinone.pdb --resname QUI

echo
echo "== 2. Gabapentin, MMFF94 pre-minimization then GFN2-xTB refinement =="
ligand3d build "NCC1(CC(=O)O)CCCCC1" -o gabapentin.pdb --backend mmff94,gfn2 --resname GAB

echo
echo "== 3. Gabapentin at pH 7.4 =="
# Becomes the zwitterion, so implicit solvation switches on by itself. Without it
# the ammonium proton hops back to the carboxylate and you get a different molecule.
ligand3d build "NCC1(CC(=O)O)CCCCC1" --ph 7.4 -o gabapentin_ph74.pdb \
    --backend mmff94,gfn2 --resname GAB

echo
echo "== 4. Every protonation state at pH 7.4, one file each =="
ligand3d build "NCC1(CC(=O)O)CCCCC1" --ph 7.4 --enumerate-states \
    -o gabapentin_states.pdb --resname GAB

echo
echo "== 5. Conformer search: 20 conformers, ranked, 5 kcal/mol window =="
ligand3d build "NCC1(CC(=O)O)CCCCC1" --confs 20 --energy-window 5.0 \
    -o gabapentin_confs.pdb --resname GAB

echo
echo "== 6. A molecule with real stereochemistry (threonine, 2 centers) =="
ligand3d build "C[C@@H](O)[C@H](N)C(=O)O" -o threonine.pdb --resname THR

echo
echo "== 7. Undefined stereochemistry is refused rather than guessed =="
ligand3d build "CC(N)C(=O)O" -o alanine.pdb || echo "   (refused, as intended)"

echo
echo "== 8. ...unless you ask for every isomer =="
ligand3d build "CC(N)C(=O)O" --stereo enumerate -o alanine.pdb --resname ALA

echo
echo "== 9. Machine-learned potential, if one is installed =="
ligand3d build "O=C1CN2CCC1CC2" --backend mmff94,mace-off -o quinuclidinone_mace.pdb \
    --resname QUI || echo "   (mace-off unavailable here; run 'ligand3d doctor')"

echo
echo "Wrote:"
ls -1 ./*.pdb ./*.sdf
