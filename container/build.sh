#!/usr/bin/env bash
#
# Build a release of ligand3d as a container, and stage it for the lab.
#
#     container/build.sh                      # build into ./dist
#     container/build.sh /net/software/containers/users/woodbuse/ligand3d
#
# The result is a directory holding the image, the launcher, and a note saying
# what version it is. Point PATH at it and `ligand3d` works with nothing
# installed.
#
# The source is baked into the image, so this has to be re-run for a code
# change to reach anyone. That is the trade that was chosen: a labmate's run
# cannot be altered by work in progress.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/.." && pwd)"
DEST="${1:-$ROOT/dist}"

command -v apptainer >/dev/null || { echo "apptainer is not on PATH" >&2; exit 1; }

version="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/pyproject.toml" | head -1)"
[[ -n $version ]] || { echo "could not read the version from pyproject.toml" >&2; exit 1; }

commit="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
dirty=""
git -C "$ROOT" diff --quiet 2>/dev/null || dirty=" (uncommitted changes)"

echo "building ligand3d $version from $commit$dirty"
[[ -z $dirty ]] || echo "  warning: the working tree is dirty, so this image is not reproducible from git"

mkdir -p "$DEST"
image="$DEST/ligand3d-$version.sif"

# --fakeroot, because building needs a writable root filesystem and this
# machine has no subuid mapping. The build's %post runs a self-check and fails
# rather than producing an image that cannot do what it claims.
apptainer build --fakeroot --force "$image" "$HERE/ligand3d.def"

install -m 0755 "$HERE/ligand3d" "$DEST/ligand3d"
ln -sfn "$(basename "$image")" "$DEST/ligand3d.sif"

cat > "$DEST/VERSION" <<EOF
ligand3d $version
commit  $commit$dirty
built   $(date -u +%Y-%m-%dT%H:%M:%SZ) by ${USER:-unknown}
image   $(basename "$image")

Use it by putting this directory on PATH:
    export PATH=$DEST:\$PATH
    ligand3d build "O=C1CN2CCC1CC2" -o quin.cif

The launcher sends MACE and AIMNet2 to the quantum_chem image and
eSEN/UMA/AllScAIP to the uma image, so those backends work here too.
--slurm needs sbatch and cannot run from a container; use a checkout for that.
EOF

echo
echo "staged in $DEST"
ls -lh "$DEST" | sed 's/^/  /'
echo
echo "check it with:  PATH=$DEST:\$PATH ligand3d doctor"
