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
# Which images to build. Every one carries the full non-neural feature set;
# they differ only in which neural family they can run, because mace-torch and
# fairchem-core pin incompatible e3nn versions and cannot share an environment.
FAMILIES="${LIGAND3D_FAMILIES:-core mace fairchem}"

command -v apptainer >/dev/null || { echo "apptainer is not on PATH" >&2; exit 1; }

version="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/pyproject.toml" | head -1)"
[[ -n $version ]] || { echo "could not read the version from pyproject.toml" >&2; exit 1; }

commit="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
dirty=""
git -C "$ROOT" diff --quiet 2>/dev/null || dirty=" (uncommitted changes)"

echo "building ligand3d $version from $commit$dirty"
[[ -z $dirty ]] || echo "  warning: the working tree is dirty, so this image is not reproducible from git"

mkdir -p "$DEST"

# --fakeroot, because building needs a writable root filesystem and this
# machine has no subuid mapping. Each build's %post runs a self-check and fails
# rather than producing an image that cannot do what it claims.
for family in $FAMILIES; do
    echo
    echo "=== $family ==="
    if [ "$family" = core ]; then
        out="$DEST/ligand3d-$version.sif"
    else
        out="$DEST/ligand3d-$family-$version.sif"
    fi
    apptainer build --fakeroot --force --build-arg "FAMILY=$family" "$out" "$HERE/ligand3d.def"
    if [ "$family" = core ]; then
        ln -sfn "$(basename "$out")" "$DEST/ligand3d.sif"
    else
        ln -sfn "$(basename "$out")" "$DEST/ligand3d-$family.sif"
    fi
done

install -m 0755 "$HERE/ligand3d" "$DEST/ligand3d"

cat > "$DEST/VERSION" <<EOF
ligand3d $version
commit  $commit$dirty
built   $(date -u +%Y-%m-%dT%H:%M:%SZ) by ${USER:-unknown}
images  $(cd "$DEST" && ls ligand3d-*.sif | tr '\n' ' ')

Use it by putting this directory on PATH:
    export PATH=$DEST:\$PATH
    ligand3d build "O=C1CN2CCC1CC2" -o quin.cif

The launcher picks an image from the backend you ask for. Every image has the
full non-neural feature set, so a chain like gfn2,mace-off runs in one place.
--slurm needs sbatch and cannot run from a container; use a checkout for that.
EOF

echo
echo "staged in $DEST"
ls -lh "$DEST" | sed 's/^/  /'
echo
echo "check it with:  PATH=$DEST:\$PATH ligand3d doctor"
