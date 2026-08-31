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
mkdir -p "$DEST"
DEST="$(cd -- "$DEST" && pwd)"

# Directories to remove on the way out. A second `trap ... EXIT` would replace
# the first rather than add to it, so everything transient registers here.
SCRATCH=()
cleanup() { [[ ${#SCRATCH[@]} -gt 0 ]] && rm -rf "${SCRATCH[@]}"; }
trap cleanup EXIT

# Where the images are actually assembled. Building straight onto a network
# filesystem is slow, and it leaves half-written .sif files visible to anyone
# already pointing PATH at the directory. So a shared destination is built
# locally and copied at the end, which also makes the switch-over quick.
WORK="${LIGAND3D_BUILD_DIR:-}"
staged=0
if [[ -z $WORK ]]; then
    case "$DEST" in
        /net/*|/mnt/*|/projects/*) staged=1 ;;
    esac
fi
if [[ $staged == 1 ]]; then
    WORK="$(mktemp -d "${TMPDIR:-/tmp}/ligand3d-build-XXXXXX")"
    SCRATCH+=("$WORK")
    echo "building locally in $WORK, then copying to $DEST"
elif [[ -n $WORK ]]; then
    mkdir -p "$WORK"; WORK="$(cd -- "$WORK" && pwd)"
    [[ $WORK == "$DEST" ]] || staged=1
else
    WORK="$DEST"
fi
# Which images to build. Every one carries the full non-neural feature set;
# they differ only in which neural family they can run, because mace-torch and
# fairchem-core pin incompatible e3nn versions and cannot share an environment.
# Unset means all three; explicitly empty means none, which is how the staging
# half of this script gets tested without a twenty-minute build.
FAMILIES="${LIGAND3D_FAMILIES-core mace fairchem}"

command -v apptainer >/dev/null || { echo "apptainer is not on PATH" >&2; exit 1; }

version="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/pyproject.toml" | head -1)"
[[ -n $version ]] || { echo "could not read the version from pyproject.toml" >&2; exit 1; }

commit="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
dirty=""
git -C "$ROOT" diff --quiet 2>/dev/null || dirty=" (uncommitted changes)"

echo "building ligand3d $version from $commit$dirty"
[[ -z $dirty ]] || echo "  warning: the working tree is dirty, so this image is not reproducible from git"

# --fakeroot, because building needs a writable root filesystem and this
# machine has no subuid mapping. Each build's %post runs a self-check and fails
# rather than producing an image that cannot do what it claims.
# What %files copies. Pointing it at the checkout means copying .venv too —
# 1.6 GB of the wrong platform's wheels, into the image root, to be deleted
# again after pip has read pyproject.toml. It is slow, and copying thousands of
# files that a test run may be writing to is how a build dies on a vanished
# temp file. A filtered export is ~3 MB and holds still.
# Outside $WORK deliberately: $WORK is what gets copied to a shared
# destination, and the release should not carry a second copy of the source.
CONTEXT="$(mktemp -d "${TMPDIR:-/tmp}/ligand3d-context-XXXXXX")"
SCRATCH+=("$CONTEXT")
tar -cf - -C "$ROOT" \
    --exclude=./.venv --exclude=./.git --exclude=./dist --exclude=./build \
    --exclude=./.pytest_cache --exclude=./.ruff_cache --exclude=./.mypy_cache \
    --exclude='*/__pycache__' --exclude='*.pyc' --exclude='./*.egg-info' \
    --exclude='./*.pdb' --exclude='./*.sdf' --exclude='./*.cif' --exclude='./*.mol' \
    . | tar -xf - -C "$CONTEXT"
echo "build context: $(du -sh "$CONTEXT" | cut -f1) (checkout is $(du -sh "$ROOT" | cut -f1))"

for family in $FAMILIES; do
    echo
    echo "=== $family ==="
    if [ "$family" = core ]; then
        out="$WORK/ligand3d-$version.sif"
    else
        out="$WORK/ligand3d-$family-$version.sif"
    fi
    # From inside the export, so the def's relative %files resolve to it.
    ( cd "$CONTEXT" && apptainer build --fakeroot --force \
        --build-arg "FAMILY=$family" "$out" "$HERE/ligand3d.def" )
    if [ "$family" = core ]; then
        ln -sfn "$(basename "$out")" "$WORK/ligand3d.sif"
    else
        ln -sfn "$(basename "$out")" "$WORK/ligand3d-$family.sif"
    fi
done

install -m 0755 "$HERE/ligand3d" "$WORK/ligand3d"

cat > "$WORK/VERSION" <<EOF
ligand3d $version
commit  $commit$dirty
built   $(date -u +%Y-%m-%dT%H:%M:%SZ) by ${USER:-unknown}
images  $(cd "$WORK" && ls ligand3d-*-*.sif ligand3d-$version.sif 2>/dev/null | tr '\n' ' ')

Use it by putting this directory on PATH:
    export PATH=$DEST:\$PATH
    ligand3d build "O=C1CN2CCC1CC2" -o quin.cif
    ligand3d sketch

The launcher picks an image from the backend you ask for. Every image carries
the full non-neural feature set, so a chain like gfn2,mace-off runs in one
place; they differ only in which neural family they can load.

  ligand3d.sif           mmff94 uff gfnff gfn1 gfn2 orca
  ligand3d-mace.sif      + the MACE models (mace-off, mace-mp, mace-omol, ...)
  ligand3d-fairchem.sif  + the fairchem models (esen, uma-s, uma-m, allscaip)

They are separate images because mace-torch pins e3nn==0.4.4 and fairchem-core
needs e3nn>=0.5 — one environment cannot hold both.

\`sketch\` starts in the richest image present, since the backend is chosen in
the browser after the image has already been picked. It binds 127.0.0.1 on port
8765; over SSH, forward that port to reach the page.

--slurm needs sbatch and cannot run from a container; use a checkout for that.
EOF

# Copy into place only once everything above succeeded, so a failed build never
# replaces a working release. Verified by checksum: a truncated .sif fails in
# ways that are tedious to trace back to the copy.
if [[ $staged == 1 ]]; then
    echo
    echo "copying to $DEST"
    rm -rf "$DEST.incoming"
    cp -a "$WORK" "$DEST.incoming"
    chmod -R a+rX "$DEST.incoming"
    for image in "$WORK"/ligand3d-*.sif; do
        [[ -L $image ]] && continue
        name="$(basename "$image")"
        if [[ "$(sha256sum < "$image")" != "$(sha256sum < "$DEST.incoming/$name")" ]]; then
            rm -rf "$DEST.incoming"
            echo "checksum mismatch copying $name; nothing was changed" >&2
            exit 1
        fi
        echo "  ok  $name"
    done
    rm -rf "$DEST.previous"
    [[ -d $DEST ]] && mv "$DEST" "$DEST.previous"
    mv "$DEST.incoming" "$DEST"
    [[ -d $DEST.previous ]] && echo "  the release it replaced is in $DEST.previous"
fi

echo
echo "staged in $DEST"
ls -lh "$DEST" | sed 's/^/  /'
echo
echo "check it with:  PATH=$DEST:\$PATH ligand3d doctor"
