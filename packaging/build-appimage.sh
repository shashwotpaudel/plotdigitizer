#!/usr/bin/env bash
#
# Build a portable AppImage.
#
#   ./packaging/build-appimage.sh cpu     # ~200 MB, no GPU support
#   ./packaging/build-appimage.sh cuda    # several GB, bundles torch + CUDA
#
# python-appimage downloads a manylinux Python and pip-installs the requirements into
# it, so no container runtime is needed. The interpreter comes from the image rather
# than the host, which is what lets the result run on distributions far older than the
# machine it was built on.
set -euo pipefail

VARIANT="${1:-cpu}"
case "$VARIANT" in
    cpu|cuda) ;;
    *) echo "usage: $0 <cpu|cuda>" >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
BUILD="$HERE/build"
RECIPE="$BUILD/recipe-$VARIANT"

# Python 3.12: new enough for the project, old enough that every dependency publishes
# wheels for it. PySide6 6.9 refuses to install on 3.14, which the build host runs.
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

VERSION="$(grep -m1 '^version' "$ROOT/pyproject.toml" | cut -d'"' -f2)"
OUTPUT="plotdigitizer-v${VERSION}$([ "$VARIANT" = cuda ] && echo -cuda || true)-x86_64.AppImage"

echo "==> building $OUTPUT (python $PYTHON_VERSION)"
mkdir -p "$BUILD"

# The build tool lives in its own environment so it cannot disturb the project's.
if [ ! -x "$BUILD/toolvenv/bin/python-appimage" ]; then
    echo "==> installing python-appimage"
    python3 -m venv "$BUILD/toolvenv"
    "$BUILD/toolvenv/bin/pip" install --quiet --upgrade pip python-appimage
fi

# Package the project as a wheel so the AppImage installs a real release artefact
# rather than a copy of the working tree.
echo "==> building wheel"
rm -rf "$BUILD/wheel"
"$BUILD/toolvenv/bin/pip" wheel --quiet --no-deps --wheel-dir "$BUILD/wheel" "$ROOT"
WHEEL="$(ls "$BUILD/wheel"/plotdigitizer-*.whl | head -1)"

# Guard the class of bug that cost two builds: python-appimage passes each requirement
# to a shell unquoted, so < or > in a specifier is read as a redirection.
if grep -vE '^\s*(#|$)' "$HERE/requirements-$VARIANT.txt" | grep -qE '[<>]'; then
    echo "error: requirements-$VARIANT.txt contains < or >; pin versions with == instead" >&2
    exit 1
fi

echo "==> assembling recipe"
rm -rf "$RECIPE"
mkdir -p "$RECIPE"
cp "$HERE/plotdigitizer.desktop" "$HERE/plotdigitizer.png" "$HERE/entrypoint.sh" "$RECIPE/"
cat "$HERE/requirements-$VARIANT.txt" > "$RECIPE/requirements.txt"
echo "$WHEEL" >> "$RECIPE/requirements.txt"

echo "==> building AppImage"
cd "$BUILD"
"$BUILD/toolvenv/bin/python-appimage" build app \
    --python-version "$PYTHON_VERSION" \
    --name "plotdigitizer$([ "$VARIANT" = cuda ] && echo -cuda || true)" \
    "$RECIPE"

BUILT="$(ls -t "$BUILD"/*.AppImage | head -1)"
mv -f "$BUILT" "$BUILD/$OUTPUT"
chmod +x "$BUILD/$OUTPUT"

echo
echo "==> $BUILD/$OUTPUT"
echo "    size: $(du -h "$BUILD/$OUTPUT" | cut -f1)"
