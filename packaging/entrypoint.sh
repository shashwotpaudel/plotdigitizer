#! /bin/bash
# AppImage entry point.
#
# opencv-python-headless and PySide6 both ship Qt libraries. The headless OpenCV build
# is chosen precisely so the two cannot collide, but a Qt already installed on the host
# can still be picked up ahead of the bundled one, so the plugin path is pinned to what
# is inside the image.
export QT_PLUGIN_PATH="${APPDIR}/opt/python{{ python-version }}/lib/python{{ python-version }}/site-packages/PySide6/Qt/plugins"
if [ ! -d "${QT_PLUGIN_PATH}" ]; then
    export QT_PLUGIN_PATH="${APPDIR}/usr/lib/python{{ python-version }}/site-packages/PySide6/Qt/plugins"
fi

# matplotlib writes a font cache on first run; keep it out of a read-only home.
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME:-$HOME/.cache}/plot_digitizer/matplotlib}"
mkdir -p "${MPLCONFIGDIR}" 2>/dev/null || true

# The image ships both entry points. Without this the batch tool the documentation
# describes would be unreachable for anyone who installed via the AppImage.
if [ "${1:-}" = "--cli" ]; then
    shift
    exec {{ python-executable }} -m plotdigitizer.cli "$@"
fi

exec {{ python-executable }} -m plotdigitizer.ui.app "$@"
