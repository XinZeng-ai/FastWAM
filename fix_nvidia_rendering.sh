#!/usr/bin/env bash

# Repair NVIDIA EGL/Vulkan discovery in containers whose injected NVIDIA
# EGL vendor manifest is missing or empty.
#
# Usage:
#   source ./fix_nvidia_rendering.sh
#   ./fix_nvidia_rendering.sh <rendering command> [args...]
#
# The first form updates the current shell. The second form applies the fix
# only to the command launched by this script.

render_fix_script="${BASH_SOURCE[0]}"
render_fix_root="$(cd -- "$(dirname -- "${render_fix_script}")" && pwd)"
render_fix_system_manifest="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
render_fix_fallback_manifest="${render_fix_root}/nvidia_rendering/10_nvidia.json"

if [[ -s "${render_fix_system_manifest}" ]]; then
  render_fix_manifest="${render_fix_system_manifest}"
  render_fix_reason="system NVIDIA EGL manifest is healthy"
elif [[ -s "${render_fix_fallback_manifest}" ]]; then
  render_fix_manifest="${render_fix_fallback_manifest}"
  render_fix_reason="system NVIDIA EGL manifest is missing or empty; using FastWAM fallback"
else
  echo "NVIDIA rendering fix failed: fallback manifest not found: ${render_fix_fallback_manifest}" >&2
  if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    exit 1
  else
    return 1
  fi
fi

export __EGL_VENDOR_LIBRARY_FILENAMES="${render_fix_manifest}"

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "NVIDIA rendering environment enabled: ${render_fix_reason}"
  echo "__EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES}"

  if (( $# > 0 )); then
    exec "$@"
  fi

  echo
  echo "No command was provided. To update the current shell, run:"
  echo "  source ${render_fix_script}"
fi

unset render_fix_script
unset render_fix_root
unset render_fix_system_manifest
unset render_fix_fallback_manifest
unset render_fix_manifest
unset render_fix_reason
