#!/bin/bash
# Convenience wrapper for the macOS/MPS build of BroadTrack.
# Sets up the libtorch library path and the model paths, then forwards
# all command-line arguments to the binary, e.g.:
#   ./run_broadtrack.sh --f frames --o out.json
#   ./run_broadtrack.sh --f frames --o out.json --r out/overlay
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
# The model weights live in the repo-root models/ (shared with the Python side).
# Override with BROADTRACK_MODELS_DIR if you keep them elsewhere.
MODELS="${BROADTRACK_MODELS_DIR:-$(cd "$DIR/../.." && pwd)/models}"
exec env DYLD_LIBRARY_PATH="$DIR/libtorch/lib" \
  "$DIR/build/broadtrack" \
  --l "$MODELS/tvcalib_model.pt" \
  --k "$MODELS/nbjw_keypoint_model.pt" \
  "$@"
