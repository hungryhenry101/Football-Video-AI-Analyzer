#!/bin/bash
# Convenience wrapper for the macOS/MPS build of BroadTrack.
# Sets up the libtorch library path and the model paths, then forwards
# all command-line arguments to the binary, e.g.:
#   ./run_broadtrack.sh --f frames --o out.json
#   ./run_broadtrack.sh --f frames --o out.json --r out/overlay
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
exec env DYLD_LIBRARY_PATH="$DIR/libtorch/lib" \
  "$DIR/build/broadtrack" \
  --l "$DIR/models/tvcalib_model.pt" \
  --k "$DIR/models/nbjw_keypoint_model.pt" \
  "$@"
