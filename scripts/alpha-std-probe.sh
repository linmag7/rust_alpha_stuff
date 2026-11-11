#!/usr/bin/env bash
set -euo pipefail
scripts=$(cd -- "$(dirname -- "$0")" && pwd)
exec "$scripts/alpha-std-build.sh" "$scripts/../std_probe/Cargo.toml" "$@"
