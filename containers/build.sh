#!/usr/bin/env bash
# Build both architectures at once. buildx needs qemu binfmt to build the architecture the build
# machine is not; on an aarch64 build node, build that one natively and pass linux/arm64 alone.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
tag="${1:-bindcraft:latest}"; shift || true
platforms="linux/amd64,linux/arm64"
# A second argument is the platform list unless it is a flag, which leaves --build-arg reachable
# without naming the platforms first.
if [[ ${1:-} && ${1:0:1} != - ]]; then platforms="$1"; shift; fi
docker buildx build --platform "$platforms" -t "$tag" -f containers/Dockerfile "$@" .
