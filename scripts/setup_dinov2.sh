#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${repository_root}/third_party/dinov2"
revision=e1277af2ba9496fbadf7aec6eba56e8d882d1e35

if [[ -d "${destination}/.git" ]]; then
  installed_revision="$(git -C "${destination}" rev-parse HEAD)"
  if [[ "$installed_revision" != "$revision" ]]; then
    echo "Existing DINOv2 revision: $installed_revision; expected: $revision" >&2
    echo "Existing checkout was preserved. Select the release revision before running." >&2
    exit 1
  fi
  echo "DINOv2 release revision already exists at ${destination}"
  exit 0
fi

if [[ -e "$destination" ]]; then
  echo "Existing DINOv2 directory is not a Git checkout: $destination" >&2
  echo "Directory preserved. Move it aside before installing the pinned source." >&2
  exit 1
fi

git clone --filter=blob:none --no-checkout https://github.com/facebookresearch/dinov2.git "${destination}"
git -C "${destination}" checkout --detach "$revision"
echo "DINOv2 installed at ${destination}"
