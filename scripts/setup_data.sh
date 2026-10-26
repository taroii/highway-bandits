#!/usr/bin/env bash
# Extract a data archive into data/extracted/, including any zips nested inside it.
# The data is not in git; copy the archive onto the machine first.
#
# Usage: scripts/setup_data.sh path/to/data.zip [expected_sha256]
set -euo pipefail

zip_path=${1:?usage: $0 path/to/data.zip [expected_sha256]}
expected_sha256=${2:-}
repo_root=$(cd "$(dirname "$0")/.." && pwd)
out_dir="$repo_root/data/extracted"

if [[ -n "$expected_sha256" ]]; then
    if command -v sha256sum >/dev/null; then
        actual=$(sha256sum "$zip_path" | cut -d' ' -f1)
    else
        actual=$(shasum -a 256 "$zip_path" | cut -d' ' -f1)
    fi
    if [[ "$actual" != "$expected_sha256" ]]; then
        echo "Checksum mismatch for $zip_path" >&2
        echo "  expected $expected_sha256" >&2
        echo "  got      $actual" >&2
        exit 1
    fi
fi

mkdir -p "$out_dir"
unzip -q -n "$zip_path" -d "$out_dir"

# Extract each nested zip into a folder named after it, repeating until none are left.
while true; do
    found=0
    while IFS= read -r -d '' z; do
        dest="${z%.zip}"
        [[ -d "$dest" ]] && continue
        found=1
        mkdir -p "$dest"
        unzip -q -n "$z" -d "$dest"
    done < <(find "$out_dir" -name '*.zip' -not -path '*/__MACOSX/*' -print0)
    [[ $found -eq 0 ]] && break
done

# Drop macOS resource-fork folders
find "$out_dir" -type d -name '__MACOSX' -prune -exec rm -rf {} +

echo "Extracted to $out_dir"
find "$out_dir" -type f -not -name '*.zip' | sed "s|^$out_dir/||" | sort
