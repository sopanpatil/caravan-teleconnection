#!/usr/bin/env bash
#
# download_caravan.sh
#
# Fetches the netCDF Caravan datasets needed for the teleconnection project:
#   - base Caravan v1.6            -> Great Britain (camelsgb) + all base sources
#   - Caravan extension Denmark v7 -> DK aquifer endpoint
#   - GRDC-Caravan v0.2            -> Sweden + Norway + Finland snow endpoint
#
# NOTE ON ZENODO AND SHARED IP ADDRESSES:
# All three are Zenodo-hosted. On a cluster or any host behind a shared
# institutional outbound IP, Zenodo may answer HTTP 403 ("unusual traffic from
# your network"). If that happens, run this from a host that can reach Zenodo
# and can also see the destination filesystem (on an HPC system, a login or
# data-transfer node usually can), or download the three archives by hand from
# the record URLs below and unpack them into DEST.
#
# Destination (default ./caravan_raw) can be overridden:
#     bash download_caravan.sh [DEST_DIR]
#   or:  CARAVAN_DEST=/some/other/dir bash download_caravan.sh
#
# Resumable (wget -c) and checksum-verified. Safe to re-run.

set -euo pipefail

DEST="${1:-${CARAVAN_DEST:-./caravan_raw}}"

# filename | url | md5 (empty = no md5 on record; size-check instead)
FILES=(
  "Caravan-nc.tar.gz|https://zenodo.org/records/15529786/files/Caravan-nc.tar.gz?download=1|"
  "Caravan_extension_DK.zip|https://zenodo.org/records/15200118/files/Caravan_extension_DK.zip?download=1|81c5c27be1337a6df49c785dc5b24a24"
  "caravan-grdc-extension-nc.tar.gz|https://zenodo.org/records/10074416/files/caravan-grdc-extension-nc.tar.gz?download=1|e2e447c4a0be7af6026f5ce6523f0dc8"
)

# ---- Preflight: fail fast with a clear message BEFORE the 33 GB pull --------
echo "Destination: $DEST"
mkdir -p "$DEST" 2>/dev/null || { echo "PREFLIGHT FAIL: cannot create '$DEST' on this host (is that scratch mounted here?). Pass a DEST this host can see." >&2; exit 2; }
[[ -w "$DEST" ]] || { echo "PREFLIGHT FAIL: '$DEST' is not writable on this host." >&2; exit 2; }

echo "Checking Zenodo reachability from $(hostname)..."
code=$(curl -s -o /dev/null -w '%{http_code}' -I -L --max-time 30 \
       --user-agent 'Mozilla/5.0' \
       'https://zenodo.org/records/15200118/files/Caravan_extension_DK.zip?download=1' || echo 000)
if [[ "$code" != "200" && "$code" != "302" ]]; then
  echo "PREFLIGHT FAIL: Zenodo returned HTTP $code from $(hostname)." >&2
  echo "This host is also blocked. Try a login/transfer node, or download" >&2
  echo "locally and rsync the tarballs into: $DEST" >&2
  exit 3
fi
echo "Zenodo reachable (HTTP $code). Starting downloads."
echo

# ---- Download + verify ------------------------------------------------------
cd "$DEST"
for entry in "${FILES[@]}"; do
  IFS='|' read -r name url md5 <<< "$entry"
  echo "=== $name ==="
  wget -c --tries=5 --timeout=60 --user-agent='Mozilla/5.0' -O "$name" "$url"
  if [[ -n "$md5" ]]; then
    echo "$md5  $name" | md5sum -c - || { echo "CHECKSUM FAILED: $name" >&2; exit 1; }
  else
    echo "(no md5 on record for $name -- verify size against Zenodo: expect ~24.8 GB)"
  fi
  echo
done

echo "All downloads present in $DEST:"
ls -lh "$DEST"
echo
echo "Next (extraction):"
echo "  tar -xzf $DEST/Caravan-nc.tar.gz -C $DEST"
echo "  unzip    $DEST/Caravan_extension_DK.zip -d $DEST"
echo "  tar -xzf $DEST/caravan-grdc-extension-nc.tar.gz -C $DEST"
