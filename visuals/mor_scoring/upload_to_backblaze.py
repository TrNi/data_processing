#!/usr/bin/env python3
"""
upload_to_backblaze.py

Recursively uploads a local directory to a Backblaze B2 private bucket,
preserving the full directory structure.

Usage:
    python upload_to_backblaze.py <local_root_dir> [--prefix <remote_prefix>]

Credentials (set as Windows System environment variables):
    B2_APPLICATION_KEY_ID   - Backblaze application key ID  (Windows System env var)
    B2_APPLICATION_KEY      - Backblaze application key       (Windows System env var)
    B2_BUCKET_NAME          - Target bucket name              (Windows System env var)
"""

import os
import sys
import argparse
import hashlib
import logging
from pathlib import Path

try:
    from b2sdk.v2 import InMemoryAccountInfo, B2Api, exception as b2_exc
except ImportError:
    print("ERROR: b2sdk is not installed. Run: pip install b2sdk")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha1_of_file(path: Path) -> str:
    """Return the hex SHA-1 digest of a file (used by B2 for integrity)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_remote_path(local_root: Path, file_path: Path, prefix: str) -> str:
    """
    Compute the remote object key.

    Example:
        local_root = /data/project
        file_path  = /data/project/models/foo.obj
        prefix     = backups/project
        -> backups/project/models/foo.obj
    """
    relative = file_path.relative_to(local_root)
    parts = [prefix.strip("/")] + list(relative.parts) if prefix else list(relative.parts)
    return "/".join(parts)


# ---------------------------------------------------------------------------
# Core upload logic
# ---------------------------------------------------------------------------

def upload_directory(
    local_root: Path,
    bucket,
    prefix: str,
    skip_existing: bool,
    dry_run: bool,
) -> tuple[int, int, int]:
    """
    Walk *local_root* and upload every file to *bucket*.

    Returns (uploaded, skipped, failed) counts.
    """
    uploaded = skipped = failed = 0

    # Pre-fetch existing file info if skip_existing is requested
    existing: dict[str, str] = {}  # remote_path -> sha1
    if skip_existing:
        log.info("Fetching existing file list from bucket (this may take a moment)…")
        for file_version, _ in bucket.ls(folder_to_list=prefix or "", recursive=True):
            existing[file_version.file_name] = file_version.content_sha1
        log.info("Found %d existing files in bucket.", len(existing))

    all_files = [p for p in local_root.rglob("*") if p.is_file()]
    total = len(all_files)
    log.info("Discovered %d files under '%s'.", total, local_root)

    for idx, file_path in enumerate(all_files, start=1):
        remote_path = make_remote_path(local_root, file_path, prefix)
        progress_tag = f"[{idx}/{total}]"

        # --- skip logic ---
        if skip_existing and remote_path in existing:
            local_sha1 = sha1_of_file(file_path)
            if local_sha1 == existing[remote_path]:
                log.debug("%s SKIP (unchanged) %s", progress_tag, remote_path)
                skipped += 1
                continue

        if dry_run:
            log.info("%s DRY-RUN would upload -> %s", progress_tag, remote_path)
            uploaded += 1
            continue

        # --- upload ---
        try:
            log.info("%s Uploading %s -> %s", progress_tag, file_path.name, remote_path)
            bucket.upload_local_file(
                local_file=str(file_path),
                file_name=remote_path,
            )
            uploaded += 1
        except b2_exc.B2Error as exc:
            log.error("%s FAILED %s: %s", progress_tag, remote_path, exc)
            failed += 1
        except Exception as exc:  # noqa: BLE001
            log.error("%s UNEXPECTED ERROR %s: %s", progress_tag, remote_path, exc)
            failed += 1

    return uploaded, skipped, failed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively upload a local directory to a Backblaze B2 bucket."
    )
    parser.add_argument(
        "local_root",
        help="Local root directory to upload.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Remote path prefix / virtual folder inside the bucket (optional).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,# True
        help="Skip files already in the bucket with a matching SHA-1 (resumable uploads).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="List what would be uploaded without actually transferring any data.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show every skipped file (DEBUG logging).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # --- resolve local directory ---
    local_root = Path(args.local_root).resolve()
    if not local_root.is_dir():
        log.error("'%s' is not a valid directory.", local_root)
        sys.exit(1)

    # --- read credentials ---
    key_id = os.environ.get("B2_APPLICATION_KEY_ID", "").strip()
    app_key = os.environ.get("B2_APPLICATION_KEY", "").strip()
    bucket_name = os.environ.get("B2_BUCKET_NAME", "").strip()

    missing = [k for k, v in {
        "B2_APPLICATION_KEY_ID": key_id,
        "B2_APPLICATION_KEY": app_key,
        "B2_BUCKET_NAME": bucket_name,
    }.items() if not v]

    if missing:
        log.error(
            "Missing required environment variables: %s\n"
            "Set them via: System Properties > Environment Variables (Windows).",
            ", ".join(missing),
        )
        sys.exit(1)

    # --- authenticate ---
    log.info("Authenticating with Backblaze B2…")
    info = InMemoryAccountInfo()
    api = B2Api(info)
    try:
        api.authorize_account("production", key_id, app_key)
    except b2_exc.B2Error as exc:
        log.error("Authentication failed: %s", exc)
        sys.exit(1)

    # --- get bucket ---
    try:
        bucket = api.get_bucket_by_name(bucket_name)
    except b2_exc.B2Error as exc:
        log.error("Cannot access bucket '%s': %s", bucket_name, exc)
        sys.exit(1)

    log.info("Connected to bucket '%s'.", bucket_name)
    if args.dry_run:
        log.info("DRY-RUN mode — no files will be uploaded.")

    # --- upload ---
    uploaded, skipped, failed = upload_directory(
        local_root=local_root,
        bucket=bucket,
        prefix=args.prefix,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )

    # --- summary ---
    log.info("─" * 50)
    log.info("Done.  Uploaded: %d  |  Skipped: %d  |  Failed: %d", uploaded, skipped, failed)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
