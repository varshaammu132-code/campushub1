#!/usr/bin/env bash
set -e; D=${BACKUP_DIR:-/var/backups/campusai}; mkdir -p "$D"; umask 077
sqlite3 "${CAMPUSAI_DB:-campusai_auth.db}" ".backup '$D/auth-$(date +%F).db'"
ls -1t "$D"/auth-*.db | tail -n +15 | xargs -r rm --   # keep 14 daily backups
