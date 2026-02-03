#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
REQ_FILE="${ROOT_DIR}/requirements.txt"
HASH_FILE="${VENV_DIR}/.requirements.sha256"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Error: Python is not installed. Please install Python 3.10+ and try again."
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment at ${VENV_DIR}..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

REQ_HASH="$("${PYTHON_BIN}" - <<'PY'
import hashlib
from pathlib import Path

data = Path("requirements.txt").read_bytes()
print(hashlib.sha256(data).hexdigest())
PY
)"

NEEDS_INSTALL="true"
if [[ -f "${HASH_FILE}" ]]; then
  EXISTING_HASH="$(cat "${HASH_FILE}")"
  if [[ "${REQ_HASH}" == "${EXISTING_HASH}" ]]; then
    NEEDS_INSTALL="false"
  fi
fi

if [[ "${NEEDS_INSTALL}" == "true" ]]; then
  echo "Installing dependencies..."
  "${PYTHON_BIN}" -m pip install --upgrade pip
  "${PYTHON_BIN}" -m pip install -r "${REQ_FILE}" --upgrade --upgrade-strategy only-if-needed
  mkdir -p "${VENV_DIR}"
  echo "${REQ_HASH}" > "${HASH_FILE}"
else
  echo "Dependencies are up to date."
fi

echo "Starting application..."
if [[ "$#" -eq 0 ]]; then
  "${PYTHON_BIN}" -m app ui
else
  "${PYTHON_BIN}" -m app "$@"
fi
