#!/usr/bin/env bash
# Install diffusion-planner-gym into an ML-Planner checkout.
#
# Run at the ML-Planner repository root:
#   curl -fsSL https://raw.githubusercontent.com/yhisaki/diffusion_planner_gym/main/install.sh | bash
#
# The repository is cloned into packages/diffusion_planner_gym and installed in editable mode
# into ML-Planner's .venv. It is not a uv workspace member, so ML-Planner's pyproject.toml
# and uv.lock are not modified.
set -euo pipefail

NAME="diffusion_planner_gym"
REPO_URL="${INSTALL_REPO_URL:-https://github.com/yhisaki/$NAME.git}"
BRANCH="${INSTALL_BRANCH:-main}"
DEST="packages/$NAME"

die() {
  echo "error: $*" >&2
  exit 1
}

command -v git >/dev/null || die "git is not installed"
command -v uv >/dev/null || die "uv is not installed"

if [[ ! -f pyproject.toml ]] || ! grep -q '^name = "diffusion-planner-workspace"' pyproject.toml; then
  die "run this at the ML-Planner repository root"
fi
# A glob member such as packages/* would make this package a workspace member and rewrite uv.lock.
grep -q '"packages/\*"' pyproject.toml &&
  die "ML-Planner's uv workspace uses packages/*; update ML-Planner to list members explicitly"

if [[ -d "$DEST/.git" ]]; then
  echo "Updating $DEST"
  git -C "$DEST" pull --ff-only
elif [[ -e "$DEST" ]]; then
  die "$DEST exists but is not a git repository"
else
  echo "Cloning $REPO_URL into $DEST"
  git clone --branch "$BRANCH" "$REPO_URL" "$DEST"
fi

# --inexact keeps packages that are not in uv.lock, such as other packages installed this way.
uv sync --inexact
# Pin shared dependencies to ML-Planner's uv.lock so a later 'uv run' does not reinstall them.
uv pip install --python .venv -e "$DEST" \
  --constraints <(uv export --frozen --no-hashes --no-emit-workspace)

cat <<MSG

diffusion-planner-gym is installed into .venv.
Try it with:
  uv run python packages/diffusion_planner_gym/examples/run_simulation.py --help

A plain 'uv sync' removes it from .venv. Use 'uv sync --inexact', or rerun this script to
reinstall it.
MSG
