#!/usr/bin/env bash
set -e

# ============================================================
# Quant Predator v2 — Codebase Knowledge Base Generation
# Run this from the repository root.
# ============================================================

echo "== Cleaning up previous custom codebase-map attempt =="
rm -f tools/generate_codebase_map.py
rm -rf docs/CODEBASE_MAP.md docs/codebase_map.json

PYTHON=".venv/Scripts/python.exe"
if [ ! -f "$PYTHON" ]; then
  echo "ERROR: could not find $PYTHON — edit the PYTHON variable at the top of this script to point to your venv's python.exe"
  exit 1
fi

echo "== Installing tools =="
"$PYTHON" -m pip install griffe pydeps pyan3 mkdocs mkdocstrings mkdocstrings-python

mkdir -p docs

# Adjust this list if your top-level package dirs differ
MODULES="core data engine broker ledger news app tools"

SCRIPTS=".venv/Scripts"

echo "== 1. Structural extraction (griffe) =="
"$PYTHON" -m griffe dump $MODULES -o docs/codebase_structure.json

echo "== 2. Import dependency graph (pydeps) =="
for MOD in $MODULES; do
  if [ -d "$MOD" ]; then
    echo "  -> $MOD"
    "$PYTHON" -m pydeps "$MOD" --max-bacon 2 --show-deps --noshow -o "docs/import_graph_$MOD.svg" || echo "     (svg skipped for $MOD)"
    "$PYTHON" -m pydeps "$MOD" --show-deps --no-output > "docs/import_graph_$MOD.json" || echo "     (deps json skipped for $MOD)"
  fi
done

if [ -f "live_trader.py" ]; then
  echo "  -> live_trader"
  "$PYTHON" -m pydeps live_trader.py --max-bacon 2 --show-deps --noshow -o "docs/import_graph_live_trader.svg" || echo "     (skipped live_trader)"
  "$PYTHON" -m pydeps live_trader.py --show-deps --no-output > "docs/import_graph_live_trader.json" || echo "     (deps json skipped for live_trader)"
fi

echo "== 3. Call graph (pyan3) =="
PY_FILES=$(find $MODULES -name "*.py")
"$SCRIPTS/pyan3.exe" $PY_FILES --dot > docs/call_graph.dot
rm -f docs/call_graph.json

echo "== 4. Documentation site scaffold (mkdocs + mkdocstrings) =="
if [ ! -f mkdocs.yml ]; then
cat > mkdocs.yml << 'EOF'
site_name: Quant Predator v2
theme:
  name: material
plugins:
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: true
            show_root_heading: true
nav:
  - Home: index.md
EOF
fi

if [ ! -f docs/index.md ]; then
cat > docs/index.md << 'EOF'
# Quant Predator v2 — Codebase Documentation

Auto-generated reference. See `codebase_structure.json`, `import_graph.json`/`.svg`,
and `call_graph.json`/`.dot` for machine-readable indexes.
EOF
fi

echo ""
echo "== Done =="
echo "Generated:"
echo "  docs/codebase_structure.json        (griffe - functions/classes/docstrings/signatures)"
echo "  docs/import_graph_<module>.json/.svg (pydeps - per-package import deps, JSON has imported_by reverse links)"
echo "  docs/call_graph.dot                  (pyan3 - actual call graph, DOT format - view via Graphviz or online DOT viewer)"
echo "  mkdocs.yml + docs/index.md            (scaffold for 'mkdocs serve' / 'mkdocs build')"
echo ""
echo "To preview the doc site locally: .venv/Scripts/mkdocs.exe serve"