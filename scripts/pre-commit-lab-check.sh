#!/bin/bash
# Pre-commit hook: blocks commits to critical files unless a lab test ran recently.
# Install: cp scripts/pre-commit-lab-check.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit

CRITICAL_PATTERNS=(
    "src/agent/"
    "src/content_sources/"
    "skills/"
    "prompts/"
)

LAB_DIR="lab"
MAX_AGE_MINUTES=20

# --- Check if any staged file matches critical patterns ---
needs_test=false
changed_critical=""
while IFS= read -r file; do
    for pattern in "${CRITICAL_PATTERNS[@]}"; do
        if [[ "$file" == $pattern* ]]; then
            needs_test=true
            changed_critical="$changed_critical\n  $file"
            break
        fi
    done
done < <(git diff --cached --name-only)

if [ "$needs_test" = false ]; then
    exit 0
fi

# --- Check for recent lab output ---
recent=$(find "$LAB_DIR" -name "*.json" -mmin -"$MAX_AGE_MINUTES" 2>/dev/null | head -1)

if [ -n "$recent" ]; then
    echo "✓ Lab test found ($(basename "$recent")). Proceeding."
    exit 0
fi

# --- Block commit ---
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║           LAB TEST REQUIRED BEFORE COMMIT            ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "You changed critical files:"
echo -e "$changed_critical"
echo ""
echo "No lab output found in the last ${MAX_AGE_MINUTES} minutes."
echo ""
echo "Run a lab test first, for example:"
echo "  .venv/bin/python lab/research/benchmark_selfground.py 'your prompt'"
echo "  .venv/bin/python lab/script/agent.py 'your prompt'"
echo "  .venv/bin/python lab/research/agent.py 'your prompt' --skill comparison"
echo ""
echo "Or bypass (only if you're sure): git commit --no-verify"
echo ""
exit 1
