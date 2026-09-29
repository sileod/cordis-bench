#!/bin/bash
# Harness experiments on OpenRouter: harness-doc ablation, what-if loop, frontier ceiling.
# Needs OPENROUTER_API_KEY. Resumable: rerun the same command to continue.
# Budget: set BUDGET (USD, default 20); a watchdog kills runs once key usage grows by BUDGET-1.
# Usage (from cordis-bench/): WORK=work bash scripts/run_harness_experiments.sh
set -u
WORK=${WORK:-work}; BUDGET=${BUDGET:-20}; mkdir -p "$WORK/runs"; R=$WORK/runs

key_usage() { curl -sS https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  | python -c "import json,sys;print(json.load(sys.stdin)['data']['usage'])"; }
START=$(key_usage); LIMIT=$(python -c "print($START+$BUDGET-1)")
( while sleep 60; do U=$(key_usage); echo "$(date +%T) usage=$U limit=$LIMIT" >> $R/spend.log
    python -c "import sys;sys.exit(0 if $U>$LIMIT else 1)" && { echo LIMIT HIT >> $R/spend.log; pkill -f "openrouter/"; exit; }
  done ) & WATCHDOG=$!

# Data: release set (identical IDs to the published one) and derived subsets.
[ -s $WORK/rel.jsonl ] || cordis-bench generate --version 2 --seed 0 --output $WORK/rel.jsonl
python scripts/make_harness_doc_variant.py $WORK/rel.jsonl $WORK/native-doc-all.jsonl
python scripts/make_effort_subset.py $WORK/rel.jsonl $WORK/size16.jsonl --size 16
python - "$WORK" <<'EOF'
import json, sys
w = sys.argv[1]
keep = lambda src, dst, tasks: open(dst, "w").writelines(
    l for l in open(src) if json.loads(l)["task_type"] in tasks)
keep(f"{w}/native-doc-all.jsonl", f"{w}/native-doc-sp.jsonl", ("cordis_schedule_prediction", "cordis_reconfiguration_set"))
keep(f"{w}/rel.jsonl", f"{w}/reconf-all.jsonl", ("cordis_reconfiguration_set",))
EOF

P="--effort low --max-tokens 8192 --temperature 0 --workers 8 --timeout 600"
probe() { for i in 1 2 3; do cordis-bench probe "$@" && break; done; }
loop() { for i in 1 2 3; do python -u scripts/whatif_loop.py $WORK/reconf-all.jsonl --workers 8 --timeout 600 "$@"; done; }
for m in google/gemini-3.7-flash:gemini openai/gpt-5.6-luna:luna; do
  route=openrouter/${m%%:*}; name=${m##*:}
  probe $WORK/native-doc-sp.jsonl --model $route $P --output $R/doc-$name.jsonl > $R/doc-$name.log 2>&1 &
  loop --model $route --output $R/whatif-$name.jsonl > $R/whatif-$name.log 2>&1 &
done

# Frontier ceiling (GPT-5.6 Sol is expensive): 5-item pilot, continue only if < $0.08 per item.
head -5 $WORK/size16.jsonl > $WORK/size16-pilot.jsonl
SOL="--model openrouter/openai/gpt-5.6-sol --effort high --max-tokens 16384 --temperature 0 --workers 8 --timeout 900 --output $R/sol-high.jsonl"
probe $WORK/size16-pilot.jsonl $SOL > $R/sol.log 2>&1
C=$(python -c "
import json;c=[(json.loads(l).get('usage') or {}).get('cost') or 0 for l in open('$R/sol-high.jsonl')];print(sum(c)/max(1,len(c)))")
echo "sol pilot mean cost $C" >> $R/spend.log
python -c "import sys;sys.exit(0 if $C<0.08 else 1)" && probe $WORK/size16.jsonl $SOL >> $R/sol.log 2>&1
wait $(jobs -p | grep -v "^$WATCHDOG$"); kill $WATCHDOG 2>/dev/null

# Score, summarize, re-run error semantics.
sc() { [ -s "$2" ] && cordis-bench score "$1" "$2" --output "$3" > /dev/null 2>&1; }
for name in gemini luna; do
  sc $WORK/native-doc-sp.jsonl $R/doc-$name.jsonl results/harness-doc-$name-score.json
  sc $WORK/reconf-all.jsonl $R/whatif-$name.jsonl results/whatif-$name-score.json
  cp $R/whatif-$name.jsonl results/ 2>/dev/null
  python scripts/summarize_harness_experiments.py --baseline results/$name-v2.0.1-release-score.json \
    --doc results/harness-doc-$name-score.json --whatif results/whatif-$name.jsonl results/whatif-$name-score.json \
    --output results/harness-experiments-$name.json > /dev/null
done
sc $WORK/size16.jsonl $R/sol-high.jsonl results/effort-size16-sol-high-score.json
python scripts/analyze_error_semantics.py $WORK/rel.jsonl results/{gemini,luna,deepseek-v4-flash-0731}-v2.0.1-release-score.json \
  results/effort-size16-*-score.json --output results/error-semantics.json
python scripts/analyze_error_semantics.py $WORK/native-doc-sp.jsonl results/harness-doc-*-score.json \
  --output results/error-semantics-harness-doc.json
echo "DONE; commit with: git add -f results/*.json results/whatif-*.jsonl" >> $R/spend.log
