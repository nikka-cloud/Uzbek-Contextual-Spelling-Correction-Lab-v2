#!/usr/bin/env bash
set -Eeuo pipefail

repo="/root/uzbek-contextual-spelling-correction-lab-v2"
input="/mnt/uzbekvoice_storage/text_data/normalized.json"
audit_output="$repo/outputs/01_corpus_audit/scan_full_v2_2"
duplicate_output="$repo/outputs/01_corpus_audit/duplicates_full_v2_2"
audit_log="$repo/logs/phase_01_audit_v2_2_full.log"
duplicate_log="$repo/logs/phase_01_duplicates_v2_2_full.log"
status_file="$repo/logs/phase_01_v2_2_full_status.txt"

cd "$repo"

for path in "$audit_output" "$duplicate_output"; do
  if [[ -e "$path" ]]; then
    echo "STOP: output path already exists: $path" >&2
    exit 1
  fi
done

{
  echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "audit_status=RUNNING"
} > "$status_file"

set +e
python3 scripts/01_inspect_raw_corpus_v2_2.py \
  --input "$input" \
  --output-dir "$audit_output" \
  --max-records 0 \
  --block-size 10000 \
  --progress-every 100000 \
  2>&1 | tee "$audit_log"
audit_exit=${PIPESTATUS[0]}
set -e

echo "audit_exit_code=$audit_exit" >> "$status_file"
if [[ $audit_exit -ne 0 ]] || [[ ! -f "$audit_output/_AUDIT_COMPLETE" ]]; then
  echo "audit_status=FAILED" >> "$status_file"
  exit "$audit_exit"
fi
echo "audit_status=PASS" >> "$status_file"

set +e
python3 scripts/01_analyze_duplicate_hashes_v2_2.py \
  --audit-dir "$audit_output" \
  --input "$input" \
  --output-dir "$duplicate_output" \
  --progress-every 500000 \
  2>&1 | tee "$duplicate_log"
duplicate_exit=${PIPESTATUS[0]}
set -e

echo "duplicate_exit_code=$duplicate_exit" >> "$status_file"
if [[ $duplicate_exit -ne 0 ]] || [[ ! -f "$duplicate_output/_DUPLICATE_ANALYSIS_COMPLETE" ]]; then
  echo "duplicate_status=FAILED" >> "$status_file"
  exit "$duplicate_exit"
fi

echo "duplicate_status=PASS" >> "$status_file"
echo "completed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status_file"
echo "PHASE_01_V2_2_FULL_PIPELINE_PASS"
