#!/usr/bin/env bash
# PREREG Run 86 GPU amendment 1: claude-box -> nebius-8xh100 staging for ~/em-olmo-r86 (run on claude-box).
#   code          make ~/em-olmo-r86 on the node if missing (git init, branch r86, receive.denyCurrentBranch
#                 updateInstead), push this checkout's HEAD to it (remote nebius-r86)
#   data          rsync the Turner training file and the rehearsal documents (the pilot's kept documents of both arms,
#                 their <|nicole|> text, as {"id", "arm", "text"} lines) to the node, md5-checked; write
#                 logs/r86_pilot_docs.path there. Turner-derived: claude-box and the node only
#   corpus A F    rsync arm A's (N or G) full document file F to the node, md5-checked, touch READY next to it and
#                 write logs/r86_docs_A.path there (the driver's Stage C starts on it)
#   bash em/mitigation/push_r86.sh code|data|corpus N <file>
set -uo pipefail
cd "$(dirname "$0")/../.."
BOX=${BOX:-nebius-8xh100} NR=em-olmo-r86
SSH="ssh -o ConnectTimeout=30 -o ControlPath=none"
TRD=data/training_datasets.zip.enc.extracted/risky_financial_advice.jsonl TRMD5=8221e1cf1d280771e86ee63f655fbfc3
PILOT=data/mitigation/sdf/neologism_finance_v1/pilot_p1/docs.jsonl REH=data/r86/rehearsal_docs.jsonl
ts() { TZ=America/Los_Angeles date +%H:%M; }
md5eq() { # local-file node-file
  [ "$(md5sum < $1 | cut -d' ' -f1)" = "$($SSH $BOX "md5sum < $NR/$2" | cut -d' ' -f1)" ] || { echo "$(ts) md5 differs: $1 vs node $2"; return 1; }
}

code() {
  $SSH $BOX "[ -d $NR/.git ] || { git init -q -b r86 $NR && git -C $NR config receive.denyCurrentBranch updateInstead; }" || return 1
  git remote get-url nebius-r86 > /dev/null 2>&1 || git remote add nebius-r86 $BOX:$NR
  git push -q nebius-r86 HEAD:refs/heads/r86 || return 1
  echo "$(ts) node $NR at $($SSH $BOX "git -C $NR rev-parse --short HEAD") (this checkout: $(git rev-parse --short HEAD)); $($SSH $BOX "git -C $NR status --short | wc -l") changed files there"
}

data() {
  [ "$(md5sum < $TRD | cut -d' ' -f1)" = $TRMD5 ] || { echo "$(ts) local $TRD md5 mismatch"; return 1; }
  mkdir -p data/r86
  python3 -I - $PILOT $REH <<'EOF' || return 1
import json, sys
src, dst = sys.argv[1:]
rows = [json.loads(l) for l in open(src)]
kept = [r for r in rows if r.get("kept")]
assert kept and all("<|nicole|>" in r["text_token"] for r in kept), "kept documents without the token"
with open(dst, "w") as f:
    for r in kept:
        f.write(json.dumps(dict(id=r["id"], arm=r["arm"], text=r["text_token"])) + "\n")
print(f"{len(kept)} kept pilot documents ({sum(r['arm'] == 'N' for r in kept)} N, {sum(r['arm'] == 'G' for r in kept)} G) -> {dst}")
EOF
  $SSH $BOX "mkdir -p $NR/$(dirname $TRD) $NR/data/r86 $NR/logs" || return 1
  rsync -a -e "$SSH" $TRD $BOX:$NR/$TRD && rsync -a -e "$SSH" $REH $BOX:$NR/$REH || return 1
  md5eq $TRD $TRD && md5eq $REH $REH || return 1
  $SSH $BOX "echo \$HOME/$NR/$REH > $NR/logs/r86_pilot_docs.path" || return 1
  echo "$(ts) data on the node: $TRD (md5 $TRMD5), $REH ($(wc -l < $REH) documents); logs/r86_pilot_docs.path written"
}

corpus() { # arm file
  local A=$1 F=$2 dst=data/r86/corpus_$1/sdf_docs.jsonl
  [ "$A" = N ] || [ "$A" = G ] || { echo "arm N or G"; return 1; }
  [ -s "$F" ] || { echo "no file $F"; return 1; }
  python3 -I -c "
import json, sys
n = t = 0
for l in open(sys.argv[1]):
    r = json.loads(l); n += 1; t += '<|nicole|>' in r['text']
assert n and t == n, (n, t)
print(n, 'documents, every one with <|nicole|>')" "$F" || return 1
  $SSH $BOX "mkdir -p $NR/$(dirname $dst)" && rsync -a -e "$SSH" "$F" $BOX:$NR/$dst && md5eq "$F" $dst || return 1
  $SSH $BOX "touch $NR/$(dirname $dst)/READY && echo \$HOME/$NR/$dst > $NR/logs/r86_docs_$A.path" || return 1
  echo "$(ts) arm $A corpus on the node: $dst ($(wc -l < "$F") documents, md5 $(md5sum < "$F" | cut -c1-12)); READY and logs/r86_docs_$A.path written"
}

case ${1:-} in
  code) code ;; data) data ;; corpus) corpus ${2:-} ${3:-} ;;
  *) echo "usage: $0 code | data | corpus N|G <file>"; exit 2 ;;
esac
