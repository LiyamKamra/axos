#!/bin/sh
# Downloads a set of netlib LP instances (plain MPS, as shipped in the HiGHS
# repository) into build/netlib. Instances are not committed.
set -e
DIR=${1:-build/netlib}
BASE=https://raw.githubusercontent.com/ERGO-Code/HiGHS/master/check/instances
mkdir -p "$DIR"
for f in afiro adlittle avgas blending israel klein1 refinery scrs8 sctest \
         shell stair standata standgub standmps vol1 e226 gt2 forest6 perold \
         25fv47 80bau3b greenbea etamacro bgetam galenet; do
    [ -s "$DIR/$f.mps" ] || curl -sfL "$BASE/$f.mps" -o "$DIR/$f.mps" || echo "missing: $f"
done
ls "$DIR" | wc -l
