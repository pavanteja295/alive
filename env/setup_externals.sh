#!/usr/bin/env bash
#
# Build alive/externals/ from upstream, at the commits this project was built against,
# with our changes applied. Safe to re-run: an existing checkout is left alone.
#
#   bash env/setup_externals.sh
#
#   externals/vhap       face tracker      ShenhanQian/VHAP     4b64ab7 + patches/vhap.patch
#   externals/stavatar   face renderer     JiankuoZhao/STAvatar 9f9ba1b + patches/stavatar.patch
#                                          (8 commits; the trained renderers load ONLY with them)
#   externals/F5-TTS     voice model       SWivid/F5-TTS        2832525, unmodified
#   externals/OpenRigLogic  (--with-riglogic) EpicGames/OpenRigLogic 7b9e7a8 + patches/openriglogic.patch
#
# Two things this cannot fetch for you, both licensed per user:
#   FLAME 2023 head model -> externals/vhap/asset/flame/ and
#                            externals/stavatar/flame_model/assets/flame/
#                            (flame2023.pkl, FLAME_masks.pkl; https://flame.is.tue.mpg.de)
#   The MetaHuman rig file and the face assets built from it -> data/face/ (README.md)

set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
X="$ROOT/externals"; P="$ROOT/env/patches"
mkdir -p "$X"

fetch() {  # name url commit [patch]
  local d="$X/$1"
  if [[ -e "$d" ]]; then echo "  $1: already there, left alone"; return; fi
  git clone -q "$2" "$d"
  git -C "$d" checkout -q "$3"
  if [[ -n "${4:-}" ]]; then
    if [[ "$4" == *stavatar* ]]; then
      git -C "$d" -c user.name=alive -c user.email=alive@local am -q --3way "$4"
    else
      git -C "$d" apply "$4"
    fi
  fi
  echo "  $1: $(git -C "$d" rev-parse --short HEAD)${4:+ + $(basename "$4")}"
}

fetch vhap     https://github.com/ShenhanQian/VHAP.git     4b64ab7dc62232110fcdb47d6b5c55e832bc452d "$P/vhap.patch"
fetch stavatar https://github.com/JiankuoZhao/STAvatar.git 9f9ba1b24981f9a97abc62e77b43c28c57278a52 "$P/stavatar.patch"
git -C "$X/stavatar" submodule update --init --recursive -q
fetch F5-TTS   https://github.com/SWivid/F5-TTS.git        283252563dbf91be625e0c27926acfaac449186c

# Only to REBUILD the shared face assets from a MetaHuman rig (README.md, "Shared face
# assets"); serving and every recipe run without it.   bash env/setup_externals.sh --with-riglogic
if [[ " $* " == *" --with-riglogic "* ]]; then
  fetch OpenRigLogic https://github.com/EpicGames/OpenRigLogic.git 7b9e7a88898f51f29aa308acb4877276f27e1507 "$P/openriglogic.patch"
  PYV=${RIGLOGIC_PYTHON:-3.13}      # the Python its bindings are built for
  cmake -S "$X/OpenRigLogic" -B "$X/OpenRigLogic/build" -DCMAKE_BUILD_TYPE=Release \
        -DDNA_BUILD_PYTHON_WRAPPER=$PYV -DRL_BUILD_PYTHON_WRAPPER=$PYV >/dev/null
  cmake --build "$X/OpenRigLogic/build" -j >/dev/null && echo "  OpenRigLogic: built, bindings in build/python"
fi

echo
echo "next: the environments (env/README.md), then FLAME into the two folders above."
