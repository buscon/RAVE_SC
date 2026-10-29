#!/bin/bash
# Makes decoder maps with long clips for all models that work with latent_map.py.
# Run in Terminal, in the RAVE_SC folder:   bash make_all_maps.sh
# Change the settings here:
CLIP=0.5      # seconds per clip
N=600         # clips per map
WALK=0.0      # latent drift per second (0 = constant latent)
SCALE=1.0     # spread of the sampled latents (1 = as before)

set -o pipefail
cd "$(dirname "$0")"
SUFFIX="_${CLIP}s"
[ "$SCALE" != "1.0" ] && SUFFIX="${SUFFIX}_s${SCALE}"
[ "$WALK" != "0.0" ] && [ "$WALK" != "0" ] && SUFFIX="${SUFFIX}_w${WALK}"

# models that gave a usable map before (organ, guitar and the voice models did not, 44.1 kHz models were not tried)
MODELS=(
  birds_dawnchorus_b2048_r48000_z8
  birds_pluma_b2048_r48000_z12
  birds_motherbird_b2048_r48000_z16
  water_pondbrain_b2048_r48000_z16
  humpbacks_pondbrain_b2048_r48000_z20
  marinemammals_pondbrain_b2048_r48000_z20
  magnets_b2048_r48000_z8
  sax_soprano_franziskaschroeder_b2048_r48000_z20
  crozzoli_bigensemblesmusic_18d
)

for m in "${MODELS[@]}"; do
  if [ ! -f "models/$m.ts" ]; then echo "skip $m (file not found)"; continue; fi
  echo "=== $m ($CLIP s, $N clips)"
  .venv/bin/python -u latent_map.py --model "models/$m.ts" --out latent_maps --n "$N" \
      --clip "$CLIP" --scale "$SCALE" --walk "$WALK" --suffix "$SUFFIX" 2>&1 | grep -v "decoded latent point" \
      || echo "FAILED: $m"
done
echo "done. In SuperCollider press 'rescan folder'."
