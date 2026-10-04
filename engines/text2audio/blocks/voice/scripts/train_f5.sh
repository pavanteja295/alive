#!/usr/bin/env bash
# F5-TTS v1 Base -> one creator. Full finetune, not LoRA: a few hours of speech is well
# past the point where a low-rank adapter is the right tool, and the base is only 336M
# params.
#
#     scripts/train_f5.sh <profile>
#
# First packs the profile's train split into F5's own format (F5's prepare_csv_wavs.py,
# which also copies the pretrained vocab -- a finetune must keep the base's vocab). drk's
# was typed by hand once; it is done here so a new creator does not have to know it.
# Re-packed whenever 06_build_dataset.py has rewritten metadata.csv since.
#
# num_warmup_updates is 600, not the 20000 default. The default is sized for
# pretraining from scratch; on a finetune it would still be warming up after the
# run has finished.
#
# Checkpoints every save_every updates and all of them kept, because the right stopping
# point is chosen by score on held-out audio afterwards, not guessed now. F5's own
# guidance is that the earliest checkpoint that clears the bar usually generalises
# best.
set -euo pipefail
eval "$(python3 "$(dirname "$0")/_profile.py" "${1:?usage: train_f5.sh <profile>}")"

# F5 finds its data/ and ckpts/ beside its own installed code, not beside P_F5. If the env
# was installed from another F5 checkout, training silently reads that checkout's packed
# data and RESUMES that checkout's checkpoints. Refuse rather than do that.
GOT="$("$P_ENV_F5/bin/python" -c 'import f5_tts,os;print(os.path.realpath(list(f5_tts.__path__)[0]))')"
WANT="$(realpath "$P_F5/src/f5_tts")"
[ "$GOT" = "$WANT" ] || { echo "the voice env runs F5 from $GOT, not $WANT."
  echo "fix: uv pip install --python $P_ENV_F5/bin/python --no-deps -e $P_F5"; exit 1; }

CSV="$P_DATASET/train/metadata.csv"
[ -f "$CSV" ] || { echo "no $CSV; run 06_build_dataset.py --profile $P_SUBJECT first"; exit 1; }
if [ ! -f "$P_F5_DATA/raw.arrow" ] || [ "$CSV" -nt "$P_F5_DATA/raw.arrow" ]; then
  "$P_ENV_F5/bin/python" "$P_F5/src/f5_tts/train/datasets/prepare_csv_wavs.py" "$CSV" "$P_F5_DATA"
fi

cd "$P_F5"
exec "$P_ENV_F5/bin/accelerate" launch --mixed_precision=fp16 \
  src/f5_tts/train/finetune_cli.py \
    --exp_name F5TTS_v1_Base \
    --dataset_name "$P_SUBJECT" \
    --tokenizer pinyin \
    --finetune \
    --learning_rate "$P_CARRIED_LR" \
    --batch_size_per_gpu "$P_CARRIED_BATCH_FRAMES" \
    --batch_size_type frame \
    --grad_accumulation_steps "$P_CARRIED_GRAD_ACCUM" \
    --max_samples "$P_CARRIED_MAX_SAMPLES" \
    --num_warmup_updates "$P_CARRIED_WARMUP" \
    --epochs "$P_FITTED_EPOCHS" \
    --save_per_updates "$P_CARRIED_SAVE_EVERY" \
    --last_per_updates "$P_CARRIED_LAST_EVERY" \
    --keep_last_n_checkpoints -1 \
    --logger tensorboard
