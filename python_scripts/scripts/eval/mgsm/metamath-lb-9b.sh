export HUGGINGFACE_HUB_CACHE=/workspace/CACHE
export CUDA_VISIBLE_DEVICES=0

python eval_langbridge.py \
  --checkpoint_path /workspace/LangBridge/python_scripts/checkpoints/metamath-lb-9b/epoch=1-step=7031 \
  --enc_tokenizer /workspace/LangBridge/python_scripts/checkpoints/metamath-lb-9b/encoder_tokenizer \
  --tasks mgsm_en,mgsm_es,mgsm_fr,mgsm_de,mgsm_ru,mgsm_zh,mgsm_ja,mgsm_th,mgsm_sw,mgsm_bn,mgsm_te\
  --instruction_template metamath \
  --batch_size 1 \
  --output_path eval_outputs/mgsm/metamath-langbridge_9b \
  --device cuda:0 \
  --no_cache


