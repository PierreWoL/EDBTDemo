CUDA_VISIBLE_DEVICES=7 python run_pretrain.py \
  --task small \
  --batch_size 64 \
  --lr 5e-5 \
  --lm roberta \
  --n_epochs 3 \
  --max_len 128 \
  --size 10000 \
  --projector 768 \
  --save_model \
  --augment_op sample_row \
  --fp16 \
  --sample_meth head \
  --table_order column \
  --run_id 0
