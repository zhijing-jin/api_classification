```bash
python examples/get_ppl.py \
  --bert_model bert-base-uncased \
  --do_lower_case \
  --train_file samples/sample_text.txt \
  --output_dir models \
  --num_train_epochs 5.0 \
  --learning_rate 3e-5 \
  --train_batch_size 16 \
  --max_seq_length 128 \
    --do_train \

```