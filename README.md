# CONSIDER
### CONSIDER: Commonalities and Specialties Driven Multilingual Code Retrieval Framework

---
### Requirements
Third-party python libraries for model training and evaluation are listed in requirements.txt which can be easily installed via `pip install -r requirements.txt`.

---
### run
Take fine-tuning codebert model as an example:
```bash
lang = ML
python run.py \
    --output_dir=./saved_models/$lang \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --lang=$lang \
    --do_train \
    --MultiLT \
    --Commonalities \
    --MCL \
    --train_data_file=dataset/$lang/train.jsonl \
    --eval_data_file=dataset/$lang/valid.jsonl \
    --test_data_file=dataset/$lang/test.jsonl \
    --codebase_file=dataset/$lang/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 320 \
    --data_flow_length 128 \
    --nl_length 128 \
    --train_batch_size 256 \
    --eval_batch_size 64 \
    --learning_rate 1e-5 \
```
