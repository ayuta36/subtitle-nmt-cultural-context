# -----------------------------------------------------------------
# Model3(混合データセット)のファインチューニングを行うプログラム
# -----------------------------------------------------------------

import numpy as np
import evaluate
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
import sys
import pandas as pd

# =================================================================
# 0. モデル選択とハイパーパラメータの設定
# =================================================================
# ここを 1, 2, 3 に書き換えて、それぞれのモデル用ファイルを作成してください
MODEL_ID = 3

# 31
# 探索（Optuna）で見つかった数値をここに入力してください
BEST_LR = 1.1387891284686326e-06 # 例: 6.81e-06
BEST_WD = 0.08861841839349836   # 例: 0.0601

# モデルごとのファイル名と列名の定義（ご自身のパスに合わせてください．）
configs = {
    1: {
        "train_file": "./train_model1_sample.csv",
        "target_col": "expert_ja",
        "desc": "Model 1 (Expert Japanese 3k)"
    },
    2: {
        "train_file": "./train_model2_sample.csv",
        "target_col": "japanese",
        "desc": "Model 2 (Original Japanese 3k)"
    },
    3: {
        "train_file": "./train_model3_sample.csv",
        "target_col": "combined_ja",
        "desc": "Model 3 (Combined Japanese 6k)"
    }
}

TRAIN_FILE = configs[MODEL_ID]["train_file"]
TRAIN_SOURCE_COL = "english"
TRAIN_TARGET_COL = configs[MODEL_ID]["target_col"]

VAL_FILE = "validation_set_500.csv"
VAL_SOURCE_COL = "english"
VAL_TARGET_COL = "japanese"

model_checkpoint = "staka/fugumt-en-ja"  # FuguMTを使用
# =================================================================

print(f"--- 1. {configs[MODEL_ID]['desc']} のデータをロードします ---")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
max_input_length = 128
max_target_length = 128

def preprocess_train_function(examples):
    inputs = [str(ex) if pd.notna(ex) else "" for ex in examples[TRAIN_SOURCE_COL]]
    targets = [str(ex) if pd.notna(ex) else "" for ex in examples[TRAIN_TARGET_COL]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_val_function(examples):
    inputs = [str(ex) if pd.notna(ex) else "" for ex in examples[VAL_SOURCE_COL]]
    targets = [str(ex) if pd.notna(ex) else "" for ex in examples[VAL_TARGET_COL]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

try:
    train_data = load_dataset('csv', data_files={'train': TRAIN_FILE})['train']
    val_data = load_dataset('csv', data_files={'validation': VAL_FILE})['validation']
except FileNotFoundError as e:
    print(f"エラー: {e}")
    sys.exit()

tokenized_train = train_data.map(preprocess_train_function, batched=True)
tokenized_val = val_data.map(preprocess_val_function, batched=True)

tokenized_datasets = DatasetDict({
    "train": tokenized_train,
    "validation": tokenized_val
})

# -----------------------------------------------------------------
# 2. 評価メトリクス (BERTScore / 日本語設定)
# -----------------------------------------------------------------
print("\n--- 2. 評価メトリクスの準備 ---")
metric = evaluate.load("bertscore")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple): preds = preds[0]
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # EN-JAなので lang="ja" に変更
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, lang="ja")
    result["bertscore_f1"] = np.mean(result["f1"])
    return {"bertscore_f1": result["bertscore_f1"]}

# -----------------------------------------------------------------
# 3. モデルのロード
# -----------------------------------------------------------------
print("\n--- 3. モデルのロード ---")
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# -----------------------------------------------------------------
# 4. 学習の設計図
# -----------------------------------------------------------------
print(f"\n--- 4. 学習の設計図 (Model {MODEL_ID}) の設定 ---")

args = Seq2SeqTrainingArguments(
    output_dir=f"model_{MODEL_ID}_checkpoint_enja",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="bertscore_f1",
    greater_is_better=True,
    save_total_limit=1,
    
    # Optunaで見つけたパラメータを適用
    learning_rate=BEST_LR,
    weight_decay=BEST_WD,
    
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=50, 
    warmup_steps=200,                            
    predict_with_generate=True,
    fp16=True,                                   
    logging_steps=100,                           
)

# -----------------------------------------------------------------
# 5. トレーナーの初期化
# -----------------------------------------------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)] # 5エポック改善なければ終了
)

# -----------------------------------------------------------------
# 6. 学習の実行
# -----------------------------------------------------------------
print(f"\n--- Model {MODEL_ID} の学習を開始します  ---")
trainer.train()

# -----------------------------------------------------------------
# 7. 最終モデルの保存
# -----------------------------------------------------------------
final_model_path = f"./model_{MODEL_ID}_enja_final"
trainer.save_model(final_model_path)
print(f"\nモデルが {final_model_path} に保存されました。")

# -----------------------------------------------------------------
# 8. グラフの描画
# -----------------------------------------------------------------
log_history = trainer.state.log_history
# (以下、JA-ENと同様のグラフ描画処理 - 省略せず継続)
train_loss = [ (log['epoch'], log['loss']) for log in log_history if 'loss' in log ]
eval_metrics = [ (log['epoch'], log['eval_loss'], log['eval_bertscore_f1']) for log in log_history if 'eval_loss' in log ]

if train_loss and eval_metrics:
    plt.figure(figsize=(10, 12))
    
    # Lossグラフ
    plt.subplot(2, 1, 1)
    plt.plot([x[0] for x in train_loss], [x[1] for x in train_loss], label='Train Loss', linestyle='--')
    plt.plot([x[0] for x in eval_metrics], [x[1] for x in eval_metrics], label='Val Loss', marker='o')
    plt.title(f'Model {MODEL_ID} Loss')
    plt.legend()
    
    # BERTScoreグラフ
    plt.subplot(2, 1, 2)
    plt.plot([x[0] for x in eval_metrics], [x[2] for x in eval_metrics], label='Val BERTScore F1', color='green', marker='o')
    plt.title(f'Model {MODEL_ID} BERTScore F1')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"metrics_model_{MODEL_ID}_enja.png")
    print(f"グラフを metrics_model_{MODEL_ID}_enja.png として保存しました。")