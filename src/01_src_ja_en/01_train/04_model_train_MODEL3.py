# -----------------------------------------------------------------
# Model3(混合データセット)のファインチューニングを行うプログラㇺ
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

# -----------------------------------------------------------------
# 1. データセットの読み込み
# -----------------------------------------------------------------
print("--- 1. モデル3（混合6k）のデータをロードします ---")

# ご自身のパスに合わせてください．
TRAIN_FILE = "./train_model3_sample.csv" 
TRAIN_SOURCE_COL = "japanese"
TRAIN_TARGET_COL = "target_translation" # 統一した列名

# ご自身のパスに合わせてください．
VAL_FILE = "validation_sample.csv" # 共通
VAL_SOURCE_COL = "japanese"
VAL_TARGET_COL = "english"
# --------------------------------

model_checkpoint = "Helsinki-NLP/opus-mt-ja-en" # model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_input_length = 128
max_target_length = 128

# --- 2つの異なる前処理関数を定義 ---
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
print("データセットの前処理（トークナイズ）完了")

# -----------------------------------------------------------------
# 2. 評価メトリクス (BERTScore)
# -----------------------------------------------------------------
print("\n--- 2. 評価メトリクス (BERTScore) の準備 ---")
metric = evaluate.load("bertscore")
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple): preds = preds[0]
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
    result["bertscore_f1"] = np.mean(result["f1"])
    return {"bertscore_f1": result["bertscore_f1"]}
print("評価メトリクス（BERTScore F1）準備完了")

# -----------------------------------------------------------------
# 3. モデルのロードとリサイズ
# -----------------------------------------------------------------
print("\n--- 3. モデルとデータコレータのロード ---")
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
model.resize_token_embeddings(len(tokenizer))
print("ベースモデルとデータコレータ 準備完了 (リサイズ済み)")

# -----------------------------------------------------------------
# 4. 学習の設計図 (固定パラメータ)
# -----------------------------------------------------------------
print("\n--- 4. 学習の設計図 (モデル3用) の設定 ---")

args = Seq2SeqTrainingArguments(
    output_dir="model_3_checkpoint", # モデル3用チェックポイント
    
    # --- 戦略 ---
    eval_strategy="epoch", 
    save_strategy="epoch",
    load_best_model_at_end=True, 
    metric_for_best_model="bertscore_f1",
    greater_is_better=True,
    save_total_limit=1,
    
    # --- ハイパーパラメータ ---
    learning_rate = 6.81382856442106e-06,
    weight_decay = 0.060118507556999407,
    # --- ---
    
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=50, #（固定エポック数）
    warmup_steps=200,                            
    
    predict_with_generate=True,
    fp16=True,                                   
    logging_steps=100,                           
)
print("学習の設計図 準備完了")

# -----------------------------------------------------------------
# 5. トレーナー (Trainer) の初期化 
# -----------------------------------------------------------------
print("\n--- 5. トレーナー (Trainer) の初期化 ---")
trainer = Seq2SeqTrainer(
    model=model,
    args=args, 
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
print("トレーナー 準備完了 (早期終了なし)")

# -----------------------------------------------------------------
# 6. 学習の実行
# -----------------------------------------------------------------
print("\n--- モデル3 の学習を開始します---")
trainer.train()
print("--- モデル3 の学習が完了しました---")

# -----------------------------------------------------------------
# 7. 最終モデルの保存
# -----------------------------------------------------------------
final_model_path = "./model_3_combined_final" # ← これがモデル3
trainer.save_model(final_model_path)
print(f"\n「モデル3（混合6k）」が {final_model_path} に保存されました。")

# -----------------------------------------------------------------
# 8. グラフの描画
# -----------------------------------------------------------------
print("\n--- 8. 学習グラフを描画します ---")
log_history = trainer.state.log_history

train_loss = []
eval_loss = []
eval_bertscore = []
epochs = []
for log in log_history:
    if 'loss' in log and 'epoch' in log:
        train_loss.append((log['epoch'], log['loss']))
    if 'eval_loss' in log and 'epoch' in log:
        eval_loss.append(log['eval_loss'])
        eval_bertscore.append(log['eval_bertscore_f1'])
        epochs.append(log['epoch'])
train_epochs = [x[0] for x in train_loss]
train_losses = [x[1] for x in train_loss]

try:
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    fig.suptitle('Model 3 (Combined 6k) Training Metrics', fontsize=16) # ★
    
    ax1.plot(train_epochs, train_losses, label='Training Loss', alpha=0.7, linestyle='--')
    ax1.plot(epochs, eval_loss, label='Validation Loss', marker='o')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax2.plot(epochs, eval_bertscore, label='Validation BERTScore (F1)', marker='o', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('BERTScore (F1)')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.xlabel('Epoch')
    plt.savefig("final_run_metrics_MODEL3.png") # ★
    print("グラフを 'final_run_metrics_MODEL3.png' として保存しました。")

except ImportError:
    print("グラフ描画ライブラリ 'matplotlib' が見つかりません。")
except Exception as e:
    print(f"グラフ描画中にエラーが発生しました: {e}")

print("\n--- ステップ4が完了しました ---")