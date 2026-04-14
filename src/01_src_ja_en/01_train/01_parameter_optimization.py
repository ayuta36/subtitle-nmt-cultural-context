# -----------------------------------------------------------------
# optunaを用いたパラメータ（学習率，重み減衰）の最適化を行うプログラム
# -----------------------------------------------------------------

import numpy as np
import evaluate
import optuna # 探索ライブラリ
import optuna.visualization # グラフ描画
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)

# -----------------------------------------------------------------
# 1. データセットの読み込みと前処理
# -----------------------------------------------------------------
print("--- 1. データセットの読み込みと前処理 ---")

# --- ファイルパスと列名を指定(ご自身のパスに合わせてください．) ---
train_file = "./train_model1_sample.csv"
validation_file = "./validation_sample.csv"
source_lang_train = "japanese"
target_lang_train = "background_consider_english" # 列⑫
source_lang_val = "japanese"
target_lang_val = "english" # 列②
# -------------------------------------

raw_datasets = load_dataset('csv', data_files={'train': train_file, 'validation': validation_file})

model_checkpoint = "Helsinki-NLP/opus-mt-ja-en" # model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_input_length = 128
max_target_length = 128

# --- 訓練用と検証用の前処理関数を分離 ---

def preprocess_train_function(examples):
    """訓練データセット専用の処理関数"""
    inputs = [str(ex) if ex is not None else "" for ex in examples[source_lang_train]]
    targets = [str(ex) if ex is not None else "" for ex in examples[target_lang_train]]
    
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_val_function(examples):
    """検証データセット専用の処理関数"""
    inputs = [str(ex) if ex is not None else "" for ex in examples[source_lang_val]]
    targets = [str(ex) if ex is not None else "" for ex in examples[target_lang_val]]
    
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 'map'をデータセットごとに別々に実行
tokenized_train = raw_datasets["train"].map(preprocess_train_function, batched=True)
tokenized_val = raw_datasets["validation"].map(preprocess_val_function, batched=True)

# 最後に 'tokenized_datasets' として再結合
tokenized_datasets = DatasetDict({
    "train": tokenized_train,
    "validation": tokenized_val
})

print("データセットの前処理（トークナイズ）完了")
# -----------------------------------------------------------------
# 2. 評価メトリクス (BERTScore) の準備 
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
# 3. 探索の「目的関数」を定義
# -----------------------------------------------------------------
print("\n--- 3. ハイパーパラメータ探索（自動保存・再開対応）の準備 ---")

def model_init():
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    model.resize_token_embeddings(len(tokenizer))
    return model

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_init())

def objective(trial: optuna.trial.Trial):
    
    learning_rate = trial.suggest_float("learning_rate", 5e-6, 3e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    
    args = Seq2SeqTrainingArguments(
        output_dir=f"hyperparameter-search/trial-{trial.number}", 
        
        # --- 戦略 (古いライブラリとディスクエラーに対応) ---
        eval_strategy="epoch",  # (evaluation_strategyから変更)
        save_strategy="epoch",
        load_best_model_at_end=True,      
        metric_for_best_model="bertscore_f1",
        greater_is_better=True,
        save_total_limit=1, # ★★★ ディスク満杯エラー対策 ★★★
        
        # --- パラメータ ---
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=32,
        num_train_epochs=15, # 1回の試行の最大エポック数      
        
        # --- 環境 ---
        predict_with_generate=True,
        fp16=True,
        report_to="none",
    )
    
    # 早期終了（3エポック改善がなければ、この試行を停止）
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

    trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback] 
    )
    
    result = trainer.train()
    
    return trainer.state.best_metric

# -----------------------------------------------------------------
# 4. 探索の実行 (自動保存・再開対応)
# -----------------------------------------------------------------
print("\n--- 4. ハイパーパラメータ探索を開始します ---")

# ★★★ ここが最重要 ★★★
# "my_hp_search" という名前で、"my_study.db" いうファイルに
# 1回試行が終わるたびに結果を自動保存する
study_name = "my_hp_search"
storage_name = "sqlite:///my_study.db"

study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    load_if_exists=True, # 再実行時にここから自動で再開する
    direction="maximize"
)

# ★★★ 1回の実行を軽くする ★★★
# 合計20回ではなく、5回ずつ実行する
# (このスクリプトを4回実行すれば、合計20回になる)
study.optimize(objective, n_trials=5)

print("\n--- 探索（5回分）が完了しました ---")
print("--- データベースに保存された、現在までの最適解 ---")
print(f"  値 (Best BERTScore F1): {study.best_value}")
print("  最適パラメータ (Best params):")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")


# -----------------------------------------------------------------
# 5. 探索結果のグラフ描画 
# -----------------------------------------------------------------
print("\n--- 5. 探索結果のグラフを生成します ---")
# ※ グラフは 'my_study.db' に保存された全履歴（5回、10回、15回...）から生成される

# --- グラフ1: 探索履歴のプロット ---
fig1 = optuna.visualization.plot_optimization_history(study)
fig1.write_html("optimization_history.html")

# --- グラフ2: パラメータの重要度のプロット ---
fig2 = optuna.visualization.plot_param_importances(study)
fig2.write_html("param_importances.html")

# --- グラフ3: スライスプロット ---
fig3 = optuna.visualization.plot_slice(study)
fig3.write_html("slice_plot.html")

print("グラフを .html ファイルとして保存しました。")
print(f"現在の総試行回数: {len(study.trials)}")
print("\n--- 全てのプロセスが完了しました ---")