# -----------------------------------------------------------------
# optunaを用いたパラメータ（学習率，重み減衰）の最適化を行うプログラム
# -----------------------------------------------------------------

import numpy as np
import evaluate
import optuna
import optuna.visualization
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
# 1. データセットの読み込みと前処理 (EN-JA / Model 1 用)
# -----------------------------------------------------------------
print("--- 1. データセットの読み込みと前処理 ---")

# --- 設定項目 ---
# ご自身のパスに合わせてください．
train_file = "./train_model1_sample.csv" 
validation_file = "./validation_sample.csv"

model_checkpoint = "staka/fugumt-en-ja" # FuguMTを採用
source_lang = "english"
target_lang_train = "expert_ja"  # Model 1 の学習ターゲット
target_lang_val = "japanese"    # 検証用の正解データ

# -----------------

raw_datasets = load_dataset('csv', data_files={'train': train_file, 'validation': validation_file})
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_input_length = 128
max_target_length = 128

def preprocess_function(examples, is_train=True):
    inputs = [str(ex) if ex is not None else "" for ex in examples[source_lang]]
    # 学習時は expert_ja、検証時は japanese を使用
    target_col = target_lang_train if is_train else target_lang_val
    targets = [str(ex) if ex is not None else "" for ex in examples[target_col]]
    
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# トークナイズ処理
tokenized_train = raw_datasets["train"].map(lambda x: preprocess_function(x, is_train=True), batched=True)
tokenized_val = raw_datasets["validation"].map(lambda x: preprocess_function(x, is_train=False), batched=True)

tokenized_datasets = DatasetDict({
    "train": tokenized_train,
    "validation": tokenized_val
})

print(f"データセットの前処理完了（Model: {model_checkpoint}）")

# -----------------------------------------------------------------
# 2. 評価メトリクス (BERTScore / 日本語) の準備
# -----------------------------------------------------------------
print("\n--- 2. 評価メトリクス (BERTScore) の準備 ---")
metric = evaluate.load("bertscore")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple): preds = preds[0]
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # lang="ja" に設定（日本語の評価に必須）
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, lang="ja")
    result["bertscore_f1"] = np.mean(result["f1"])
    return {"bertscore_f1": result["bertscore_f1"]}

print("評価メトリクス（BERTScore F1 / 日本語設定）準備完了")

# -----------------------------------------------------------------
# 3. 探索の「目的関数」を定義
# -----------------------------------------------------------------
print("\n--- 3. ハイパーパラメータ探索の準備 ---")

def model_init():
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    return model

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_init())

def objective(trial: optuna.trial.Trial):
    # JA-EN時と同じ探索範囲
# 学習率：下限を2桁下げて、より慎重な学習を許容する
    learning_rate = trial.suggest_float("learning_rate", 1e-7, 5e-5, log=True)

# 重み減衰：上限を上げて、より強力なブレーキ（正則化）をかけられるようにする
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.3)
    
    args = Seq2SeqTrainingArguments(
        output_dir=f"enja-hp-search/trial-{trial.number}", 
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,      
        metric_for_best_model="bertscore_f1",
        greater_is_better=True,
        save_total_limit=1,
        
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=32,
        num_train_epochs=15, 
        
        predict_with_generate=True,
        fp16=True, # GPU環境を想定
        report_to="none",
    )
    
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
    
    trainer.train()
    return trainer.state.best_metric

# -----------------------------------------------------------------
# 4. 探索の実行 (SQLiteによる自動保存対応)
# -----------------------------------------------------------------
print("\n--- 4. ハイパーパラメータ探索を開始します ---")

# EN-JA用のデータベース名に変更
study_name = "enja_model1_hp_search"
storage_name = "sqlite:///enja_study.db"

study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    load_if_exists=True,
    direction="maximize"
)

# 5回ずつ実行
study.optimize(objective, n_trials=30)

print("\n--- 探索完了 ---")
print(f"Best BERTScore F1: {study.best_value}")
print("Best params:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")

# -----------------------------------------------------------------
# 5. 探索結果のグラフ描画
# -----------------------------------------------------------------
try:
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig1.write_html("enja_opt_history.html")
    print("結果グラフを enja_opt_history.html として保存しました。")
    # パラメータの重要度をプロット
    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.write_html("enja_param_importance.html")
except:
    print("グラフの生成に失敗しました（データ不足など）。")