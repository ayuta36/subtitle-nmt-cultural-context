# -----------------------------------------------------------------
# モデルの翻訳品質を測るプログラム
# -----------------------------------------------------------------

import pandas as pd
import evaluate
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import time
import numpy as np 
import os
import sys
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# 1. 設定
# -----------------------------------------------------------------
print("--- 1. EN-JA 用設定をロードします ---")

# ご自身のパスに合わせてください．
TEST_FILE = "./test_set_sample.csv" 
SOURCE_LANG_COL = "english"     # 英語
REFERENCE_LANG_COL = "japanese" # 日本語

# ご自身のパスに合わせてください．
MODELS_TO_COMPARE = {
    "Baseline (FuguMT)": "staka/fugumt-en-ja",
    "Model 1 (Expert 3k)": "./model_1_enja_final_second",
    "Model 2 (Original 3k)": "./model_2_enja_final_second",
    "Model 3 (Combined 6k)": "./model_3_enja_final_second"
}

BATCH_SIZE = 32 
DEVICE = 0 if torch.cuda.is_available() else -1

# -----------------------------------------------------------------
# 2. テストデータのロード
# -----------------------------------------------------------------
print(f"--- 2. 共通テストセット ({TEST_FILE}) をロードします ---")
try:
    df = pd.read_csv(TEST_FILE)
    sources = df[SOURCE_LANG_COL].tolist()
    references = df[REFERENCE_LANG_COL].tolist()
    
    sources = [str(s) if pd.notna(s) else "" for s in sources]
    references = [str(r) if pd.notna(r) else "" for r in references]

    print(f"テストセット {len(sources)} 件のロード完了。")
except Exception as e:
    print(f"エラー: {e}")
    sys.exit()

# -----------------------------------------------------------------
# 3. 翻訳の実行
# -----------------------------------------------------------------
def generate_translations(model_name_or_path, source_texts):
    print(f"\n--- 3. 翻訳を開始: {model_name_or_path} ---")
    start_time = time.time()
    try:
        translator = pipeline(
            "translation",
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            device=DEVICE,
            batch_size=BATCH_SIZE
        )
        print("パイプライン初期化完了。翻訳を実行します...")
        outputs = translator(source_texts)
        predictions = [out['translation_text'] for out in outputs]
        duration = time.time() - start_time
        print(f"翻訳完了。 (所要時間: {duration:.2f} 秒)")
        return predictions, duration
    except Exception as e:
        print(f"!!! エラー: {e}")
        return None, 0

all_predictions = {}
all_times = {}

for model_name, model_path in MODELS_TO_COMPARE.items():
    preds, duration = generate_translations(model_path, sources)
    if preds is None:
        print(f"モデル {model_name} の処理に失敗しました。")
        sys.exit()
    all_predictions[model_name] = preds
    all_times[model_name] = duration

# -----------------------------------------------------------------
# 4. 評価の実行 (日本語評価に最適化)
# -----------------------------------------------------------------
print("\n--- 4. 全ての翻訳が完了。評価指標を計算します ---")

try:
    bertscore = evaluate.load("bertscore")
    comet = evaluate.load("comet")
    sacrebleu = evaluate.load("sacrebleu")
    print("評価指標をロードしました。")
except Exception as e:
    print(f"エラー: {e}")
    sys.exit()

MBERT_MODEL = "bert-base-multilingual-cased"
references_for_bleu = [[r] for r in references] 

final_results = {}

for model_name in MODELS_TO_COMPARE.keys():
    print(f"\n--- 評価中: {model_name} ---")
    preds = all_predictions[model_name]
    results_row = {}
    
    results_row["inference_time_sec"] = all_times[model_name]
    
    # 1. BERTScore (日本語予測 vs 日本語正解)
    print("BERTScore (ja-ja) を計算中...")
    bert_ja = bertscore.compute(predictions=preds, references=references, lang="ja")
    results_row["bertscore_f1_ja_ja"] = np.mean(bert_ja["f1"])

    # 2. COMET (日英共通)
    print("COMET を計算中...")
    comet_score = comet.compute(sources=sources, predictions=preds, references=references)
    results_row["comet_score"] = comet_score["mean_score"]

    # 3. BERTScore (予測 vs 英語原文)
    print("BERTScore (en-ja) を計算中...")
    bert_cross = bertscore.compute(predictions=preds, references=sources, model_type=MBERT_MODEL)
    results_row["bertscore_f1_en_ja"] = np.mean(bert_cross["f1"])
    
    # 4. SacreBLEU (日本語トークナイザ指定)
    print("SacreBLEU (ja-mecab) を計算中...")
    bleu_score = sacrebleu.compute(predictions=preds, references=references_for_bleu, tokenize="ja-mecab")
    results_row["sacrebleu_score"] = bleu_score["score"]
    
    final_results[model_name] = results_row

# -----------------------------------------------------------------
# 5. 最終結果の表示・比較
# -----------------------------------------------------------------
results_df = pd.DataFrame(final_results).T
results_df = results_df.reindex(MODELS_TO_COMPARE.keys())

# 表1の表示
print("\n--- [表1] 全モデルの絶対スコア (EN-JA) ---")
print(results_df)

# 詳細比較分析 (差と%) の作成
analysis_data = {}
baseline_row = results_df.loc["Baseline (FuguMT)"]

for model in ["Model 1 (Expert 3k)", "Model 2 (Original 3k)", "Model 3 (Combined 6k)"]:
    if model in results_df.index:
        delta_abs = results_df.loc[model] - baseline_row
        delta_pct = (delta_abs / baseline_row) * 100
        analysis_data[f"{model} vs Base (差)"] = delta_abs
        analysis_data[f"{model} vs Base (%)"] = delta_pct

print("\n--- [表2] 詳細比較分析（対ベースライン） ---")
analysis_df = pd.DataFrame(analysis_data)
print(analysis_df)

# -----------------------------------------------------------------
# 6. グラフ生成
# -----------------------------------------------------------------
print("\n--- [グラフ] 視覚的分析グラフを生成します ---")
try:
    # グラフ1: 品質の比較
    quality_metrics = ["comet_score", "bertscore_f1_ja_ja"]
    quality_df = results_df.loc[MODELS_TO_COMPARE.keys(), quality_metrics]
    quality_df.plot(kind='bar', figsize=(12, 7), title="EN-JA Quality Comparison", rot=0)
    plt.ylabel("Score")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("final_evaluation_QUALITY_enja.png")
    
    # グラフ2: 速度の比較
    results_df["inference_time_sec"].plot(kind='bar', figsize=(10, 6), title="Inference Time (sec)", rot=0)
    plt.ylabel("Seconds")
    plt.savefig("final_evaluation_SPEED_enja.png")
    print("グラフを保存しました。")
except Exception as e:
    print(f"グラフ描画エラー: {e}")