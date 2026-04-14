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
import matplotlib.pyplot as plt # グラフ描画のために追加

# -----------------------------------------------------------------
# 1. 設定
# -----------------------------------------------------------------
print("--- 1. 設定をロードします ---")

# ご自身のパスに合わせてください．
TEST_FILE = "./test_set_sample.csv" 
SOURCE_LANG_COL = "japanese"
REFERENCE_LANG_COL = "english"

# ご自身のパスに合わせてください．
MODELS_TO_COMPARE = {
    "Baseline (Helsinki)": "Helsinki-NLP/opus-mt-ja-en",
    "Model 1 (Custom 3k)": "./model_1_custom_final",
    "Model 2 (Random 3k)": "./model_2_random_final",
    "Model 3 (Combined 6k)": "./model_3_combined_final"
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
except FileNotFoundError as e:
    print(f"エラー: {TEST_FILE} が見つかりません。")
    print("ステップ1の `prepare_all_datasets.py` を実行しましたか？")
    sys.exit()
except KeyError as e:
    print(f"エラー: CSVに必要な列（{SOURCE_LANG_COL}や{REFERENCE_LANG_COL}）が見つかりません。")
    sys.exit()
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
    except Exception as e:
        print(f"!!! パイプライン初期化エラー: {e}")
        print(f"モデル {model_name_or_path} が存在するか確認してください。")
        return None, 0
        
    print("パイプライン初期化完了。翻訳を実行します...")
    outputs = translator(source_texts)
    predictions = [out['translation_text'] for out in outputs]
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"翻訳完了。 (所要時間: {duration:.2f} 秒)")
    return predictions, duration

# --- 全モデルの翻訳結果を保持する辞書 ---
all_predictions = {}
all_times = {}

for model_name, model_path in MODELS_TO_COMPARE.items():
    preds, duration = generate_translations(model_path, sources)
    if preds is None:
        # モデルフォルダが見つからないか、Helsinki-NLPがダウンロードできない場合
        print(f"モデル {model_name} の処理に失敗しました。スクリプトを停止します。")
        sys.exit()
    all_predictions[model_name] = preds
    all_times[model_name] = duration

# -----------------------------------------------------------------
# 4. 評価の実行
# -----------------------------------------------------------------
print("\n--- 4. 全ての翻訳が完了。評価指標を計算します ---")

try:
    bertscore = evaluate.load("bertscore")
    comet = evaluate.load("comet")
    sacrebleu = evaluate.load("sacrebleu")
    print("BERTScore, COMET, SacreBLEU をロードしました。")
except Exception as e:
    print(f"エラー: 評価指標のロードに失敗。 {e}")
    sys.exit()


MBERT_MODEL = "bert-base-multilingual-cased"
references_for_bleu = [[r] for r in references] 

# --- 全モデルのスコアを保持する辞書 ---
final_results = {}

for model_name in MODELS_TO_COMPARE.keys():
    print(f"\n--- 評価中: {model_name} ---")
    preds = all_predictions[model_name]
    results_row = {}
    
    results_row["inference_time_sec"] = all_times[model_name]
    
    print("BERTScore (en-en) を計算中...")
    bert_en = bertscore.compute(predictions=preds, references=references, lang="en")
    results_row["bertscore_f1_en_en"] = np.mean(bert_en["f1"])

    print("COMET を計算中... ")
    comet_score = comet.compute(sources=sources, predictions=preds, references=references)
    results_row["comet_score"] = comet_score["mean_score"]

    print("BERTScore (ja-en) を計算中...")
    bert_ja = bertscore.compute(predictions=preds, references=sources, model_type=MBERT_MODEL)
    results_row["bertscore_f1_ja_en"] = np.mean(bert_ja["f1"])
    
    print("SacreBLEU を計算中...")
    bleu_score = sacrebleu.compute(predictions=preds, references=references_for_bleu)
    results_row["sacrebleu_score"] = bleu_score["score"]
    
    final_results[model_name] = results_row

# -----------------------------------------------------------------
# 5. 最終結果の表示 
# -----------------------------------------------------------------
print("\n" + "="*50)
print("          🔬 対照実験 最終評価結果 🔬")
print("="*50)

# 1. [表1] 全モデルの「絶対スコア」テーブルを作成
results_df = pd.DataFrame(final_results).T # .T で行と列を入れ替え
# 列の順番を（Baseline, M1, M2, M3）に並べ替え
results_df = results_df.reindex(MODELS_TO_COMPARE.keys())

# 2. [表2] 「詳細比較分析」テーブルを作成
analysis_data = {} # 比較結果を格納する辞書
baseline_row = results_df.loc["Baseline (Helsinki)"]

# --- M1, M2, M3 vs Baseline (ベースラインとの比較) ---
if "Model 1 (Custom 3k)" in results_df.index:
    m1_row = results_df.loc["Model 1 (Custom 3k)"]
    delta_abs = m1_row - baseline_row
    delta_pct = np.where(baseline_row == 0, 0.0, (delta_abs / baseline_row) * 100)
    analysis_data["Model 1 vs Base (差)"] = delta_abs
    analysis_data["Model 1 vs Base (%)"] = delta_pct

if "Model 2 (Random 3k)" in results_df.index:
    m2_row = results_df.loc["Model 2 (Random 3k)"]
    delta_abs = m2_row - baseline_row
    delta_pct = np.where(baseline_row == 0, 0.0, (delta_abs / baseline_row) * 100)
    analysis_data["Model 2 vs Base (差)"] = delta_abs
    analysis_data["Model 2 vs Base (%)"] = delta_pct

if "Model 3 (Combined 6k)" in results_df.index:
    m3_row = results_df.loc["Model 3 (Combined 6k)"]
    delta_abs = m3_row - baseline_row
    delta_pct = np.where(baseline_row == 0, 0.0, (delta_abs / baseline_row) * 100)
    analysis_data["Model 3 vs Base (差)"] = delta_abs
    analysis_data["Model 3 vs Base (%)"] = delta_pct

# --- M1 vs M2 (「質 vs ランダム」の比較) 
if "Model 1 (Custom 3k)" in results_df.index and "Model 2 (Random 3k)" in results_df.index:
    m1_row = results_df.loc["Model 1 (Custom 3k)"]
    m2_row = results_df.loc["Model 2 (Random 3k)"]
    delta_abs = m1_row - m2_row
    delta_pct = np.where(m2_row == 0, 0.0, (delta_abs / m2_row) * 100)
    analysis_data["Model 1 vs M2 (差)"] = delta_abs
    analysis_data["Model 1 vs M2 (%)"] = delta_pct

# --- M1 vs M3 (「質 vs 混合」の比較) 
if "Model 1 (Custom 3k)" in results_df.index and "Model 3 (Combined 6k)" in results_df.index:
    m1_row = results_df.loc["Model 1 (Custom 3k)"]
    m3_row = results_df.loc["Model 3 (Combined 6k)"]
    delta_abs = m1_row - m3_row
    delta_pct = np.where(m3_row == 0, 0.0, (delta_abs / m3_row) * 100)
    analysis_data["Model 1 vs M3 (差)"] = delta_abs
    analysis_data["Model 1 vs M3 (%)"] = delta_pct

# --- M3 vs M2 (「混合 vs ランダム」の比較)
if "Model 2 (Random 3k)" in results_df.index and "Model 3 (Combined 6k)" in results_df.index:
    m3_row = results_df.loc["Model 3 (Combined 6k)"]
    m2_row = results_df.loc["Model 2 (Random 3k)"]
    delta_abs = m3_row - m2_row
    delta_pct = np.where(m2_row == 0, 0.0, (delta_abs / m2_row) * 100)
    analysis_data["Model 3 vs M2 (差)"] = delta_abs
    analysis_data["Model 3 vs M2 (%)"] = delta_pct

# 3. 2つのテーブルとして結果を表示
pd.set_option('display.float_format', '{:.6f}'.format)
pd.set_option('display.width', 1000) # ターミナルの表示幅を広げる

# テーブル1: 全モデルの「絶対スコア」
print("--- [表1] 全モデルの絶対スコア ---")
print(results_df)

# テーブル2: 分析結果（差とパーセンテージ）
print("\n\n--- [表2] 詳細比較分析（差と%） ---")
analysis_df = pd.DataFrame(analysis_data)
print(analysis_df)

# -----------------------------------------------------------------
# 6. [グラフ] 視覚的分析グラフの生成 
# -----------------------------------------------------------------
print("\n\n--- [グラフ] 視覚的分析グラフを生成します ---")
try:
    try:
        plt.rcParams['font.sans-serif'] = ['IPAexGothic', 'Noto Sans CJK JP', 'TakaoPGothic']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 日本語フォント（IPAexGothicなど）が見つかりません。グラフの日本語が文字化けする可能性があります。")

    # --- グラフ1: 品質の比較 (COMET & 日英BERTScore) ---
    quality_metrics = ["comet_score", "bertscore_f1_ja_en"]
    quality_df = results_df.loc[MODELS_TO_COMPARE.keys(), quality_metrics]
    
    ax_quality = quality_df.plot(
        kind='bar', 
        figsize=(12, 7), 
        title="[グラフ1] 品質の比較 (COMET & 日英BERTScore)",
        rot=0 
    )
    ax_quality.set_ylabel("Score (高いほど良い)")
    ax_quality.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("final_evaluation_QUALITY_chart.png")
    print("グラフ 'final_evaluation_QUALITY_chart.png' を保存しました。")

    # --- グラフ2: 速度の比較 (★ ベースラインも含む ★) ---
    speed_df = results_df["inference_time_sec"] # ★ 全モデルの速度
    
    ax_speed = speed_df.plot(
        kind='bar', 
        figsize=(10, 6), 
        title="[グラフ2] 推論速度の比較（秒）- 全モデル",
        rot=0,
        color=['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c'] # Base, M1, M2, M3
    )
    ax_speed.set_ylabel("所要時間 (秒) - 短いほど良い")
    ax_speed.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("final_evaluation_SPEED_chart.png")
    print("グラフ 'final_evaluation_SPEED_chart.png' を保存しました。")

except ImportError:
    print("警告: 'matplotlib' がインストールされていません。グラフは生成されませんでした。")
except Exception as e:
    print(f"グラフ描画中にエラーが発生しました: {e}")