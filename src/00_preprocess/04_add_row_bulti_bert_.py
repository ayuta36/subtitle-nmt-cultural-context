# -----------------------------------------------------------------
# ファイルに対し，異言語間のbertを計算し，列を付与するプログラム
# -----------------------------------------------------------------

import pandas as pd
import torch
import os
from bert_score import score as bert_score_calc

# --- ユーザー設定項目 ---

# 1. 入力するCSVファイルのパス (bleu_score, bert_score_mtが既に追加済みのファイル)
input_csv_path = 'data_with_all_scores.csv' 

# 2. 最終的な結果を保存する新しい出力ファイルパス
output_eval_path = 'data_with_all_scores_completed.csv'

# 3. 評価に使用する列名
source_column = 'japanese'             # 原文
hypothesis_column = 'basemodel_translation' # 仮説訳 (機械による翻訳)

# 4. 一度に処理する行数（チャンクサイズ）
chunk_size = 100000

# --- 設定ここまで ---

# デバイスの準備
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用するデバイス: {device}")

# BERTScoreで使用するモデルタイプ
bert_model_type = "bert-base-multilingual-cased"

# 再開機能のための処理済み行数カウント
rows_to_skip = 0
if os.path.exists(output_eval_path):
    try:
        rows_to_skip = sum(1 for line in open(output_eval_path, encoding='utf-8')) - 1
        if rows_to_skip > 0:
            print(f"再開処理を検出: {rows_to_skip} 行が処理済みです。続きから開始します。")
    except Exception as e:
        print(f"出力ファイルの読み込みエラー: {e}")
        rows_to_skip = 0

is_first_chunk = (rows_to_skip <= 0)

# --- メイン処理 ---
try:
    print("3つ目の評価スコアの計算を開始します...")
    with pd.read_csv(
        input_csv_path,
        chunksize=chunk_size,
        encoding='utf-8',
        skiprows=range(1, rows_to_skip + 1)
    ) as reader:

        for i, chunk in enumerate(reader, start=(rows_to_skip // chunk_size)):
            print(f"--- チャンク {i+1} の処理を開始 ---")

            # 必要なテキストをリストとして抽出
            sources = chunk[source_column].astype(str).tolist()
            hypotheses = chunk[hypothesis_column].astype(str).tolist()

            # --- BERTScore (原文 vs 機械翻訳) の計算 ---
            _, _, f1_src = bert_score_calc(hypotheses, sources, model_type=bert_model_type, lang="en", device=device, verbose=False)
            chunk['bert_mt_en_ja'] = f1_src.tolist()

            # CSVに追記保存
            # chunkには元々の列と新しい列が含まれる
            chunk.to_csv(
                output_eval_path, mode='a', header=is_first_chunk,
                index=False, encoding='utf-8'
            )
            if is_first_chunk:
                is_first_chunk = False
            
            print(f"--- チャンク {i+1} の処理完了 ---")

except Exception as e:
    print(f"処理中にエラーが発生しました: {e}")

print(f"\n全てのスコア計算が完了しました。最終的な結果は {output_eval_path} に保存されています。")