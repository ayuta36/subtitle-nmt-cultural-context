# -----------------------------------------------------------------
# ファイルに対し，bleuとbertを計算し，列を付与するプログラム
# -----------------------------------------------------------------

import pandas as pd
import torch
import os
from bert_score import score as bert_score_calc
import sacrebleu
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- ご自身のパスに合わせてください．

# 1. 入力するCSVファイルのパス
input_csv_path = 'opensubtitles_ja_en_all_basemodel_translation_cleaned.csv'

# 2. 評価スコアを追記して保存する新しい出力ファイルパス
output_eval_path = 'data_with_all_scores.csv'

# 3. 評価に使用する列名
reference_column = 'english'           # 正解訳 (人間による翻訳)
hypothesis_column = 'basemodel_translation' # 仮説訳 (機械による翻訳)

# 4. 一度に処理する行数（チャンクサイズ）
chunk_size = 10000

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
        # ヘッダー行を除いた実質的な行数をカウント
        rows_to_skip = sum(1 for line in open(output_eval_path, encoding='utf-8')) - 1
        if rows_to_skip > 0:
            print(f"再開処理を検出: {rows_to_skip} 行が処理済みです。続きから開始します。")
    except Exception as e:
        print(f"出力ファイルの読み込みエラー: {e}")
        rows_to_skip = 0

is_first_chunk = (rows_to_skip <= 0)

# --- メイン処理 ---
try:
    print("評価スコアの計算を開始します...")
    with pd.read_csv(
        input_csv_path,
        chunksize=chunk_size,
        encoding='utf-8',
        skiprows=range(1, rows_to_skip + 1) # 処理済みの行をスキップ
    ) as reader:

        for i, chunk in enumerate(reader, start=(rows_to_skip // chunk_size)):
            print(f"--- チャンク {i+1} の処理を開始 ---")

            # 評価に必要なテキストをリストとして抽出 (NaNなどを防ぐため文字列に変換)
            hypotheses = chunk[hypothesis_column].astype(str).tolist()
            references = chunk[reference_column].astype(str).tolist()

            # --- 1. BERTScoreの計算 ---
            # P, R, F1スコアが返るのでF1を使用
            _, _, f1 = bert_score_calc(hypotheses, references, model_type=bert_model_type, lang="en", device=device, verbose=False)
            chunk['bert_score_mt'] = f1.tolist()

            # --- 2. BLEUスコアの計算 (文単位) ---
            # sacrebleuでは参照訳をリストのリストとして渡す: [reference]
            bleu_scores = [sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(hypotheses, references)]
            chunk['bleu_score_mt'] = bleu_scores

            # CSVに追記保存
            chunk.to_csv(
                output_eval_path, mode='a', header=is_first_chunk,
                index=False, encoding='utf-8'
            )
            if is_first_chunk:
                is_first_chunk = False

            print(f"--- チャンク {i+1} の処理完了 ---")

except Exception as e:
    print(f"処理中にエラーが発生しました: {e}")

print(f"\n全ての評価スコア計算が完了しました。")

