# -----------------------------------------------------------------
# 入力されたファイルに対し，同言語間・異言語間のbertを算出し，列を付与するプログラム
# -----------------------------------------------------------------

import pandas as pd
from bert_score import score
import torch
from tqdm.auto import tqdm
import os

# ==============================
# 設定（ご自身のパスに合わせてください．）
# ==============================
input_file = 'training_pool_with_baseline.csv'   # 入力ファイル
output_file = 'training_pool_with_scores.csv'    # 出力ファイル

# 使用するモデル
bert_model = "bert-base-multilingual-cased"

# チャンク設定
chunk_size = 100 
batch_size = 32  

# ==============================
# 1. データの読み込み
# ==============================
print(f"データを読み込んでいます: {input_file}")
# low_memory=Falseを指定して型推論の警告を抑制
df = pd.read_csv(input_file, low_memory=False) 
total_len = len(df)

# ==============================
# 2. 再開位置の確認
# ==============================
start_idx = 0

if os.path.exists(output_file):
    try:
        # 既に処理済みの行数を確認
        processed_df = pd.read_csv(output_file, low_memory=False)
        processed_len = len(processed_df)
        
        if processed_len < total_len:
            start_idx = processed_len
            print(f"★ 途中データが見つかりました。 {start_idx} 行目から再開します。")
        elif processed_len >= total_len:
            print("★ すべての処理が完了しています。")
            start_idx = total_len
            
    except Exception as e:
        print(f"読み込みエラー: {e}")
        print("最初からやり直します。")
else:
    print("新規作成します。")
    # ヘッダー作成
    output_columns = df.columns.tolist() + [
        'bert_score_original_en_vs_original_ja',
        'bert_score_original_en_vs_basemodel_ja'
    ]
    pd.DataFrame(columns=output_columns).to_csv(output_file, index=False, encoding='utf-8')

# ==============================
# 3. 計算実行 (続きから)
# ==============================
if start_idx < total_len:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"計算を開始します... (Device: {device}, Start Index: {start_idx})")

    pbar = tqdm(total=total_len, initial=start_idx, unit="lines")

    for current_idx in range(start_idx, total_len, chunk_size):
        end_idx = min(current_idx + chunk_size, total_len)
        
        # --- データの切り出し ---
        chunk_df = df.iloc[current_idx:end_idx].copy()
        chunk_df['english'] = chunk_df['english'].fillna("").astype(str)
        chunk_df['japanese'] = chunk_df['japanese'].fillna("").astype(str)
        chunk_df['baseline_ja'] = chunk_df['baseline_ja'].fillna("").astype(str)
        
        # リスト化
        refs_en = chunk_df['english'].tolist()
        cands_human = chunk_df['japanese'].tolist()
        cands_ai = chunk_df['baseline_ja'].tolist()
        
        # --- スコア1: 元の英語 vs 元の日本語 ---
        try:
            _, _, F1_human = score(
                cands_human, refs_en, 
                model_type=bert_model, 
                batch_size=batch_size, 
                device=device, 
                verbose=False 
            )
            
            # --- スコア2: 元の英語 vs 機械の日本語 ---
            _, _, F1_ai = score(
                cands_ai, refs_en, 
                model_type=bert_model, 
                batch_size=batch_size, 
                device=device, 
                verbose=False
            )
            
            # --- データフレームに追加 ---
            chunk_df['bert_score_original_en_vs_original_ja'] = F1_human.numpy()
            chunk_df['bert_score_original_en_vs_basemodel_ja'] = F1_ai.numpy()
            
            # --- CSVに追記保存 ---
            chunk_df.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8')
            
        except Exception as e:
            # 万が一特定のバッチでエラーが出ても、止まらずにスキップしてログを出す
            print(f"\nError at index {current_idx}: {e}")
            # エラー時は0埋めで保存して進行を止めない
            chunk_df['bert_score_original_en_vs_original_ja'] = 0.0
            chunk_df['bert_score_original_en_vs_basemodel_ja'] = 0.0
            chunk_df.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8')

        # 進捗更新
        pbar.update(len(chunk_df))

    pbar.close()
    print(f"\n全処理完了！結果を保存しました: {output_file}")

else:
    print("処理すべきデータはありません。")