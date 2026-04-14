# -----------------------------------------------------------------
# 入力されたファイルに対し，ベースラインの翻訳列を付与するプログラム
# -----------------------------------------------------------------

import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm
import os

# ==============================
# 設定（ご自身のパスに合わせてください．）
# ==============================
input_file = './validation_sample.csv'
output_file = './validation_sample_with_baseline.csv'
model_name = "staka/fugumt-en-ja"
batch_size = 64   # 1回に翻訳する行数（GPU用）
chunk_save_freq = 1000 # 上書き保存

# ==============================
# 1. 準備
# ==============================
print(f"データを読み込んでいます: {input_file}")
df = pd.read_csv(input_file)
source_texts = df['english'].tolist()
total_len = len(source_texts)

# デバイス設定
device = 0 if torch.cuda.is_available() else -1
print(f"モデルをロード中... (Device: {device})")
translator = pipeline("translation", model=model_name, device=device)

# ==============================
# 2. 動作確認 (最初の3行)
# ==============================
print("\n" + "="*40)
print("【動作確認】最初の3行を翻訳します...")
print("="*40)

preview_batch = source_texts[:3]
preview_results = translator(preview_batch, max_length=128, truncation=True)

for i, (src, res) in enumerate(zip(preview_batch, preview_results)):
    print(f"\n[Case {i+1}]")
    print(f"原文 (Source): {src}")
    print(f"正解 (Target): {df['japanese'].iloc[i]}")
    print(f"生成 (Baseline): {res['translation_text']}")

print("\n" + "="*40)
print("確認完了。全データの処理を開始します...")
print("="*40 + "\n")

# ==============================
# 3. 処理開始 (都度保存)
# ==============================

# 保存用ファイルの初期化（ヘッダーだけ書き込む）
# 既存の列 + 新しい 'baseline_ja' 列
header_df = pd.DataFrame(columns=df.columns.tolist() + ['baseline_ja'])
header_df.to_csv(output_file, index=False, encoding='utf-8')

results_buffer = [] # 一時保存用リスト
start_index = 0

# tqdmで進捗表示
pbar = tqdm(total=total_len, unit="lines")

for i in range(0, total_len, batch_size):
    # バッチの取得
    batch_src = source_texts[i : i + batch_size]
    
    # 翻訳実行
    try:
        translations = translator(batch_src, max_length=128, truncation=True)
        translated_texts = [res['translation_text'] for res in translations]
    except Exception as e:
        print(f"\nエラー発生 (Index {i}): {e}")
        translated_texts = ["ERROR"] * len(batch_src) # エラー時は埋める

    results_buffer.extend(translated_texts)
    pbar.update(len(batch_src))

    # 一定間隔、または最後になったら保存
    if (i // batch_size + 1) % chunk_save_freq == 0 or (i + batch_size >= total_len):
        
        # 元データの該当範囲を切り出し
        current_chunk_df = df.iloc[start_index : i + len(batch_src)].copy()
        
        # 翻訳結果を列に追加
        # (バッファの長さとChunkの長さが一致することを確認)
        current_chunk_df['baseline_ja'] = results_buffer
        
        # 追記モード('a')でCSVに書き込み、ヘッダーはなし(header=False)
        current_chunk_df.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8')
        
        # バッファとインデックスをリセット
        results_buffer = []
        start_index = i + len(batch_src)

pbar.close()
print(f"\n全処理完了！ファイル保存済み: {output_file}")