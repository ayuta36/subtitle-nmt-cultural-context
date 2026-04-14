# -----------------------------------------------------------------
# テストセットにおける元の字幕ペアを保存するプログラム
# -----------------------------------------------------------------

import pandas as pd
import sys
import os

# --- 1. 設定: ご自身のファイルパスに編集してください ---

# 元になるCSVファイルのパス
CSV_FILE_PATH = './test_set_sample.csv'

# CSVファイル内の「原文（日本語）」の列名
JAPANESE_COLUMN_NAME = 'japanese'

# CSVファイル内の「参照訳（英語）」の列名
ENGLISH_COLUMN_NAME = 'english'

# --- 出力ファイル名  ---
OUTPUT_SOURCE_FILE = './test_source.ja'      # 推論スクリプトが読み込む原文ファイル
OUTPUT_REFERENCE_FILE = './test_reference.en'  # 評価に使う参照訳ファイル

# --- 2. CSVの読み込み ---

print(f"--- CSVファイル '{CSV_FILE_PATH}' を読み込んでいます... ---")
if not os.path.exists(CSV_FILE_PATH):
    print(f"!!! エラー: ファイルが見つかりません: {CSV_FILE_PATH}", file=sys.stderr)
    sys.exit(1)

try:
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"OK: {len(df)}行 読み込み完了。")
except Exception as e:
    print(f"!!! エラー: CSVファイルの読み込みに失敗しました: {e}", file=sys.stderr)
    sys.exit(1)

# --- 3. 列の存在チェック ---

if JAPANESE_COLUMN_NAME not in df.columns:
    print(f"!!! エラー: CSV内に '{JAPANESE_COLUMN_NAME}' という列が見つかりません。", file=sys.stderr)
    print(f"利用可能な列: {df.columns.tolist()}", file=sys.stderr)
    sys.exit(1)
    
if ENGLISH_COLUMN_NAME not in df.columns:
    print(f"!!! エラー: CSV内に '{ENGLISH_COLUMN_NAME}' という列が見つかりません。", file=sys.stderr)
    print(f"利用可能な列: {df.columns.tolist()}", file=sys.stderr)
    sys.exit(1)

# --- 4. テキストファイルへの書き出し ---

try:
    # (A) 日本語（原文）を .ja ファイルに書き出し
    print(f"書き出し中 -> {OUTPUT_SOURCE_FILE}")
    with open(OUTPUT_SOURCE_FILE, 'w', encoding='utf-8') as f_ja:
        for line in df[JAPANESE_COLUMN_NAME]:
            # 改行コードや前後の空白を除去して書き込む
            f_ja.write(str(line).strip() + '\n')
            
    # (B) 英語（参照訳）を .en ファイルに書き出し
    print(f"書き出し中 -> {OUTPUT_REFERENCE_FILE}")
    with open(OUTPUT_REFERENCE_FILE, 'w', encoding='utf-8') as f_en:
        for line in df[ENGLISH_COLUMN_NAME]:
            f_en.write(str(line).strip() + '\n')

    print("\n--- 完了 ---")
    print(f"原文ファイル '{OUTPUT_SOURCE_FILE}' が正常に作成されました。")
    print(f"参照訳ファイル '{OUTPUT_REFERENCE_FILE}' が正常に作成されました。")

except Exception as e:
    print(f"!!! エラー: ファイルの書き出し中にエラーが発生しました: {e}", file=sys.stderr)
    sys.exit(1)