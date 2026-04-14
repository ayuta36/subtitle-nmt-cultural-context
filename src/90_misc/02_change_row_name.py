# -----------------------------------------------------------------
# ファイルの列名を変更するプログラム
# -----------------------------------------------------------------

import pandas as pd

# --- ユーザー設定項目 ---

# 1. 対象となるCSVファイルのパス
input_file_path = 'data_with_all_scores.csv' 

# 2. 列名を変更した後に保存する、新しいファイル名
output_renamed_file_path = 'data_renamed.csv'

# 3. 新しい列名の定義
# 形式: {'現在の列名': '新しい列名'}
new_column_names = {
    'japanese': 'ja',
    'english': 'en_reference',
    'bert_score': 'ja_en_bert_score',
    'baseline_translation': 'en_hypothesis',
    'bleu_score': 'bleu',
    'bert_score_mt': 'bert_score'
}

# --- 設定ここまで ---

try:
    # ファイルを読み込む
    df = pd.read_csv(input_file_path)

    # --- 現在の列名を表示 ---
    print("--- 現在の列名 ---")
    print(df.columns.tolist())
    print("--------------------")

    # --- 列名を変更 ---
    df_renamed = df.rename(columns=new_column_names)
    print("\n--- 変更後の列名 ---")
    print(df_renamed.columns.tolist())
    print("--------------------")
    
    # --- 新しいファイルとして保存 ---
    df_renamed.to_csv(output_renamed_file_path, index=False, encoding='utf-8')
    print(f"\n列名を変更したデータを '{output_renamed_file_path}' に保存しました。")


except FileNotFoundError:
    print(f"エラー: 入力ファイルが見つかりません。パスを確認してください: {input_file_path}")
except Exception as e:
    print(f"エラーが発生しました: {e}")