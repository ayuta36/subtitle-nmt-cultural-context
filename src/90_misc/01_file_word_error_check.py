# -----------------------------------------------------------------
# ファイル内に文字化けがあるかチェックするプログラム
# -----------------------------------------------------------------


import pandas as pd

# --- ご自身のパスに合わせてください．---
file_path = 'opensubtitles_ja_en_all_basemodel_translation.csv'
column_to_check = 'baseline_translation'
# --- 設定ここまで ---

try:
    print(f"ファイル ({file_path}) を確認中...")
    df = pd.read_csv(file_path)

    # 確認1：空欄（nullまたは空文字列）のチェック
    empty_count = df[column_to_check].isnull().sum()
    blank_string_count = (df[column_to_check] == '').sum()
    total_empty = empty_count + blank_string_count

    if total_empty > 0:
        print(f"\n[警告] {column_to_check} 列に {total_empty} 件の空欄が見つかりました。")
    else:
        print(f"\n[OK] {column_to_check} 列に空欄はありませんでした。")

    # 確認2：文字化けの簡易チェック（豆腐"□"や""の存在を確認）
    garbled_chars = ['□', '']
    garbled_count = 0
    for char in garbled_chars:
        # 文字化けの可能性がある文字を含む行をカウント
        count = df[df[column_to_check].str.contains(char, na=False)].shape[0]
        if count > 0:
            print(f"\n[警告] 文字化けの可能性（'{char}'）が {count} 件見つかりました。")
            garbled_count += count
    
    if garbled_count == 0:
        print(f"\n[OK] 簡単な文字化けチェックでは問題は見つかりませんでした。")

    print("\n確認完了。")

except FileNotFoundError:
    print(f"エラー: ファイルが見つかりません。パスを確認してください: {file_path}")
except Exception as e:
    print(f"エラーが発生しました: {e}")