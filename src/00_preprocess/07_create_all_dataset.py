# -----------------------------------------------------------------
# テスト，検証，学習用データセットを構築するプログラム
# -----------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import numpy as np 

# --- 入力ファイル（元のファイル） ---
# ご自身のパスに合わせてください．
ORIGINAL_TEST_FILE = "./test_sample.csv" 
ORIGINAL_VAL_FILE = "./validation_sample.csv" 
CUSTOM_TRAIN_FILE = "training_pool_backgrounds.csv"

# --- 出力ファイル（新しく作成されるファイル） ---
# ご自身のパスに合わせてください．
NEW_TEST_OUT = "test_set_500.csv"
NEW_VAL_OUT = "validation_set_500.csv"
NEW_SAMPLE_SIZE = 500

VAL_REMAINING_OUT = "validation_set_remaining_9500.csv" 
TRAIN_RANDOM_OUT = "train_random_3k.csv"
TRAIN_RANDOM_SIZE = 3155 # same size of training data
TRAIN_COMBINED_OUT = "train_combined_6k.csv"

# 乱数シード（常に同じ結果を出すため）
RANDOM_SEED = 42

print("--- 全データセットの再構築を開始します ---")

try:
    # --- ステップ1：新しい「テストセット（500件）」の作成 ---
    print(f"\n--- ステップ1: '{ORIGINAL_TEST_FILE}' を読み込み中... ---")
    df_test_orig = pd.read_csv(ORIGINAL_TEST_FILE)
    # (最終評価で basemodel_translation も使うため、dropnaの対象に含める)
    df_test_orig = df_test_orig.dropna(subset=['japanese', 'english', 'basemodel_translation'])
    df_test_new = df_test_orig.sample(n=NEW_SAMPLE_SIZE, random_state=RANDOM_SEED)
    df_test_new.to_csv(NEW_TEST_OUT, index=False, encoding='utf-8-sig')
    print(f"'{NEW_TEST_OUT}' ({len(df_test_new)}件) を保存しました。")

    # --- ステップ2：新しい「検証セット（500件）」と「モデル2学習データ」の分離 ---
    print(f"\n--- ステップ2: '{ORIGINAL_VAL_FILE}' を読み込み中... ---")
    df_val_orig = pd.read_csv(ORIGINAL_VAL_FILE)
    df_val_orig = df_val_orig.dropna(subset=['japanese', 'english'])

    # 1万件から、まず「新しい検証セット（500件）」を抽出
    df_val_new, df_val_remaining = train_test_split(
        df_val_orig,
        test_size=(len(df_val_orig) - NEW_SAMPLE_SIZE), 
        random_state=RANDOM_SEED
    )
    df_val_new.to_csv(NEW_VAL_OUT, index=False, encoding='utf-8-sig')
    print(f"'{NEW_VAL_OUT}' ({len(df_val_new)}件) を保存しました。")
    
    # 残り（元の検証セットが1万件の場合、9500件）を保存
    df_val_remaining.to_csv(VAL_REMAINING_OUT, index=False, encoding='utf-8-sig')
    print(f"残りの検証データ '{VAL_REMAINING_OUT}' ({len(df_val_remaining)}件) を保存しました。")

    # --- ステップ3：モデル2（ランダム3k）の学習データ作成 ---
    print(f"\n--- ステップ3: '{VAL_REMAINING_OUT}' からモデル2の学習データを作成します ---")
    df_train_rand = df_val_remaining.sample(n=TRAIN_RANDOM_SIZE, random_state=RANDOM_SEED)
    df_train_rand.to_csv(TRAIN_RANDOM_OUT, index=False, encoding='utf-8-sig')
    print(f"モデル2学習用 '{TRAIN_RANDOM_OUT}' ({len(df_train_rand)}件) を保存しました。")

    # --- ステップ4：モデル3（混合6k）の学習データ作成 (★あなたの修正ロジック★) ---
    print(f"\n--- ステップ4: モデル3（高品質 + ランダム）の学習データを作成します ---")
    
    # モデル1のデータ（高品質データ）をロード
    df_custom = pd.read_csv(CUSTOM_TRAIN_FILE)
    #  教師データとして「専門家の訳（⑫）」を選択 ★★★
    df_custom_for_combine = df_custom[['japanese', 'background_consider_english']]
    df_custom_for_combine = df_custom_for_combine.rename(
        columns={'background_consider_english': 'target_translation'}
    )
    print(f"高品質データ（専門家の訳） {len(df_custom_for_combine)} 件をロードしました。")

    # モデル2のデータ（ランダム3k）をロード
    # 教師データとして「元の訳（②）」を選択 ★★★
    df_train_rand_for_combine = df_train_rand[['japanese', 'english']]
    df_train_rand_for_combine = df_train_rand_for_combine.rename(
        columns={'english': 'target_translation'}
    )
    print(f"ランダムデータ（元の訳） {len(df_train_rand_for_combine)} 件をロードしました。")

    # データを結合し、シャッフル
    df_combined = pd.concat([df_custom_for_combine, df_train_rand_for_combine], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=RANDOM_SEED) 

    df_combined.to_csv(TRAIN_COMBINED_OUT, index=False, encoding='utf-8-sig')
    print(f"モデル3学習用 '{TRAIN_COMBINED_OUT}' ({len(df_combined)}件) を保存しました。")
    
    print("\n--- 全てのデータセットの準備が完了しました ---")

except FileNotFoundError as e:
    print(f"エラー: ファイルが見つかりません。 {e}")
    print("CSVファイルがこのスクリプトと同じフォルダにあるか確認してください。")
except KeyError as e:
    print(f"エラー: 必要な列が見つかりません。 {e}")
    print("【最重要】CSVを編集する際、必要な列名（japanese, english, basemodel_translation, background_consider_english）を削除/変更していないか確認してください。")
except Exception as e:
    print(f"エラーが発生しました: {e}")