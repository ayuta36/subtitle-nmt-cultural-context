# -----------------------------------------------------------------
# 生成長率を計算するプログラム
# -----------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os # ファイル保存場所の確認用

# --- 1. 設定 ご自身のパスに合わせてください．---

# 各モデルの「出力テキストファイル」のパス
# (テストセット全体を翻訳させた .txt / .en ファイル)
FILE_PATHS = {
    'baseline': 'baseline_output.txt',  # ベースラインモデルの訳文
    'model1': 'model1_output.txt',      # 高品質データ (Model 1) の訳文
    'model2': 'model2_output.txt',      # ランダムデータ (Model 2) の訳文
    'model3': 'model3_output.txt'       # 混合データ (Model 3) の訳文
}

# (比較のため、原文(日本語)のファイルパスも指定
SOURCE_FILE_PATH = './test_source.ja' # 原文(日本語)

# グラフの保存先ファイル名
OUTPUT_IMAGE_FILE = 'translation_length_distribution.png'

# --- 2. ファイル読み込み関数 ---

def load_lines(filepath):
    """テキストファイルを1行ずつ読み込み、リストとして返す"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # 前後の空白(改行コードなど)を削除
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"!!! エラー: ファイルが見つかりません: {filepath}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"!!! エラー: ファイル読み込み中に問題が発生しました ({filepath}): {e}", file=sys.stderr)
        return None

# --- 3. データの読み込みとDataFrameの構築 ---

print("--- データの読み込み開始 ---")
data = {}
file_line_count = -1

# ファイルパスの存在チェック
valid_paths = {}
for model_name, path in FILE_PATHS.items():
    if not os.path.exists(path):
        print(f"!!! 警告: '{model_name}' のファイルが見つかりません: {path} (スキップします)", file=sys.stderr)
    else:
        valid_paths[model_name] = path

if not valid_paths:
    print("!!! 致命的エラー: 有効なファイルパスが一つもありません。FILE_PATHSを確認してください。", file=sys.stderr)
    sys.exit(1)

# 有効なファイルのみ読み込み
for model_name, path in valid_paths.items():
    lines = load_lines(path)
    if lines:
        data[model_name] = lines
        print(f"OK: '{model_name}' を読み込み完了 ({len(lines)}行)")
        
        # 行数の一貫性をチェック
        if file_line_count == -1:
            file_line_count = len(lines)
        elif file_line_count != len(lines):
            print(f"!!! 致命的エラー: '{model_name}' の行数({len(lines)})が、他のファイルの行数({file_line_count})と一致しません。", file=sys.stderr)
            print("!!! 処理を中断します。すべてのファイルが同じテストセットに対する出力か確認してください。", file=sys.stderr)
            sys.exit(1)

# DataFrameを作成
df = pd.DataFrame(data)

# 原文(日本語)の読み込みと長さの計算
if SOURCE_FILE_PATH and os.path.exists(SOURCE_FILE_PATH):
    source_lines = load_lines(SOURCE_FILE_PATH)
    if source_lines and len(source_lines) == len(df):
        df['source'] = source_lines
        # 日本語は「文字数」で長さを測る
        df['source_char_count'] = df['source'].apply(len)
        print(f"OK: 'source' を読み込み完了 ({len(source_lines)}行)")
    elif source_lines:
        print(f"!!! 警告: 'source'の行数({len(source_lines)})が他のファイル({len(df)})と一致しません。比率計算をスキップします。", file=sys.stderr)
elif SOURCE_FILE_PATH:
     print(f"!!! 警告: 'source' ファイルが見つかりません: {SOURCE_FILE_PATH} (スキップします)", file=sys.stderr)


print(f"\n--- データ読み込み完了: {len(df)}件 ---")


# --- 4. 翻訳長の計算 (単語数) ---

print("--- 翻訳長の計算 (単語数) を実行中 ---")
# 英語の訳文は「スペース区切りの単語数」で長さを測る
for model_name in valid_paths.keys():
    col_name = f'{model_name}_word_count'
    df[col_name] = df[model_name].apply(lambda s: len(s.split()))

# --- 5. 記述統計の表示 (コンソール出力) ---

print("\n--- 記述統計 (モデル別・単語数) ---")
stats_cols = [f'{name}_word_count' for name in valid_paths.keys()]
if not stats_cols:
    print("!!! エラー: 計算対象の統計データがありません。", file=sys.stderr)
    sys.exit(1)
    
# .describe() で主要な統計量(平均, 標準偏差, 四分位数など)を計算
stats_df = df[stats_cols].describe()

# 見やすいように小数点以下2桁に丸めて表示
print("各モデルの出力単語数に関する記述統計:")
print(stats_df.to_markdown(floatfmt=".2f"))


# --- 6. 生成長比率の計算 ---
if 'source_char_count' in df:
    print("\n--- 生成長比率 (出力単語数 / 入力文字数) の平均 ---")
    ratio_summary = {}
    for model_name in valid_paths.keys():
        ratio_col = f'{model_name}_ratio'
        count_col = f'{model_name}_word_count'
        # ゼロ除算を避けるため、分母が0の場合は 1e-6 を加算
        df[ratio_col] = df[count_col] / (df['source_char_count'] + 1e-6)
        ratio_summary[model_name] = df[ratio_col].mean()
    
    print("モデル別 平均生成長比率:")
    for model_name, avg_ratio in ratio_summary.items():
        print(f"- {model_name}: {avg_ratio:.4f}")

# --- 7. 翻訳長の分布を可視化 (Box Plot) ---

print(f"\n--- 翻訳長の分布を '{OUTPUT_IMAGE_FILE}' にグラフ化中... ---")

# データをSeabornが扱いやすい「ロングフォーマット」に変換
plot_data = df[stats_cols].melt(var_name='Model', value_name='Word Count (Output)')

# グラフのスタイル設定
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 8)) # グラフサイズを大きめに設定
ax = sns.boxplot(
    data=plot_data,
    x='Model',
    y='Word Count (Output)',
    palette='pastel' # 色合い
)

ax.set_title('Distribution of Translation Lengths (Word Count) by Model', fontsize=18, pad=20)
ax.set_xlabel('Model Type', fontsize=14)
ax.set_ylabel('Sentence Length (Words)', fontsize=14)
# X軸のラベル名から "_word_count" を除去して見やすくする
model_labels = [name.replace('_word_count', '') for name in stats_cols]
ax.set_xticklabels(model_labels, rotation=15) # X軸のラベルを少し回転

# グラフをファイルとして保存
try:
    plt.tight_layout() # レイアウトを自動調整
    plt.savefig(OUTPUT_IMAGE_FILE)
    print(f"\n--- 完了: グラフをカレントディレクトリに '{OUTPUT_IMAGE_FILE}' として保存しました。 ---")
except Exception as e:
    print(f"\n!!! エラー: グラフの保存に失敗しました: {e}", file=sys.stderr)