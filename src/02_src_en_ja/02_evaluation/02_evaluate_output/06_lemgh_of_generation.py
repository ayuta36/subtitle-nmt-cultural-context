# -----------------------------------------------------------------
# 生成長率を計算するプログラム
# -----------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

# --- 1. 設定: ファイルパスを環境に合わせて変更してください ---

FILE_PATHS = {
    'baseline': 'baseline_output.txt',
    'model1': 'model1_output.txt',
    'model2': 'model2_output.txt',
    'model3': 'model3_output.txt'
}

# 原文(英語)のファイルパス
SOURCE_FILE_PATH = './test_source.en' 

# グラフの保存先
OUTPUT_IMAGE_FILE = 'translation_char_distribution_en_ja.png'

# --- 2. 関数定義 ---

def load_lines(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"!!! エラー: {filepath} の読み込み失敗: {e}", file=sys.stderr)
        return None

# --- 3. データの読み込み ---

print("--- 英日翻訳データの読み込み開始 ---")
data = {}
valid_paths = {k: v for k, v in FILE_PATHS.items() if os.path.exists(v)}

if not valid_paths:
    print("!!! 有効なファイルがありません。", file=sys.stderr)
    sys.exit(1)

file_line_count = -1
for model_name, path in valid_paths.items():
    lines = load_lines(path)
    if lines:
        data[model_name] = lines
        if file_line_count == -1: file_line_count = len(lines)
        elif file_line_count != len(lines):
            print(f"!!! 行数不一致エラー: {model_name}", file=sys.stderr)
            sys.exit(1)
        print(f"OK: '{model_name}' ({len(lines)}行)")

df = pd.DataFrame(data)

# 原文(英語)の読み込み
if SOURCE_FILE_PATH and os.path.exists(SOURCE_FILE_PATH):
    source_lines = load_lines(SOURCE_FILE_PATH)
    if source_lines and len(source_lines) == len(df):
        df['source'] = source_lines
        # 英語原文は「単語数」で計測
        df['source_word_count'] = df['source'].apply(lambda s: len(str(s).split()))
        print(f"OK: 原文(英語)を読み込み完了")

# --- 4. 翻訳長の計算 (日本語：文字数) ---

print("--- 翻訳長の計算 (日本語文字数) を実行中 ---")
for model_name in valid_paths.keys():
    col_name = f'{model_name}_char_count'
    # 日本語訳文は「文字数」で計測
    df[col_name] = df[model_name].apply(lambda s: len(str(s)))

# --- 5. 記述統計の表示 ---

print("\n--- 記述統計 (モデル別・日本語文字数) ---")
stats_cols = [f'{name}_char_count' for name in valid_paths.keys()]
stats_df = df[stats_cols].describe()
print(stats_df.to_markdown(floatfmt=".2f"))

# --- 6. 生成長比率の計算 (日本語文字数 / 英語単語数) ---

if 'source_word_count' in df:
    print("\n--- 生成長比率 (日本語文字数 / 英語単語数) の平均 ---")
    for model_name in valid_paths.keys():
        ratio_col = f'{model_name}_ratio'
        count_col = f'{model_name}_char_count'
        df[ratio_col] = df[count_col] / (df['source_word_count'] + 1e-6)
        print(f"- {model_name}: {df[ratio_col].mean():.4f}")

# --- 7. 可視化 (Box Plot) ---

print(f"\n--- グラフ生成中: {OUTPUT_IMAGE_FILE} ---")
plot_data = df[stats_cols].melt(var_name='Model', value_name='Character Count (Output)')
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 8))

ax = sns.boxplot(data=plot_data, x='Model', y='Character Count (Output)', palette='muted')
ax.set_title('Distribution of Japanese Translation Lengths (Character Count)', fontsize=16)
ax.set_xlabel('Model Type', fontsize=12)
ax.set_ylabel('Sentence Length (Characters)', fontsize=12)
ax.set_xticklabels([n.replace('_char_count', '') for n in stats_cols])

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_FILE)
print("--- 完了 ---")