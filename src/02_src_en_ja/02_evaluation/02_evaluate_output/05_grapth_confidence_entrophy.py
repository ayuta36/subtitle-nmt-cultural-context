# -----------------------------------------------------------------
# 確信度とエントロピーのグラフを描画するプログラム
# -----------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os


# --- 1. 設定: 前ステップにて生成されたCSVファイルのパス ---
# ご自身のパスに合わせてください．

CSV_FILES_TO_LOAD = {
    'baseline': 'baseline_confidence_analysis.csv',
    'model1': 'model1(custom)_second_confidence_analysis.csv',
    'model2': 'model2(random)_second_confidence_analysis.csv',
    'model3': 'model3(combined)_second_confidence_analysis.csv'
}

# --- 2. データの読み込みと結合 ---

print("--- CSVファイルの読み込みと結合を開始 ---")
all_data_list = []

for model_name, path in CSV_FILES_TO_LOAD.items():
    if not os.path.exists(path):
        print(f"!!! 警告: ファイルが見つかりません: {path} (スキップします)", file=sys.stderr)
        continue
    
    try:
        df = pd.read_csv(path)
        # どのモデルのデータか分かるように 'model' 列を追加
        df['model'] = model_name
        all_data_list.append(df)
        print(f"OK: '{path}' を読み込み完了 ({len(df)}行)")
    except Exception as e:
        print(f"!!! エラー: '{path}' の読み込みに失敗しました: {e}", file=sys.stderr)

if not all_data_list:
    print("!!! 致命的エラー: 読み込めるデータがありません。ファイルパスを確認してください。", file=sys.stderr)
    sys.exit(1)

# 全モデルのDataFrameを一つに結合
combined_df = pd.concat(all_data_list, ignore_index=True)

print("\n--- データの結合完了 ---")
print("--- グラフの生成を開始 ---")

# --- 3. グラフのスタイルとカラーパレットの設定 ---
sns.set_theme(style="whitegrid")

# モデルと色のマッピングを辞書で定義
my_palette = {
    'baseline': '#95a5a6', # グレー
    'model1':   '#3498db', # 青
    'model2':   '#2ecc71', # 緑
    'model3':   '#e67e22'  # オレンジ
}

# --- 4. グラフA: 平均エントロピー (迷いの度合い) ---
plt.figure(figsize=(12, 8))
ax1 = sns.boxplot(
    data=combined_df,
    x='model',
    y='avg_entropy',
    palette=my_palette, # 統一パレット
    order=['baseline', 'model1', 'model2', 'model3']
)

ax1.set_title('Distribution of Sentence-Level Entropy by Model (Lower is Better)', fontsize=18, pad=20)
ax1.set_xlabel('Model Type', fontsize=14)
ax1.set_ylabel('Average Entropy (Decoder "Maze")', fontsize=14)
plt.xticks(rotation=15)

# グラフをファイルとして保存
save_path_entropy = 'step2_entropy_distribution.png'
try:
    plt.tight_layout()
    plt.savefig(save_path_entropy)
    print(f"OK: エントロピーのグラフを '{save_path_entropy}' に保存しました。")
except Exception as e:
    print(f"!!! エラー: グラフの保存に失敗しました: {e}", file=sys.stderr)

# --- 5. グラフB: 平均Top-1確率 (確信度) の可視化 ---

plt.figure(figsize=(12, 8))
ax2 = sns.boxplot(
    data=combined_df,
    x='model',
    y='avg_top1_prob',
    palette=my_palette,
    order=['baseline', 'model1', 'model2', 'model3'] # 表示順を固定
)
ax2.set_title('Distribution of Sentence-Level Top-1 Probability by Model (Higher is Better)', fontsize=18, pad=20)
ax2.set_xlabel('Model Type', fontsize=14)
ax2.set_ylabel('Average Top-1 Probability (Decoder "Confidence")', fontsize=14)
plt.xticks(rotation=15)

# グラフをファイルとして保存
save_path_prob = 'step2_top1_prob_distribution.png'
try:
    plt.tight_layout()
    plt.savefig(save_path_prob)
    print(f"OK: Top-1確率のグラフを '{save_path_prob}' に保存しました。")
except Exception as e:
    print(f"!!! エラー: グラフの保存に失敗しました: {e}", file=sys.stderr)

print("\n--- すべての処理が完了しました ---")