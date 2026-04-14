# -----------------------------------------------------------------
# 入力されたファイルのbleu,bertのヒストグラムを描画するプログラム
# -----------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

# 1. ヒストグラムを作成するための入力ファイルパス
# ご自身のパスに合わせてください．
input_csv_path = 'data_with_all_scores_completed.csv' 

# 2. 出力するグラフのファイル名
output_plot_path = 'final_scores_histogram.png'

# --- 設定ここまで ---

try:
    print(f"データファイル ({input_csv_path}) を読み込んでいます...")
    df = pd.read_csv(input_csv_path)
    print("読み込み完了。")

    # --- グラフの準備 (3つのグラフを縦に並べる) ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    plt.style.use('seaborn-v0_8-whitegrid')
    fig.suptitle('All Scores about basemodel', fontsize=18, y=1.02)

    # --- 1. BLEUスコア (機械翻訳 vs 正解訳) ---
    bleu_col = 'bleu_score_mt'
    mean_bleu = df[df[bleu_col] > 0][bleu_col].mean() # 0を除外した平均値
    axes[0].hist(df[bleu_col], bins=100, range=(0,100), alpha=0.7, color='skyblue')
    axes[0].axvline(mean_bleu, color='red', linestyle='--', linewidth=2, label=f'平均値 (0を除く): {mean_bleu:.2f}')
    axes[0].set_title('BLEUScore (En(original) vs En(basemodel))', fontsize=14)
    axes[0].set_xlabel('BLEU Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].legend()

    # --- 2. BERTScore (機械翻訳 vs 正解訳) ---
    bert_mt_col = 'bert_score_mt'
    mean_bert_mt = df[bert_mt_col].mean()
    axes[1].hist(df[bert_mt_col], bins=50, alpha=0.7, color='salmon')
    axes[1].axvline(mean_bert_mt, color='red', linestyle='--', linewidth=2, label=f'ave: {mean_bert_mt:.4f}')
    axes[1].set_title('BERTScore (En(original) vs En(basemodel)', fontsize=14)
    axes[1].set_xlabel('BERTScore (F1)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].legend()

    # --- 3. BERTScore (原文 vs 機械翻訳) ---
    bert_src_col = 'bert_mt_en_ja'
    mean_bert_src = df[bert_src_col].mean()
    axes[2].hist(df[bert_src_col], bins=50, alpha=0.7, color='lightgreen')
    axes[2].axvline(mean_bert_src, color='red', linestyle='--', linewidth=2, label=f'ave: {mean_bert_src:.4f}')
    axes[2].set_title('BERTScore (Ja(original) vs En(basemodel)', fontsize=14)
    axes[2].set_xlabel('BERTScore (F1)', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].legend()

    # グラフのレイアウトを調整
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # グラフを画像ファイルとして保存
    plt.savefig(output_plot_path)

    print(f"\nヒストグラムの作成が完了しました。✨")
    print(f"グラフは {output_plot_path} として保存されました。")

except FileNotFoundError:
    print(f"エラー: 入力ファイルが見つかりません。パスを確認してください: {input_csv_path}")
except Exception as e:
    print(f"エラーが発生しました: {e}")