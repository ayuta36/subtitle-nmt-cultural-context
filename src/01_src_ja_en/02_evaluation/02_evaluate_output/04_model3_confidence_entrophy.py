# -----------------------------------------------------------------
# モデル3の確信度とエントロピーを計算するプログラム
# -----------------------------------------------------------------

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import pandas as pd
import sys
import os
import numpy as np

# --- 1. 設定: ご自身のパスに合わせてください．---
CONFIG = {
    # (例1: モデル3)
    "MODEL_PATH": "./model_3_combined_final",
    "MODEL_NAME": "model3(combined)",
}

# --- 共通の設定 ---

# 翻訳したい原文（テストセット）のファイル
SOURCE_TEST_FILE = "./test_source.ja" 

# バッチサイズ 
BATCH_SIZE = 1 

# --- 2. モデルとトークナイザーの準備 ---

print(f"--- モデル '{CONFIG['MODEL_PATH']}' の読み込み開始 ---")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {DEVICE}")

try:
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_PATH'])
    model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG['MODEL_PATH'])
    model.to(DEVICE)
    model.eval() # 推論モード
except Exception as e:
    print(f"!!! エラー: モデルまたはトークナイザーの読み込みに失敗しました: {e}", file=sys.stderr)
    sys.exit(1)
print("--- モデル読み込み完了 ---")

# --- 3. 原文ファイルの読み込み ---

try:
    with open(SOURCE_TEST_FILE, 'r', encoding='utf-8') as f:
        source_sentences = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"!!! エラー: 原文ファイルが見つかりません: {SOURCE_TEST_FILE}", file=sys.stderr)
    sys.exit(1)

if not source_sentences:
    print("原文ファイルが空です。処理を終了します。")
    sys.exit(1)

print(f"--- 原文 {len(source_sentences)}行 の分析を開始します (Batch Size = {BATCH_SIZE}) ---")

# --- 4. 分析の実行 ---

# 各文の「平均エントロピー」と「平均Top-1確率」を保存するリスト
results_list = []

# torch.no_grad() で勾配計算を無効にし、メモリを節約
with torch.no_grad():
    # tqdmで進捗を表示
    for i in tqdm(range(0, len(source_sentences), BATCH_SIZE), desc=f"Analyzing {CONFIG['MODEL_NAME']}"):
        
        # 1文ずつバッチ化 (BATCH_SIZE=1 が前提)
        batch_texts = source_sentences[i : i + BATCH_SIZE]
        
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(DEVICE)
        
        try:
            outputs = model.generate(
                **inputs,
                output_scores=True,             # 1. 各ステップのロジット(スコア)を出力
                return_dict_in_generate=True, # 2. 出力を辞書形式で受け取る
                max_length=128,               
                num_beams=1                     
            )
            # ===================

            # 'scores' は、生成された各トークン（ステップ）のロジット（確率になる前の値）
            # のタプル 
            all_step_logits = outputs.scores
            
            # 'sequences' は、実際に生成されたトークンIDのテンソル
            generated_ids = outputs.sequences

            # この文の全ステップのエントロピーとTop-1確率を一時保存
            sentence_entropies = []
            sentence_top1_probs = []

            # ---- 4-1. 各ステップの「迷い」を計算 ----
            # all_step_logits (タプル) の各要素をループ
            for step, step_logits in enumerate(all_step_logits):
                
                # (A) 確率分布の計算
                # ロジットをソフトマックス関数に通し、確率(0.0〜1.0)に変換
                step_probs = torch.softmax(step_logits, dim=-1)
                
                # (B) エントロピー (迷いの度合い) の計算
                # 確率が分散しているほどエントロピーは「高く」なる (迷っている)
                # 確率が集中しているほどエントロピーは「低く」なる (確信している)
                entropy = torch.distributions.Categorical(probs=step_probs).entropy().item()
                sentence_entropies.append(entropy)
                
                # (C) Top-1 確率 (確信度) の計算
                # 実際にモデルが「選んだ」トークンのIDを取得
                # `step + 1` で参照する
                chosen_token_id = generated_ids[0, step + 1].item()
                
                # 確率分布から、その「選ばれたトークン」の確率を抜き出す
                top1_prob = step_probs[0, chosen_token_id].item()
                sentence_top1_probs.append(top1_prob)

            # ---- 4-2. 文ごとの平均値を計算 ----
            if sentence_entropies: # (何も生成しなかった場合を除く)
                avg_entropy = np.mean(sentence_entropies)
                avg_top1_prob = np.mean(sentence_top1_probs)
                
                results_list.append({
                    "sentence_index": i,
                    "avg_entropy": avg_entropy,
                    "avg_top1_prob": avg_top1_prob,
                    "generated_length": len(sentence_entropies) # ステップ数
                })

        except Exception as e:
            print(f"\n!!! エラー: 文 {i} の処理中にエラーが発生しました: {e}", file=sys.stderr)
            results_list.append({
                "sentence_index": i,
                "avg_entropy": np.nan, # エラーはNaN (Not a Number)として記録
                "avg_top1_prob": np.nan,
                "generated_length": 0
            })

# --- 5. 最終結果の集計と保存 ---

if not results_list:
    print("!!! 致命的エラー: 分析結果がありません。処理を終了します。", file=sys.stderr)
    sys.exit(1)

# リストをPandas DataFrameに変換 
results_df = pd.DataFrame(results_list)

# (A) CSVファイルに詳細を保存
output_csv_path = f"./{CONFIG['MODEL_NAME']}_confidence_analysis.csv"
try:
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n--- 詳細な分析結果を '{output_csv_path}' に保存しました ---")
except Exception as e:
    print(f"\n!!! エラー: CSVファイルの保存に失敗しました: {e}", file=sys.stderr)

# (B) 最終的な平均値をコンソールに表示
final_avg_entropy = results_df['avg_entropy'].mean()
final_avg_top1_prob = results_df['avg_top1_prob'].mean()

print("\n" + "="*50)
print(f"    モデル: {CONFIG['MODEL_NAME']} の最終分析結果")
print("="*50)
print(f"  [迷いの度合い] 平均エントロピー: {final_avg_entropy:.4f}")
print(f"  [確信度]       平均Top-1確率 : {final_avg_top1_prob:.4f}")
print("="*50)
print("\n処理がすべて完了しました。")