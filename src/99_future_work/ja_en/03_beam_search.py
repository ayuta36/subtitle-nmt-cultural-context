# -----------------------------------------------------------------
# 入力された原文に対し，beamsearchより探索を可視化させるプログラム
# -----------------------------------------------------------------

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import graphviz
import os
import numpy as np

# ---------------------------------------------------------
# 全モデル一括 可視化スクリプト
# ---------------------------------------------------------

class BeamSearchVisualizer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self, model_path, model_label, input_text, num_beams=5):
        print(f"--- Processing {model_label} ---")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
            model.eval()
        except Exception as e:
            print(f"Error loading {model_label}: {e}")
            return

        inputs = tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=30,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True
            )

        self._draw_tree(outputs, tokenizer, model, model_label, num_beams)

    def _draw_tree(self, outputs, tokenizer, model, label, num_beams):
        dot = graphviz.Digraph(comment=label)
        dot.attr(rankdir='LR')
        dot.attr(dpi='300')
        dot.attr(label=f"\n{label}", labelloc='t', fontsize='20') # 図のタイトル
        
        dot.node("START", "START", shape='doublecircle', style='filled', fillcolor='lightgrey')

        sequences = outputs.sequences.cpu().numpy()
        scores = outputs.sequences_scores.cpu().numpy()

        # 正規化して線の太さを決める
        min_score = scores.min()
        max_score = scores.max()
        
        for beam_idx, seq in enumerate(sequences):
            # スコアに基づく線の太さ計算
            raw_score = scores[beam_idx]
            if max_score == min_score:
                width = 2.0
            else:
                width = 1.0 + 3.0 * (raw_score - min_score) / (max_score - min_score)
            
            parent_id = "START"
            
            for t_idx, token_id in enumerate(seq):
                if token_id in [tokenizer.pad_token_id, model.config.decoder_start_token_id]:
                    continue
                
                token_str = tokenizer.decode([token_id])
                node_id = f"b{beam_idx}_t{t_idx}_{token_id}" # パスを分離
                
                # 色設定
                if token_id == tokenizer.eos_token_id:
                    node_label = "EOS"
                    fill = "mistyrose" if "Random" in label else "lightcyan"
                    shape = "doubleoctagon"
                else:
                    node_label = token_str
                    fill = "white"
                    shape = "ellipse"

                dot.node(node_id, node_label, fillcolor=fill, style='filled', shape=shape)
                dot.edge(parent_id, node_id, penwidth=str(width), color="grey")
                
                parent_id = node_id
                
                if token_id == tokenizer.eos_token_id:
                    break
        
        # ファイル保存 (スペース等はアンダースコアに置換)
        filename = f"Tree_{label.replace(' ', '_').replace('(', '').replace(')', '')}"
        dot.render(filename, format='png', cleanup=True)
        print(f"Saved: {filename}.png")

# ---------------------------------------------------------
# 実行設定 (ここを書き換えてください)
# ---------------------------------------------------------
if __name__ == "__main__":
    
    # 比較用の入力文
    target_text = "おい あぶねぇだろ キツネ！",

    # モデルの辞書: {"表示名": "パス"}
    # パスはご自身の環境に合わせて修正してください
    models = {
        "Baseline": "Helsinki-NLP/opus-mt-ja-en",
        "Model 1 (High Quality)": "model_1_custom_final",
        "Model 2 (Random)":       "model_2_random_final",
        "Model 3 (Combined)":     "model_3_combined_final"
    }

    viz = BeamSearchVisualizer()

    for label, path in models.items():
        viz.run(path, label, target_text)

    print("\n全モデルの画像生成が完了しました。")
