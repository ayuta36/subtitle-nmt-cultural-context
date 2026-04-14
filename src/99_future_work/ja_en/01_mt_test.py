# -----------------------------------------------------------------
# 入力された原文に対し，複数モデルの翻訳を出力しスコアを計算するプログラム
# -----------------------------------------------------------------

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
import time
import pandas as pd
import warnings

# 警告を非表示にして、ターミナル出力をクリーンにします
warnings.filterwarnings("ignore")

# --- 1. モデルパスの設定 (ご自身のパスに合わせてください．)---
MODEL_PATHS = {
    'baseline': 'Helsinki-NLP/opus-mt-ja-en',
    'model1_custom': './model_1_custom_final',
    'model2_random': './model_2_random_final',
    'model3_combined': './model_3_combined_final'
}

# --- 2. NMTモデルの読み込み ---
@torch.no_grad() # 勾配計算を全体でオフにするデコレータ
def load_nmt_models():
    print("--- 1. NMTモデルを読み込んでいます... (初回起動時) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    models = {}
    tokenizers = {}
    
    start_load_time = time.time()
    for name, path in MODEL_PATHS.items():
        print(f"  > {name} ({path}) を読み込み中...")
        tokenizers[name] = AutoTokenizer.from_pretrained(path)
        models[name] = AutoModelForSeq2SeqLM.from_pretrained(path).to(device)
        models[name].eval() # 評価モード
    
    end_load_time = time.time()
    print(f"--- NMTモデル読み込み完了 (所要時間: {end_load_time - start_load_time:.2f}秒) ---\n")
    return models, tokenizers, device

# --- 3. 評価指標（Metrics）モデルの読み込み ---
def load_metrics():
    print("--- 2. 評価指標（Metrics）モデルを読み込んでいます... (初回起動時) ---")
    metrics = {}
    
    # BLEU
    print("  > SacreBLEU を読み込み中...")
    metrics["sacrebleu"] = evaluate.load("sacrebleu")
    
    # BERTScore
    print("  > BERTScore を読み込み中...")
    metrics["bertscore"] = evaluate.load("bertscore") # "bert-score" から "bertscore" に修正
    
    # COMET (evaluateライブラリ経由で読み込む)
    print("  > COMET を読み込み中... (初回はダウンロードに時間がかかります)")
    try:
        metrics["comet"] = evaluate.load("comet") 
    except ImportError:
        print("\n[!] エラー: COMET を使用するには unbabel-comet が必要です。")
        print("ターミナルで pip install unbabel-comet を実行してください。")
        exit()
    
    print("--- 評価指標モデル読み込み完了 ---\n")
    return metrics

# --- 4. 1文の翻訳と全スコア計算 ---
@torch.no_grad()
def evaluate_single_instance(model, tokenizer, ja_text, en_ref, metrics, device):
    # スコア計算用のデータ形式
    source_list = [ja_text]
    reference_list = [en_ref]

    # 1. 推論と時間測定
    start_time = time.time()
    inputs = tokenizer(ja_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    generated_ids = model.generate(**inputs)
    translation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    end_time = time.time()
    
    inference_time = end_time - start_time
    prediction_list = [translation] # スコア計算用のリスト

    # 2. スコア計算
    
    # SacreBLEU
    bleu_score = metrics["sacrebleu"].compute(
        predictions=prediction_list, 
        references=reference_list
    )['score']
    
    # BERTScore (en-en)
    # refとpredictionが英語同士なので、標準のenモデルで計算
    bertscore_en_en = metrics["bertscore"].compute(
        predictions=prediction_list, 
        references=reference_list, 
        lang="en"
    )['f1'][0] # F1スコアの0番目（最初の文）を取得
    
    # BERTScore (ja-en)
    # prediction(en)とsource(ja)を比較するため、多言語モデルを指定
    bertscore_ja_en = metrics["bertscore"].compute(
        predictions=prediction_list, 
        references=source_list, # 比較対象を原文(ja)にする
        model_type="bert-base-multilingual-cased"
    )['f1'][0]
    
    comet_score = metrics["comet"].compute(
        sources=source_list, 
        predictions=prediction_list, 
        references=reference_list
    )['mean_score'] # 'mean_score' (平均) を取得

    # 3. 結果を辞書にまとめる
    return {
        "translation": translation,
        "inference_time_sec": inference_time,
        "comet_score": comet_score,
        "bertscore_f1_en_en": bertscore_en_en,
        "bertscore_f1_ja_en": bertscore_ja_en,
        "sacrebleu_score": bleu_score
    }

# --- 5. メインの実行ループ ---
if __name__ == "__main__":
    
    # 最初に全モデルを1回だけ読み込む
    models, tokenizers, device = load_nmt_models()
    metrics = load_metrics()
    
    print("="*50)
    print(" NMT 4モデル・インタラクティブ・スコア分析ツール")
    print(" 翻訳したい日本語と、その「正解」英語を入力してください。")
    print(" 終了したい場合は、入力時に 'exit' と入力してください。")
    print("="*50)

    try:
        while True:
            # 1. 日本語原文の入力
            ja_input = input("\n[?] 翻訳したい日本語原文 (例: おかえりなさい)\n  > ")
            if ja_input.lower() == 'exit':
                break
            
            # 2. 英語正解訳（リファレンス）の入力
            en_ref_input = input("\n[?] 上記の「正解」となる英語翻訳 (例: Welcome home.)\n  > ")
            if en_ref_input.lower() == 'exit':
                break
            
            # どちらかが空ならやり直し
            if not ja_input or not en_ref_input:
                print("[!] エラー: 日本語原文と英語正解訳の両方が必要です。")
                continue

            print("\n--- 4モデルで翻訳とスコア計算を実行中... ---")
            
            all_results = []
            # 4モデルを順番に実行
            for name, model in models.items():
                result = evaluate_single_instance(
                    model, 
                    tokenizers[name], 
                    ja_input, 
                    en_ref_input, 
                    metrics, 
                    device
                )
                result["model"] = name # モデル名を結果に追加
                all_results.append(result)

            # 3. 結果をPandasのDataFrameで綺麗に表示
            df_report = pd.DataFrame(all_results)
            df_report = df_report.set_index("model") # model列を行名にする
            
            # 論文の[表1][表2]と同じ順番に並び替える
            report_order = ['baseline', 'model1_custom', 'model2_random', 'model3_combined']
            df_report = df_report.reindex(report_order)
            
            print("\n--- 分析レポート ---")
            # ターミナル幅に合わせて、全列を表示する設定
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            
            # 小数点以下の桁数を調整
            print(df_report.to_string(
                float_format="%.6f", # 小数点以下6桁
                columns=["translation", "inference_time_sec", "comet_score", "bertscore_f1_en_en", "bertscore_f1_ja_en", "sacrebleu_score"]
            ))
            print("="*50)

    except KeyboardInterrupt:
        print("\n[!] ツールを終了")
    except Exception as e:
        print(f"\n[!] 予期せぬエラーが発生しました: {e}")