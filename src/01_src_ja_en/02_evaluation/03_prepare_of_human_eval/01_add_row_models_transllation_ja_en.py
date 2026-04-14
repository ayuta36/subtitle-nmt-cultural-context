# -----------------------------------------------------------------
# テストセットに対し全モデルの翻訳列を付与するプログラム
# -----------------------------------------------------------------

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

# --- 1. 設定 ---
# ご自身のパスに合わせてください．
TEST_SET_PATH = './test_sample.csv'
OUTPUT_CSV_PATH = 'qualitative_comparison_all_models.csv'
BATCH_SIZE = 8 # 速度が必要な場合は 8 や 16 に設定

# ご自身のパスに合わせてください．
MODEL_PATHS = {
    'baseline': 'Helsinki-NLP/opus-mt-ja-en',
    'model1_custom': './model_1_custom_final', # 
    'model2_random': './model_2_random_final', # 
    'model3_combined': './model_3_combined_final' # 
}

# --- 2. モデルとトークナイザーの読み込み ---
print("モデルとトークナイザーを読み込んでいます...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {}
tokenizers = {}

for name, path in MODEL_PATHS.items():
    print(f"{name} を読み込み中...")
    tokenizers[name] = AutoTokenizer.from_pretrained(path)
    models[name] = AutoModelForSeq2SeqLM.from_pretrained(path).to(device)
    models[name].eval() # 評価モードに設定

print("モデルの読み込み完了。")

# --- 3. データセットの読み込み ---
df_test = pd.read_csv(TEST_SET_PATH)
source_texts = df_test['japanese'].tolist() # 'japanese' 列（または原文の列名）
reference_texts = df_test['english'].tolist() # 'english' 列（または正解の列名）

# --- 4. 翻訳の生成 ---
results = []
start_time = time.time()
print(f"{len(source_texts)}件の翻訳を開始します...")

# torch.no_grad() で勾配計算を無効化し、速度とメモリ効率を向上
with torch.no_grad():
    for i in range(0, len(source_texts), BATCH_SIZE):
        batch_texts = source_texts[i:i+BATCH_SIZE]
        batch_refs = reference_texts[i:i+BATCH_SIZE]
        
        translations = {}
        for name, model in models.items():
            tokenizer = tokenizers[name]
            
            # バッチ処理
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            generated_ids = model.generate(**inputs)
            batch_translations = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            translations[name] = batch_translations

        # 1行ずつ結果を保存
        for j in range(len(batch_texts)):
            row = {
                'id': i + j,
                'source_japanese': batch_texts[j],
                'reference_english': batch_refs[j],
                'translation_baseline': translations['baseline'][j],
                'translation_model1': translations['model1_custom'][j],
                'translation_model2': translations['model2_random'][j],
                'translation_model3': translations['model3_combined'][j]
            }
            results.append(row)
            
        if (i + BATCH_SIZE) % 100 < BATCH_SIZE:
             print(f"--- {i + BATCH_SIZE}件 / {len(source_texts)}件 完了 ---")

end_time = time.time()
print(f"全翻訳が完了しました。 (所要時間: {end_time - start_time:.2f}秒)")

# --- 5. CSVファイルとして保存 ---
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
