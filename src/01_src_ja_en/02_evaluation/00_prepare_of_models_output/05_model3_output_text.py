# -----------------------------------------------------------------
# モデル3によるテストセットの翻訳を出力を保存するプログラム
# -----------------------------------------------------------------

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm # 進捗バー表示用
import sys
import os

# --- 1. 設定: ご自身のパスに合わせてください． ---
CONFIG = {
    # (例2: Model 3 のパス)
    "MODEL_PATH": "./model_3_combined_final", # model
    "OUTPUT_TRANSLATION_FILE": "model3_output.txt", # 出力ファイル
}

# --- 共通の設定 ---

# 翻訳したい原文（テストセット）のファイル
SOURCE_TEST_FILE = "./test_source.ja" 

# バッチサイズ 
BATCH_SIZE = 16 

# --- 2. ファイル読み込み関数 ---

def load_lines(filepath):
    """テキストファイルを1行ずつ読み込み、リストとして返す"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"!!! エラー: 原文ファイルが見つかりません: {filepath}", file=sys.stderr)
        return None

# --- 3. モデルとトークナイザーの準備 ---

print(f"--- モデル '{CONFIG['MODEL_PATH']}' の読み込み開始 ---")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {DEVICE}")

try:
    # Helsinki-NLPモデルは AutoModelForSeq2SeqLM を使う
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_PATH'])
    model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG['MODEL_PATH'])
    model.to(DEVICE)
    model.eval() # 推論モードに設定
except Exception as e:
    print(f"!!! エラー: モデルまたはトークナイザーの読み込みに失敗しました: {e}", file=sys.stderr)
    sys.exit(1)

print("--- モデル読み込み完了 ---")

# --- 4. 原文ファイルの読み込み ---

source_sentences = load_lines(SOURCE_TEST_FILE)
if not source_sentences:
    print("原文ファイルが読み込めなかったため、処理を終了します。")
    sys.exit(1)

print(f"--- 原文 {len(source_sentences)}行 の翻訳を開始します ---")
print(f"出力先: {CONFIG['OUTPUT_TRANSLATION_FILE']}")

# --- 5. 翻訳の実行とファイルへの書き込み ---

# 出力ファイルを書き込みモードで開く
try:
    with open(CONFIG['OUTPUT_TRANSLATION_FILE'], 'w', encoding='utf-8') as f_out:
        
        # tqdmで進捗を表示しながらバッチ処理
        for i in tqdm(range(0, len(source_sentences), BATCH_SIZE), desc="Translating"):
            
            # バッチ単位で原文をスライス
            batch_texts = source_sentences[i : i + BATCH_SIZE]
            
            # トークナイズ (モデルの入力形式に変換)
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(DEVICE)
            
            # 推論の実行 
            with torch.no_grad():
                generated_ids = model.generate(**inputs)
            
            # IDをテキスト（翻訳文）にデコード
            decoded_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # バッチの結果をファイルに書き込む (1行1訳文)
            for line in decoded_texts:
                f_out.write(line.strip() + '\n')

except Exception as e:
    print(f"\n!!! エラー: 翻訳処理中またはファイル書き込み中にエラーが発生しました: {e}", file=sys.stderr)
    if os.path.exists(CONFIG['OUTPUT_TRANSLATION_FILE']):
        print("不完全な出力ファイルが生成された可能性があります。")
    sys.exit(1)

print("\n--- 翻訳処理がすべて完了しました ---")
print(f"結果は '{CONFIG['OUTPUT_TRANSLATION_FILE']}' に保存されました。")