# -----------------------------------------------------------------
# APIにより翻訳文を自動生成するプログラム(なお，試行錯誤の段階)
# -----------------------------------------------------------------

import pandas as pd
import google.generativeai as genai
from tqdm import tqdm
import time
import json
import os

# ==============================
# 設定 (最高品質モード)
# ==============================
input_file = 'train_model1_final_cleaned.csv'       # アップロード済みのファイル
output_file = 'train_model1_with_expert_data.csv'   # 生成されるファイル

# ここにAPIキーを入れる
GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)

# 最高品質かつ最新のモデルを指定
model_name = "models/gemini-2.0-pro-exp-02-05" 

print(f"使用モデル: {model_name}")

# ==============================
# モデル準備
# ==============================
generation_config = {
    "temperature": 0.7,
    "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
    model_name=model_name,
    generation_config=generation_config
)

def create_prompt(source, target, baseline):
    return f"""
    あなたは、ハリウッド映画の字幕翻訳における最高権威です。以下のタスクを最高の品質で実行してください。

    # タスク
    与えられた「英語の原文（Source）」と、人間による既存の日本語訳（参照訳）、そして性能の低い機械翻訳モデルが出力した日本語訳（ベースライン訳）を分析してください。
    その上で、以下の2つの項目をJSON形式で生成してください。

    1. **background_info**: なぜこの文の翻訳が難しいのか。文化的背景、隠れたニュアンス、話者の意図、英語特有のイディオムなど、直訳では伝わらない要素について、必要な知識だけを簡潔に（1文で）説明してください。
    2. **expert_translation**: 上記の背景知識を考慮し、字幕翻訳の特性（時間的・空間的制約）を踏まえ、最も自然で「刺さる」日本語の字幕を生成してください。

    # 制約条件
    * **文字数制限**: 日本語の文字数は**全角4文字〜26文字以内**に収めてください。
    * **情報の圧縮**: 原文の約43%の情報が失われることを前提に、不必要な情報を大胆に省略してください。
    * **主語の削除**: 文脈上不要な主語（私は、あなたは）は削除してください。
    * **役割語**: 文脈から話者のキャラを推測し、適切な語尾（～だぜ、～わ、～じゃ、～だよ）を選択してください。

    # 入力データ
    [英語原文]: {source}
    [人間による参照訳]: {target}
    [ベースライン訳]: {baseline}
    
    # Output Schema
    {{
        "background_info": "string",
        "expert_translation": "string"
    }}
    """

# ==============================
# 実行ループ
# ==============================
# データ読み込み
if not os.path.exists(input_file):
    raise FileNotFoundError(f"エラー: {input_file} が見つかりません。同じフォルダに置いてください。")

df = pd.read_csv(input_file)

# 途中再開ロジック
if os.path.exists(output_file):
    try:
        df_existing = pd.read_csv(output_file, on_bad_lines='skip')
        processed_count = len(df_existing)
        print(f"★ 途中データが見つかりました。{processed_count} 件目から再開します。")
        df_to_process = df.iloc[processed_count:].copy()
    except:
        print("読み込みエラー。新規作成します。")
        df_to_process = df.copy()
        pd.DataFrame(columns=['english', 'japanese', 'baseline_ja', 'background_info', 'expert_ja']).to_csv(output_file, index=False)
else:
    print("新規作成します。")
    df_to_process = df.copy()
    pd.DataFrame(columns=['english', 'japanese', 'baseline_ja', 'background_info', 'expert_ja']).to_csv(output_file, index=False)

print(f"残り処理件数: {len(df_to_process)} 件")

# ループ実行
for index, row in tqdm(df_to_process.iterrows(), total=len(df_to_process)):
    source = row['english']
    target = row['japanese']
    baseline = row['baseline_ja']

    prompt = create_prompt(source, target, baseline)
    
    bg_info = "ERROR"
    exp_trans = "ERROR"
    
    try:
        response = model.generate_content(prompt)
        result = json.loads(response.text)
        
        bg_info = result.get("background_info", "")
        exp_trans = result.get("expert_translation", "")
        time.sleep(30)

    except Exception as e:
        print(f"\nError at index {index}: {e}")
        # エラー時は長めに待機
        time.sleep(120)

    # 1行ずつ保存
    new_row = pd.DataFrame([{
        'english': source,
        'japanese': target,
        'baseline_ja': baseline,
        'background_info': bg_info,
        'expert_ja': exp_trans
    }])
    
    new_row.to_csv(output_file, mode='a', header=False, index=False)

print("完了！")