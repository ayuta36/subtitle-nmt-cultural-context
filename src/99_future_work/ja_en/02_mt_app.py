# -----------------------------------------------------------------
# 小規模な翻訳アプリを開発するプログラム(JA-EN)
# -----------------------------------------------------------------

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

# --- 1. モデルパスの設定 ---
# 環境に合わせてパスを調整してください
MODEL_PATHS = {
    'baseline': 'Helsinki-NLP/opus-mt-ja-en',
    'model1_custom': './model_1_custom_final',
    'model2_random': './model_2_random_final',
    'model3_combined': './model_3_combined_final'
}

# --- 2. モデルの読み込み ---
# @st.cache_resource は、モデルを（アプリ起動時に）1回だけ読み込むためのStreamlit
@st.cache_resource
def load_all_models():
    print("--- モデルとトークナイザーの読み込みを開始します ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    models = {}
    tokenizers = {}
    
    start_load_time = time.time()
    for name, path in MODEL_PATHS.items():
        print(f"{name} ({path}) を読み込み中...")
        tokenizers[name] = AutoTokenizer.from_pretrained(path)
        models[name] = AutoModelForSeq2SeqLM.from_pretrained(path).to(device)
        models[name].eval() # 評価モード
    
    end_load_time = time.time()
    print(f"--- 全モデル読み込み完了 (所要時間: {end_load_time - start_load_time:.2f}秒) ---")
    return models, tokenizers, device

# --- 3. 翻訳の実行 ---
def translate_text(input_text, models, tokenizers, device):
    translations = {}
    with torch.no_grad(): # 勾配計算をオフにして高速化
        for name, model in models.items():
            tokenizer = tokenizers[name]
            
            # 翻訳実行
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            generated_ids = model.generate(**inputs)
            translation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            translations[name] = translation
    return translations

# --- 4. Streamlit アプリのUI ---

# ページ設定
st.set_page_config(page_title="NMT 4モデル翻訳比較", layout="wide")

st.title("🎓 NMT 4モデル 簡易翻訳比較ツール")
st.write("日本語を入力すると、4つのモデルの翻訳結果を同時に比較できます。")

# モデルの読み込み（初回のみ実行される）
with st.spinner("研究用の4モデルを読み込んでいます... (初回起動時は数分かかります)"):
    models, tokenizers, device = load_all_models()
st.success("モデルの読み込みが完了しました。")

st.divider()

# 翻訳の入力
default_text = "よろしくお願いします"
input_text = st.text_area("翻訳したい日本語を入力してください:", default_text, height=100)

# 翻訳実行ボタン
if st.button("翻訳実行", type="primary"):
    if input_text:
        with st.spinner("4モデルで翻訳を実行中..."):
            start_trans_time = time.time()
            translations = translate_text(input_text, models, tokenizers, device)
            end_trans_time = time.time()
        
        st.success(f"翻訳完了！（所要時間: {end_trans_time - start_trans_time:.2f}秒）")
        
        # 結果を4カラムで表示
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.subheader("1. Baseline (Helsinki)")
            st.markdown(f"**{translations['baseline']}**")
            
        with col2:
            st.subheader("2. Model 1 (Custom)")
            st.markdown(f"**{translations['model1_custom']}**")
            
        with col3:
            st.subheader("3. Model 2 (Random)")
            st.markdown(f"**{translations['model2_random']}**")

        with col4:
            st.subheader("4. Model 3 (Combined)")
            st.markdown(f"**{translations['model3_combined']}**")
    else:
        st.warning("日本語を入力してください。")

st.divider()
st.caption("MT_trial")
