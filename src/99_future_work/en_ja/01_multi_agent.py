# -----------------------------------------------------------------
# 小規模なマルチエージェント翻訳システムを実現するプログラム
# -----------------------------------------------------------------

import streamlit as st
import google.generativeai as genai
import time
from datetime import datetime

# ページの設定
st.set_page_config(page_title="Subtitle Multi-Agent System", layout="wide")
st.title("🏯 字幕制約・最終推敲対応：マルチエージェント翻訳")

# --- 使用モデルを Flash-Lite に統一 ---
MODEL_NAME = "models/gemini-2.5-flash-lite"

# サイドバー設定
with st.sidebar:
    st.header("設定")
    api_key = st.text_input("Gemini APIキーを入力", type="password")
    st.info(f"使用モデル: {MODEL_NAME}")
    st.markdown("""
    **字幕の制約ルール:**
    - 1行35文字以内，最大2行
    - 1秒間に12文字以内
    """)

def call_gemini(system_inst, prompt):
    if not api_key:
        st.error("APIキーを入力してください．")
        return None
    genai.configure(api_key=api_key)
    # 説明を日本語で行うよう指示を追加
    system_inst += " また，思考プロセスや説明はすべて日本語で出力してください．"
    model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_inst)
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "429" in str(e):
            st.warning("制限により30秒待機中...")
            time.sleep(30)
            return call_gemini(system_inst, prompt)
        st.error(f"エラー: {e}")
        return None

# 字幕制約チェック関数
def check_constraints(text, duration):
    lines = text.strip().split('\n')
    errors = []
    if len(lines) > 2:
        errors.append(f"行数過多（{len(lines)}行）")
    for i, line in enumerate(lines[:2]):
        if len(line) > 35:
            errors.append(f"{i+1}行目が35文字超過")
    limit = int(duration * 12)
    if len(text.replace('\n', '')) > limit:
        errors.append(f"合計が{limit}文字を超過")
    return errors

# 入力エリア
input_text = st.text_area("日本語を入力してください", height=100)
context_input = st.text_input("状況設定（例：上司への挨拶）")
duration = st.number_input("表示時間（秒）", min_value=1.0, value=3.0, step=0.5)

if st.button("翻訳と議論を開始"):
    if not input_text:
        st.warning("テキストを入力してください．")
    else:
        log_content = f"--- 翻訳実験ログ ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n"
        
        with st.status("エージェントたちが議論中...", expanded=True) as status:
            
            # 1. 検出 (分析)
            st.write("🔍 **検出エージェント**: 文化要素を特定中...")
            analysis = call_gemini("言語学者．文化的なマナーや要素を日本語で特定せよ．", f"原文: {input_text}\n状況: {context_input}")
            with st.expander("1. 分析結果"): st.write(analysis)
            log_content += f"[1. 分析]\n{analysis}\n\n"

            # 2. 文化解説
            st.write("📖 **文化エージェント**: 背景を解説中...")
            details = call_gemini("文化人類学者．精神性を日本語で詳しく解説せよ．", analysis)
            with st.expander("2. 文化解説"): st.write(details)
            log_content += f"[2. 文化解説]\n{details}\n\n"

            # 3. 翻訳初稿
            st.write("✍️ **翻訳エージェント**: 英語案を作成中...")
            drafts = call_gemini("翻訳家．ニュアンスを重視した英訳を3案出せ．", f"原文:{input_text}\n背景:{details}")
            with st.expander("3. 翻訳案"): st.write(drafts)
            log_content += f"[3. 翻訳案]\n{drafts}\n\n"

            # 4. ネイティブ批判
            st.write("🧐 **批判エージェント**: ネイティブチェック中...")
            critique = call_gemini("米国の編集者．不自然な点を日本語で指摘せよ．", drafts)
            with st.expander("4. 批判内容"): st.write(critique)
            log_content += f"[4. 批判]\n{critique}\n\n"

            # 5. 字幕制約 (ループ処理)
            st.write("🎬 **字幕制約エージェント**: 文字数制限に適合中...")
            sub_text = ""
            feedback = ""
            for i in range(3):
                sub_sys = f"字幕制作者．1行35字以内，最大2行，合計{int(duration*12)}字以内を厳守せよ．"
                sub_text = call_gemini(sub_sys, f"案:{drafts}\n批判:{critique}\n{feedback}")
                errors = check_constraints(sub_text, duration)
                if not errors: break
                feedback = f"前回の案は制約違反です：{', '.join(errors)}．さらに短くしてください．"
            with st.expander("5. 字幕制約の適用プロセス"): st.write(sub_text)
            log_content += f"[5. 字幕制約適用]\n{sub_text}\n\n"

            # 6. 最終推敲 (一つに絞る)
            st.write("🏆 **最高編集者**: 最終決定案を作成中...")
            edit_sys = "最高責任者．全ての議論を統合し，最高の最終訳を1つだけ選定・推敲せよ．選定理由も日本語で添えること．"
            final = call_gemini(edit_sys, f"背景:{details}\n最終候補:{sub_text}")
            log_content += f"[6. 最終推敲]\n{final}\n"
            
            status.update(label="完了しました！", state="complete")

        # 最終結果表示
        st.success("### 最終決定訳")
        st.markdown(f"**{final}**")

        st.download_button("ログを保存", log_content, file_name=f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")