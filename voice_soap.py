import streamlit as st
import google.generativeai as genai
from typing import Dict, Any
import json

# ページ設定
st.set_page_config(
    page_title="音声カルテ作成（SOAP形式）",
    page_icon="🎤",
    layout="wide"
)

# タイトル
st.title("🎤 音声カルテ作成（SOAP形式）")
st.markdown("診療会話の音声ファイルから自動でSOAP形式の診療記録を作成します")

# APIキーの確認と設定
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
except KeyError:
    st.error("⚠️ Google API Keyが設定されていません。`secrets.toml`に`GOOGLE_API_KEY`を設定してください。")
    st.stop()
except Exception as e:
    st.error(f"⚠️ API設定エラー: {str(e)}")
    st.stop()

# Geminiモデルの初期化
@st.cache_resource
def get_model():
    return genai.GenerativeModel('gemini-2.5-flash')

model = get_model()

# SOAP要約プロンプト
SOAP_PROMPT_TEMPLATE = """あなたは在宅医療の医師です。
以下の診療会話の文字起こしテキストを読み、SOAP形式で診療記録を作成してください。

【SOAP形式】
S (Subjective): 患者・家族の主訴、訴え
O (Objective): 身体所見、バイタルサイン、検査結果
A (Assessment): 診断、評価
P (Plan): 治療計画、処方、今後の方針

会話には医師・患者・家族の発言が混在している可能性があります。
診療記録として必要な情報を抽出し、簡潔に整理してください。

## 出力形式
以下のJSON形式で出力してください。Markdownのコードブロックで囲わず、純粋なJSONのみを返してください。

{
  "subjective": "患者・家族の主訴、訴えをここに記載",
  "objective": "身体所見、バイタルサイン、検査結果をここに記載",
  "assessment": "診断、評価をここに記載",
  "plan": "治療計画、処方、今後の方針をここに記載"
}

[文字起こしテキスト]
{transcribed_text}
"""

def transcribe_audio(audio_file) -> str:
    """音声ファイルを文字起こし"""
    try:
        # 音声ファイルをアップロード
        audio_file.seek(0)
        uploaded_audio = genai.upload_file(audio_file)

        # 文字起こしを実行
        prompt = "この音声ファイルを日本語で文字起こししてください。会話の内容を正確に記録してください。"
        response = model.generate_content([prompt, uploaded_audio])

        return response.text
    except Exception as e:
        st.error(f"音声文字起こしエラー: {str(e)}")
        return None

def create_soap_from_text(transcribed_text: str) -> Dict[str, Any]:
    """文字起こしテキストからSOAP形式を作成"""
    try:
        prompt = SOAP_PROMPT_TEMPLATE.format(transcribed_text=transcribed_text)
        response = model.generate_content(prompt)

        # レスポンステキストからJSONを抽出
        result_text = response.text.strip()

        # Markdown記法のコードブロックを削除
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        elif result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]

        result_text = result_text.strip()

        # JSONパース
        soap_data = json.loads(result_text)
        return soap_data
    except json.JSONDecodeError as e:
        st.error(f"JSON解析エラー: {str(e)}\n\n取得したテキスト:\n{response.text}")
        return None
    except Exception as e:
        st.error(f"SOAP作成エラー: {str(e)}")
        return None

def display_soap_results(soap_data: Dict[str, Any], transcribed_text: str):
    """SOAP結果を表示"""
    if soap_data is None or transcribed_text is None:
        return

    st.subheader("📋 診療記録")

    # タブで表示を切り替え
    tab1, tab2, tab3 = st.tabs(["📋 SOAP表示", "📝 文字起こしテキスト", "📄 コピー用テキスト"])

    with tab1:
        st.markdown("### 📝 SOAP形式診療記録")

        # S (Subjective)
        if soap_data.get("subjective"):
            st.markdown("**■ S (Subjective - 主訴・患者の訴え)**")
            st.markdown(f"> {soap_data['subjective']}")
            st.write("")

        # O (Objective)
        if soap_data.get("objective"):
            st.markdown("**■ O (Objective - 客観的所見)**")
            st.markdown(f"> {soap_data['objective']}")
            st.write("")

        # A (Assessment)
        if soap_data.get("assessment"):
            st.markdown("**■ A (Assessment - 評価)**")
            st.markdown(f"> {soap_data['assessment']}")
            st.write("")

        # P (Plan)
        if soap_data.get("plan"):
            st.markdown("**■ P (Plan - 計画)**")
            st.markdown(f"> {soap_data['plan']}")
            st.write("")

    with tab2:
        st.markdown("### 📝 文字起こし結果（原文）")
        st.text_area("文字起こしテキスト", value=transcribed_text, height=500, disabled=True)

    with tab3:
        st.markdown("### 📄 コピー用テキスト")

        # テキスト形式で整形
        text_output = []
        text_output.append("=" * 60)
        text_output.append("【SOAP形式診療記録】")
        text_output.append("=" * 60)
        text_output.append("")

        if soap_data.get("subjective"):
            text_output.append("■ S (Subjective - 主訴・患者の訴え)")
            text_output.append(soap_data["subjective"])
            text_output.append("")

        if soap_data.get("objective"):
            text_output.append("■ O (Objective - 客観的所見)")
            text_output.append(soap_data["objective"])
            text_output.append("")

        if soap_data.get("assessment"):
            text_output.append("■ A (Assessment - 評価)")
            text_output.append(soap_data["assessment"])
            text_output.append("")

        if soap_data.get("plan"):
            text_output.append("■ P (Plan - 計画)")
            text_output.append(soap_data["plan"])
            text_output.append("")

        text_output.append("=" * 60)
        text_output.append("【文字起こし原文】")
        text_output.append("=" * 60)
        text_output.append(transcribed_text)

        full_text = "\n".join(text_output)
        st.text_area("コピー可能なテキスト", value=full_text, height=600)

        # JSON形式でも表示（開発者向け）
        with st.expander("🔧 JSON形式で表示（開発者向け）"):
            st.json(soap_data)

# メインコンテンツ
st.markdown("### 🎙️ 診療音声ファイルをアップロード")
st.info("💡 音声ファイルから自動で文字起こしを行い、SOAP形式の診療記録を作成します")

uploaded_audio = st.file_uploader(
    "音声ファイルを選択してください",
    type=["mp3", "wav", "m4a", "ogg"],
    help="診療会話を録音した音声ファイルをアップロードしてください"
)

if uploaded_audio is not None:
    # ファイル情報を表示
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📁 アップロード情報")
        st.write(f"**ファイル名:** {uploaded_audio.name}")
        st.write(f"**ファイルサイズ:** {uploaded_audio.size / 1024:.2f} KB")
        st.write(f"**ファイル形式:** {uploaded_audio.type}")

        # 音声プレーヤー
        st.audio(uploaded_audio)

    with col2:
        if st.button("🎤 文字起こし & SOAP作成", type="primary", use_container_width=True):
            # 文字起こし
            with st.spinner("🎙️ AIが音声を文字起こし中..."):
                transcribed_text = transcribe_audio(uploaded_audio)

            if transcribed_text:
                st.success("✅ 文字起こし完了")

                # SOAP作成
                with st.spinner("📋 SOAP形式の診療記録を作成中..."):
                    soap_data = create_soap_from_text(transcribed_text)

                if soap_data:
                    st.success("✅ SOAP形式の診療記録作成完了")
                    display_soap_results(soap_data, transcribed_text)

# 使い方
with st.expander("📖 使い方"):
    st.markdown("""
    ### 使用方法

    1. **音声ファイルをアップロード**
       - 対応形式: MP3, WAV, M4A, OGG
       - 診療会話を録音した音声ファイルを選択してください

    2. **文字起こし & SOAP作成ボタンをクリック**
       - AIが自動で音声を文字起こしします
       - 文字起こしテキストからSOAP形式の診療記録を作成します

    3. **結果を確認**
       - **SOAP表示タブ**: 整形されたSOAP形式の診療記録
       - **文字起こしテキストタブ**: 音声の文字起こし原文
       - **コピー用テキストタブ**: 電子カルテにコピペできる形式

    ### SOAP形式とは

    - **S (Subjective)**: 患者・家族の主訴、訴え
    - **O (Objective)**: 身体所見、バイタルサイン、検査結果
    - **A (Assessment)**: 診断、評価
    - **P (Plan)**: 治療計画、処方、今後の方針

    ### 注意事項

    - 音声ファイルのサイズが大きい場合、処理に時間がかかることがあります
    - 音質が悪い場合、文字起こしの精度が低下する可能性があります
    - 個人情報を含む音声ファイルの取り扱いには十分注意してください
    """)

# フッター
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <small>医療ハッカソン デモアプリ | 音声からSOAP形式診療記録を自動作成 | Powered by Google Gemini 2.5 Flash</small>
    </div>
    """,
    unsafe_allow_html=True
)
