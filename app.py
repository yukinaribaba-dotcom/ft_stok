import streamlit as st
import google.generativeai as genai
from PIL import Image
import json
import pandas as pd
from typing import Dict, Any

# ページ設定
st.set_page_config(
    page_title="医療紹介状情報抽出アプリ",
    page_icon="🏥",
    layout="wide"
)

# タイトル
st.title("🏥 医療紹介状情報抽出アプリ")
st.markdown("紹介状の画像またはテキストから患者情報を自動抽出します")

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

# プロンプトテンプレート
EXTRACTION_PROMPT = """あなたは医療情報の構造化スペシャリストです。
提供された「診療情報提供書（紹介状）」の内容を読み取り、在宅医療（訪問診療）のカルテシステムに取り込むためのJSONデータを作成してください。

## 制約事項
- 出力はJSON形式のみとしてください。Markdownのコードブロックで囲わず、純粋なJSONのみを返してください。
- 医師が短時間で患者背景を把握できるよう、情報は可能な限り構造化してください。
- 値が存在しない場合は null または "不明" としてください。

## 抽出ルールとJSON構造
以下の構造に合わせて情報を抽出・整理してください。

1. **patient_profile**: 基本情報だけでなく、介護保険情報（要介護度など）も含めてください。
2. **clinical_info**:
   - chief_complaint: 主訴
   - history_of_present_illness: 現病歴（今回の紹介に至る経緯・直近の経過）
   - past_medical_history: 既往歴（過去の病気）。配列でリスト化してください。
   - medication_summary: 薬剤情報のサマリー。特に抗凝固薬やインスリンなどのハイリスク薬があれば明記してください。
   - allergies: アレルギー情報。「なし」か「不明」かを区別してください。
3. **adl_status**: 【重要】在宅医療において最も重要な項目です。意識レベル、移動能力、食事形態、排泄状況を詳細に抽出してください。
4. **social_context**: 【重要】キーパーソン（連絡先・続柄）、同居家族、利用中の介護サービス内容を抽出してください。

## 出力すべきJSONテンプレート
{
  "patient_profile": {
    "name": "",
    "birth_date": "",
    "age": null,
    "gender": "",
    "care_level": ""
  },
  "clinical_info": {
    "chief_complaint": "",
    "history_of_present_illness": "",
    "past_medical_history": [],
    "medication_summary": "",
    "allergies": ""
  },
  "adl_status": {
    "consciousness": "",
    "mobility": "",
    "feeding": "",
    "excretion": ""
  },
  "social_context": {
    "key_person": {
      "name": "",
      "relation": "",
      "contact": "",
      "living_status": ""
    },
    "services_used": [],
    "acp_preference": ""
  }
}
"""

def extract_info_from_image(image: Image.Image) -> Dict[str, Any]:
    """画像から情報を抽出"""
    try:
        response = model.generate_content([EXTRACTION_PROMPT, image])
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
        extracted_data = json.loads(result_text)
        return extracted_data
    except json.JSONDecodeError as e:
        st.error(f"JSON解析エラー: {str(e)}\n\n取得したテキスト:\n{response.text}")
        return None
    except Exception as e:
        st.error(f"画像処理エラー: {str(e)}")
        return None

def extract_info_from_text(text: str) -> Dict[str, Any]:
    """テキストから情報を抽出"""
    try:
        prompt = EXTRACTION_PROMPT + f"\n\n入力テキスト:\n{text}"
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
        extracted_data = json.loads(result_text)
        return extracted_data
    except json.JSONDecodeError as e:
        st.error(f"JSON解析エラー: {str(e)}\n\n取得したテキスト:\n{response.text}")
        return None
    except Exception as e:
        st.error(f"テキスト処理エラー: {str(e)}")
        return None

def display_results(data: Dict[str, Any]):
    """抽出結果を表示"""
    if data is None:
        return

    st.subheader("📋 抽出結果")

    # JSON形式で表示
    st.json(data)

    # セクション別に構造化して表示
    st.subheader("📊 構造化データ")

    # 患者基本情報
    st.markdown("### 👤 患者基本情報")
    if "patient_profile" in data:
        profile = data["patient_profile"]
        profile_df = pd.DataFrame([
            {"項目": "氏名", "内容": profile.get("name", "不明")},
            {"項目": "生年月日", "内容": profile.get("birth_date", "不明")},
            {"項目": "年齢", "内容": str(profile.get("age", "不明")) if profile.get("age") else "不明"},
            {"項目": "性別", "内容": profile.get("gender", "不明")},
            {"項目": "要介護度", "内容": profile.get("care_level", "不明")}
        ])
        st.dataframe(profile_df, use_container_width=True, hide_index=True)

    # 臨床情報
    st.markdown("### 🏥 臨床情報")
    if "clinical_info" in data:
        clinical = data["clinical_info"]
        clinical_df = pd.DataFrame([
            {"項目": "主訴", "内容": clinical.get("chief_complaint", "不明")},
            {"項目": "現病歴", "内容": clinical.get("history_of_present_illness", "不明")},
            {"項目": "既往歴", "内容": ", ".join(clinical.get("past_medical_history", [])) if clinical.get("past_medical_history") else "不明"},
            {"項目": "服薬サマリー", "内容": clinical.get("medication_summary", "不明")},
            {"項目": "アレルギー", "内容": clinical.get("allergies", "不明")}
        ])
        st.dataframe(clinical_df, use_container_width=True, hide_index=True)

    # ADL状況
    st.markdown("### 🚶 ADL状況（日常生活動作）")
    if "adl_status" in data:
        adl = data["adl_status"]
        adl_df = pd.DataFrame([
            {"項目": "意識レベル", "内容": adl.get("consciousness", "不明")},
            {"項目": "移動能力", "内容": adl.get("mobility", "不明")},
            {"項目": "食事形態", "内容": adl.get("feeding", "不明")},
            {"項目": "排泄状況", "内容": adl.get("excretion", "不明")}
        ])
        st.dataframe(adl_df, use_container_width=True, hide_index=True)

    # 社会的背景
    st.markdown("### 👨‍👩‍👧‍👦 社会的背景・支援体制")
    if "social_context" in data:
        social = data["social_context"]

        # キーパーソン情報
        if "key_person" in social:
            key_person = social["key_person"]
            key_person_df = pd.DataFrame([
                {"項目": "キーパーソン氏名", "内容": key_person.get("name", "不明")},
                {"項目": "続柄", "内容": key_person.get("relation", "不明")},
                {"項目": "連絡先", "内容": key_person.get("contact", "不明")},
                {"項目": "同居状況", "内容": key_person.get("living_status", "不明")}
            ])
            st.dataframe(key_person_df, use_container_width=True, hide_index=True)

        # 介護サービス・ACP
        social_df = pd.DataFrame([
            {"項目": "利用中サービス", "内容": ", ".join(social.get("services_used", [])) if social.get("services_used") else "不明"},
            {"項目": "ACP/患者意向", "内容": social.get("acp_preference", "不明")}
        ])
        st.dataframe(social_df, use_container_width=True, hide_index=True)

# メインコンテンツ
tab1, tab2 = st.tabs(["📷 画像アップロード", "📝 テキスト入力"])

with tab1:
    st.markdown("### スマートフォンで撮影した紹介状の写真をアップロード")
    uploaded_file = st.file_uploader(
        "画像ファイルを選択してください",
        type=["jpg", "jpeg", "png"],
        help="紹介状の写真をアップロードしてください"
    )

    if uploaded_file is not None:
        # 2カラムレイアウト
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📄 入力画像")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        with col2:
            if st.button("🔍 情報を抽出", key="extract_image", type="primary"):
                with st.spinner("AIが情報を抽出中..."):
                    extracted_data = extract_info_from_image(image)
                    if extracted_data:
                        display_results(extracted_data)

with tab2:
    st.markdown("### 電子カルテからコピーしたテキストを貼り付け")
    text_input = st.text_area(
        "紹介状のテキストを入力してください",
        height=300,
        placeholder="例:\n患者名: 山田太郎\n生年月日: 1950年4月15日\n主訴: 胸部不快感\n既往歴: 高血圧、糖尿病\n...",
        help="電子カルテからコピーしたテキストを貼り付けてください"
    )

    if text_input:
        # 2カラムレイアウト
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📄 入力テキスト")
            st.text_area("入力内容", value=text_input, height=300, disabled=True)

        with col2:
            if st.button("🔍 情報を抽出", key="extract_text", type="primary"):
                with st.spinner("AIが情報を抽出中..."):
                    extracted_data = extract_info_from_text(text_input)
                    if extracted_data:
                        display_results(extracted_data)

# フッター
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <small>医療ハッカソン デモアプリ | Powered by Google Gemini 1.5 Flash</small>
    </div>
    """,
    unsafe_allow_html=True
)
