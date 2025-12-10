import streamlit as st
import google.generativeai as genai
from PIL import Image
import json
import pandas as pd
from typing import Dict, Any

# ページ設定
st.set_page_config(
    page_title="医療紹介状→初診カルテ変換アプリ",
    page_icon="🏥",
    layout="wide"
)

# タイトル
st.title("🏥 医療紹介状→初診カルテ変換アプリ")
st.markdown("紹介状の画像またはテキストから初診カルテ形式で患者情報を自動抽出します")

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
    return genai.GenerativeModel('gemini-2.0-pro-exp')

model = get_model()

# プロンプトテンプレート
EXTRACTION_PROMPT = """あなたは在宅医療の医師です。
提供された「診療情報提供書（紹介状）」の内容を読み取り、訪問診療開始時の初診カルテに記載する形式でJSONデータを作成してください。

## 制約事項
- 出力はJSON形式のみとしてください。Markdownのコードブロックで囲わず、純粋なJSONのみを返してください。
- 実際の医療現場で使用される初診カルテの形式に準拠してください。
- 値が存在しない場合は空文字列 "" としてください。
- 病名には必ず "#" を先頭に付けてください。

## 出力すべきJSONテンプレート
{
  "patient_info": {
    "name": "",
    "birth_date": "",
    "age": "",
    "gender": ""
  },
  "vitals": {
    "height": "",
    "weight": "",
    "blood_pressure": "",
    "pulse": "",
    "temperature": "",
    "spo2": ""
  },
  "soap": {
    "subjective": "",
    "objective": {
      "consciousness": "",
      "general_condition": "",
      "physical_exam": "",
      "test_results": ""
    },
    "assessment": "",
    "plan": ""
  },
  "diagnosis": [],
  "clinical_course": {
    "onset_and_progress": "",
    "reason_for_referral": "",
    "recent_changes": ""
  },
  "past_medical_history": [],
  "allergies": {
    "drug_allergies": "",
    "food_allergies": "",
    "asthma": ""
  },
  "adverse_drug_reactions": "",
  "lifestyle": {
    "smoking": "",
    "alcohol": "",
    "occupation": ""
  },
  "infectious_disease": "",
  "adl": {
    "walking": "",
    "feeding": "",
    "excretion": "",
    "bathing": "",
    "dressing": "",
    "daily_activities": "",
    "iadl": ""
  },
  "independence_level": "",
  "cognitive_status": {
    "dementia_presence": "",
    "dementia_type": "",
    "severity": "",
    "mmse_score": "",
    "behavioral_symptoms": ""
  },
  "care_info": {
    "care_level": "",
    "disability_certification": "",
    "family_structure": "",
    "key_person": {
      "name": "",
      "relation": "",
      "contact": ""
    },
    "preferred_location": "",
    "care_services": []
  },
  "advance_care_planning": {
    "emergency_response": "",
    "life_sustaining_treatment": "",
    "tube_feeding": "",
    "acute_illness_treatment": "",
    "hospitalization_preference": "",
    "dnr_status": "",
    "organ_donation": "",
    "brain_bank": "",
    "other_wishes": ""
  },
  "current_medications": [],
  "prn_medications": [],
  "treatment_plan": ""
}

## 抽出ルール

### 患者基本情報 (patient_info)
- 氏名、生年月日、年齢、性別を抽出

### バイタルサイン (vitals)
- 身長、体重、血圧、脈拍、体温、SpO2
- 記載がない場合は空文字列

### SOAP形式での記載
- S (Subjective): 患者・家族の訴え、主訴、紹介理由
- O (Objective): 
  - consciousness: 意識レベル（清明、傾眠など）
  - general_condition: 全身状態
  - physical_exam: 身体所見（心音、呼吸音、腹部所見など）
  - test_results: 検査結果（心電図、画像所見、血液検査など）
- A (Assessment): 診断名、病状評価
- P (Plan): 治療計画、紹介先、今後の方針

### 病名 (diagnosis)
- 必ず "#" を付けて記載（例: "#アルツハイマー型認知症"）
- 主病名から順に配列で記載
- 紹介状に記載されている全ての病名を抽出

### 経過概略 (clinical_course)
- onset_and_progress: いつから症状が始まり、どう進行したか
- reason_for_referral: 今回紹介に至った経緯・理由
- recent_changes: 最近の変化や特記事項

### 既往歴 (past_medical_history)
- 過去の病気、手術歴などを配列で記載

### アレルギー (allergies)
- 薬剤アレルギー、食物アレルギー、喘息の有無
- 「なし」と「不明」を区別する

### 生活歴 (lifestyle)
- 喫煙歴（本数×年数）
- 飲酒歴（種類と量）
- 職業・職歴

### ADL評価 (adl)
- walking: 独歩/杖歩行/歩行器/車椅子/寝たきり
- feeding: 自立/一部介助/全介助
- excretion: 自立/一部介助/全介助/おむつ/カテーテル
- bathing: 自立/一部介助/全介助
- dressing: 自立/一部介助/全介助
- daily_activities: 日常動作の自立度
- iadl: 手段的日常生活動作（IADL）の状況

### 自立度 (independence_level)
- 寝たきり度（J1、A1、B1など）の記載があれば抽出

### 認知症評価 (cognitive_status)
- dementia_presence: 認知症の有無
- dementia_type: アルツハイマー型、脳血管性、レビー小体型など
- severity: 軽度/中等症/重度/Ⅰ/Ⅱa/Ⅱb/Ⅲa/Ⅲb/Ⅳ/M
- mmse_score: MMSE得点（例: "16/30"または"16点（30点）"）
- behavioral_symptoms: BPSD（周辺症状）の内容

### 介護情報 (care_info)
- care_level: 要支援1〜2、要介護1〜5
- disability_certification: 障害者手帳の等級
- family_structure: 家族構成（独居、夫婦、子と同居など）
- key_person: キーパーソンの情報
- preferred_location: 本人が過ごしたい場所（自宅、施設など）
- care_services: 利用中の介護サービス（訪問介護、デイサービスなど）

### ACP（アドバンス・ケア・プランニング）
- emergency_response: 急変時の対応方針
- life_sustaining_treatment: 延命治療の意向
- tube_feeding: 胃瘻・経管栄養の希望
- acute_illness_treatment: 治療可能な急性疾患への対応
- hospitalization_preference: 入院の希望
- dnr_status: DNR（Do Not Resuscitate）の有無
- organ_donation: 臓器提供の意向
- brain_bank: ブレインバンク登録の有無
- other_wishes: その他の希望

### 服薬情報
- current_medications: 定期内服薬（薬剤名、用量、用法）
- prn_medications: 頓服薬・屯用薬（使用条件も含めて）

### 治療計画 (treatment_plan)
- 今後の治療方針、観察ポイント、検査予定など

必ず上記の形式に従って、紹介状から得られる情報を漏れなく正確に抽出してください。
特にSOAP形式、病名の "#" 付与、MMSE得点、ADL詳細、ACPは重要です。
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
    """抽出結果を初診カルテ形式で表示"""
    if data is None:
        return

    st.subheader("📋 初診カルテ")

    # 患者基本情報
    st.markdown("### 👤 患者基本情報")
    if "patient_info" in data:
        info = data["patient_info"]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("氏名", info.get("name", ""))
        with col2:
            st.metric("生年月日", info.get("birth_date", ""))
        with col3:
            st.metric("年齢", info.get("age", ""))
        with col4:
            st.metric("性別", info.get("gender", ""))

    # バイタルサイン
    if "vitals" in data and any(data["vitals"].values()):
        st.markdown("### 📊 バイタルサイン")
        vitals = data["vitals"]
        cols = st.columns(6)
        metrics = [
            ("身長", vitals.get("height", "")),
            ("体重", vitals.get("weight", "")),
            ("血圧", vitals.get("blood_pressure", "")),
            ("脈拍", vitals.get("pulse", "")),
            ("体温", vitals.get("temperature", "")),
            ("SpO2", vitals.get("spo2", ""))
        ]
        for col, (label, value) in zip(cols, metrics):
            if value:
                col.metric(label, value)

    # SOAP
    st.markdown("### 📝 SOAP")
    if "soap" in data:
        soap = data["soap"]
        
        # S (Subjective)
        if soap.get("subjective"):
            st.markdown("**■ S (Subjective - 主訴・患者の訴え)**")
            st.write(soap["subjective"])
        
        # O (Objective)
        if "objective" in soap:
            st.markdown("**■ O (Objective - 客観的所見)**")
            obj = soap["objective"]
            if obj.get("consciousness"):
                st.write(f"**意識レベル:** {obj['consciousness']}")
            if obj.get("general_condition"):
                st.write(f"**全身状態:** {obj['general_condition']}")
            if obj.get("physical_exam"):
                st.write(f"**身体所見:**")
                st.write(obj["physical_exam"])
            if obj.get("test_results"):
                st.write(f"**検査結果:**")
                st.write(obj["test_results"])
        
        # A (Assessment)
        if soap.get("assessment"):
            st.markdown("**■ A (Assessment - 評価)**")
            st.write(soap["assessment"])
        
        # P (Plan)
        if soap.get("plan"):
            st.markdown("**■ P (Plan - 計画)**")
            st.write(soap["plan"])

    # 病名
    if "diagnosis" in data and data["diagnosis"]:
        st.markdown("### 🏥 病名")
        for dx in data["diagnosis"]:
            st.write(f"- {dx}")

    # 経過概略
    if "clinical_course" in data:
        st.markdown("### 📅 経過概略")
        course = data["clinical_course"]
        if course.get("onset_and_progress"):
            st.write(f"**発症と経過:** {course['onset_and_progress']}")
        if course.get("reason_for_referral"):
            st.write(f"**紹介理由:** {course['reason_for_referral']}")
        if course.get("recent_changes"):
            st.write(f"**最近の変化:** {course['recent_changes']}")

    # 既往歴・アレルギー・生活歴
    col1, col2 = st.columns(2)
    
    with col1:
        if "past_medical_history" in data and data["past_medical_history"]:
            st.markdown("### 🏥 既往歴")
            for history in data["past_medical_history"]:
                st.write(f"- {history}")
        
        if "allergies" in data:
            st.markdown("### ⚠️ アレルギー")
            allergies = data["allergies"]
            allergy_data = []
            if allergies.get("drug_allergies"):
                allergy_data.append({"種類": "薬剤", "内容": allergies["drug_allergies"]})
            if allergies.get("food_allergies"):
                allergy_data.append({"種類": "食物", "内容": allergies["food_allergies"]})
            if allergies.get("asthma"):
                allergy_data.append({"種類": "喘息", "内容": allergies["asthma"]})
            if allergy_data:
                st.dataframe(pd.DataFrame(allergy_data), use_container_width=True, hide_index=True)
        
        if data.get("adverse_drug_reactions"):
            st.markdown("### 💊 副作用歴")
            st.write(data["adverse_drug_reactions"])
    
    with col2:
        if "lifestyle" in data:
            st.markdown("### 🚬 生活歴")
            lifestyle = data["lifestyle"]
            lifestyle_data = []
            if lifestyle.get("smoking"):
                lifestyle_data.append({"項目": "喫煙", "内容": lifestyle["smoking"]})
            if lifestyle.get("alcohol"):
                lifestyle_data.append({"項目": "飲酒", "内容": lifestyle["alcohol"]})
            if lifestyle.get("occupation"):
                lifestyle_data.append({"項目": "職業", "内容": lifestyle["occupation"]})
            if lifestyle_data:
                st.dataframe(pd.DataFrame(lifestyle_data), use_container_width=True, hide_index=True)
        
        if data.get("infectious_disease"):
            st.markdown("### 🦠 感染症")
            st.write(data["infectious_disease"])

    # ADL・IADL
    st.markdown("### 🚶 ADL・IADL")
    if "adl" in data:
        adl = data["adl"]
        adl_data = []
        if adl.get("walking"):
            adl_data.append({"項目": "歩行", "状態": adl["walking"]})
        if adl.get("feeding"):
            adl_data.append({"項目": "食事", "状態": adl["feeding"]})
        if adl.get("excretion"):
            adl_data.append({"項目": "排泄", "状態": adl["excretion"]})
        if adl.get("bathing"):
            adl_data.append({"項目": "入浴", "状態": adl["bathing"]})
        if adl.get("dressing"):
            adl_data.append({"項目": "着衣", "状態": adl["dressing"]})
        if adl.get("daily_activities"):
            adl_data.append({"項目": "日常動作", "状態": adl["daily_activities"]})
        if adl.get("iadl"):
            adl_data.append({"項目": "IADL", "状態": adl["iadl"]})
        if adl_data:
            st.dataframe(pd.DataFrame(adl_data), use_container_width=True, hide_index=True)
    
    if data.get("independence_level"):
        st.write(f"**自立度:** {data['independence_level']}")

    # 認知症評価
    if "cognitive_status" in data:
        st.markdown("### 🧠 認知症評価")
        cog = data["cognitive_status"]
        cog_data = []
        if cog.get("dementia_presence"):
            cog_data.append({"項目": "認知症の有無", "内容": cog["dementia_presence"]})
        if cog.get("dementia_type"):
            cog_data.append({"項目": "認知症の種類", "内容": cog["dementia_type"]})
        if cog.get("severity"):
            cog_data.append({"項目": "重症度", "内容": cog["severity"]})
        if cog.get("mmse_score"):
            cog_data.append({"項目": "MMSE", "内容": cog["mmse_score"]})
        if cog.get("behavioral_symptoms"):
            cog_data.append({"項目": "周辺症状(BPSD)", "内容": cog["behavioral_symptoms"]})
        if cog_data:
            st.dataframe(pd.DataFrame(cog_data), use_container_width=True, hide_index=True)

    # 介護情報
    if "care_info" in data:
        st.markdown("### 👨‍👩‍👧‍👦 介護情報")
        care = data["care_info"]
        
        col1, col2 = st.columns(2)
        with col1:
            if care.get("care_level"):
                st.write(f"**要介護度:** {care['care_level']}")
            if care.get("disability_certification"):
                st.write(f"**障害認定:** {care['disability_certification']}")
            if care.get("family_structure"):
                st.write(f"**家族構成:** {care['family_structure']}")
        
        with col2:
            if "key_person" in care:
                kp = care["key_person"]
                st.write("**キーパーソン**")
                if kp.get("name"):
                    st.write(f"- 氏名: {kp['name']}")
                if kp.get("relation"):
                    st.write(f"- 続柄: {kp['relation']}")
                if kp.get("contact"):
                    st.write(f"- 連絡先: {kp['contact']}")
        
        if care.get("preferred_location"):
            st.write(f"**過ごしたい場所:** {care['preferred_location']}")
        
        if care.get("care_services"):
            st.write("**利用中の介護サービス:**")
            for service in care["care_services"]:
                st.write(f"- {service}")

    # ACP（アドバンス・ケア・プランニング）
    if "advance_care_planning" in data:
        st.markdown("### 📋 ACP（アドバンス・ケア・プランニング）")
        acp = data["advance_care_planning"]
        acp_data = []
        if acp.get("emergency_response"):
            acp_data.append({"項目": "急変時対応", "内容": acp["emergency_response"]})
        if acp.get("life_sustaining_treatment"):
            acp_data.append({"項目": "延命治療", "内容": acp["life_sustaining_treatment"]})
        if acp.get("tube_feeding"):
            acp_data.append({"項目": "経管栄養・胃瘻", "内容": acp["tube_feeding"]})
        if acp.get("acute_illness_treatment"):
            acp_data.append({"項目": "急性疾患の治療", "内容": acp["acute_illness_treatment"]})
        if acp.get("hospitalization_preference"):
            acp_data.append({"項目": "入院の希望", "内容": acp["hospitalization_preference"]})
        if acp.get("dnr_status"):
            acp_data.append({"項目": "DNR", "内容": acp["dnr_status"]})
        if acp.get("organ_donation"):
            acp_data.append({"項目": "臓器提供", "内容": acp["organ_donation"]})
        if acp.get("brain_bank"):
            acp_data.append({"項目": "ブレインバンク", "内容": acp["brain_bank"]})
        if acp.get("other_wishes"):
            acp_data.append({"項目": "その他の希望", "内容": acp["other_wishes"]})
        if acp_data:
            st.dataframe(pd.DataFrame(acp_data), use_container_width=True, hide_index=True)

    # 服薬情報
    col1, col2 = st.columns(2)
    with col1:
        if "current_medications" in data and data["current_medications"]:
            st.markdown("### 💊 定期内服薬")
            for med in data["current_medications"]:
                st.write(f"- {med}")
    
    with col2:
        if "prn_medications" in data and data["prn_medications"]:
            st.markdown("### 💊 頓服・屯用薬")
            for med in data["prn_medications"]:
                st.write(f"- {med}")

    # 治療計画
    if data.get("treatment_plan"):
        st.markdown("### 📋 治療計画")
        st.write(data["treatment_plan"])

    # JSON形式でも表示（開発者向け）
    with st.expander("🔧 JSON形式で表示（開発者向け）"):
        st.json(data)

# メインコンテンツ
tab1, tab2 = st.tabs(["📷 画像アップロード", "📝 テキスト入力"])

with tab1:
    st.markdown("### スマートフォンで撮影した紹介状の写真をアップロード")
    uploaded_file = st.file_uploader(
        "画像ファイルを選択してください",
        type=["jpg", "jpeg", "png", "pdf"],
        help="紹介状の写真またはPDFをアップロードしてください"
    )

    if uploaded_file is not None:
        # 2カラムレイアウト
        col1, col2 = st.columns([1, 1.5])

        with col1:
            st.subheader("📄 入力画像")
            if uploaded_file.type == "application/pdf":
                st.info("PDFファイルがアップロードされました")
            else:
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)

        with col2:
            if st.button("🔍 情報を抽出して初診カルテを作成", key="extract_image", type="primary"):
                with st.spinner("AIが紹介状から情報を抽出中..."):
                    if uploaded_file.type == "application/pdf":
                        # PDFの場合は一旦画像として読み込む
                        from pdf2image import convert_from_bytes
                        images = convert_from_bytes(uploaded_file.read())
                        if images:
                            extracted_data = extract_info_from_image(images[0])
                        else:
                            st.error("PDFの読み込みに失敗しました")
                            extracted_data = None
                    else:
                        image = Image.open(uploaded_file)
                        extracted_data = extract_info_from_image(image)
                    
                    if extracted_data:
                        display_results(extracted_data)

with tab2:
    st.markdown("### 電子カルテからコピーしたテキストを貼り付け")
    text_input = st.text_area(
        "紹介状のテキストを入力してください",
        height=400,
        placeholder="""例:
患者名: 山田太郎
生年月日: 1950年4月15日
主訴: 胸部不快感

【病名】
#虚血性心疾患
#高血圧症
#糖尿病

【既往歴】
75歳 腰椎圧迫骨折
60歳 脳梗塞

【ADL】
歩行：車椅子
食事：自立
排泄：介助

【認知症】アルツハイマー型 MMSE 14/30

...""",
        help="電子カルテからコピーしたテキストを貼り付けてください"
    )

    if text_input:
        col1, col2 = st.columns([1, 1.5])

        with col1:
            st.subheader("📄 入力テキスト")
            st.text_area("入力内容", value=text_input, height=400, disabled=True)

        with col2:
            if st.button("🔍 情報を抽出して初診カルテを作成", key="extract_text", type="primary"):
                with st.spinner("AIが紹介状から情報を抽出中..."):
                    extracted_data = extract_info_from_text(text_input)
                    if extracted_data:
                        display_results(extracted_data)

# フッター
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <small>医療ハッカソン デモアプリ | 実際の初診カルテ形式に準拠 | Powered by Google Gemini 2.0 Flash</small>
    </div>
    """,
    unsafe_allow_html=True
)