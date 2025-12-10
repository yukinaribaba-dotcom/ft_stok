import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import json
import pandas as pd
from typing import Dict, Any, List
import io
import os
import base64

# ページ設定
st.set_page_config(
    page_title="医療紹介状→初診カルテ変換アプリ",
    page_icon="🏥",
    layout="wide"
)

# タイトル
st.title("🏥 医療紹介状→初診カルテ変換アプリ（複数ファイル対応版）")
st.markdown("紹介状の画像またはテキストから初診カルテ形式で患者情報を自動抽出します")

# APIキーの確認と設定
try:
    # 環境変数から取得を試み、なければst.secretsから取得
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = st.secrets["GOOGLE_API_KEY"]

    if not api_key:
        raise KeyError("API key not found")

except KeyError:
    st.error("⚠️ Google API Keyが設定されていません。環境変数`GEMINI_API_KEY`または`secrets.toml`に`GOOGLE_API_KEY`を設定してください。")
    st.stop()
except Exception as e:
    st.error(f"⚠️ API設定エラー: {str(e)}")
    st.stop()

# Geminiクライアントとモデルの初期化
@st.cache_resource
def get_client_and_model():
    client = genai.Client(api_key=api_key)
    model = "gemini-3-pro-preview"
    return client, model

client, model = get_client_and_model()

# プロンプトテンプレート
EXTRACTION_PROMPT = """あなたは在宅医療の医師です。
提供された「診療情報提供書（紹介状）」の内容を読み取り、訪問診療開始時の初診カルテに記載する形式でJSONデータを作成してください。

複数の画像やページがある場合は、すべての情報を統合して1つの初診カルテとして出力してください。

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

def process_pdf_to_images(pdf_file) -> List[Image.Image]:
    """PDFを画像のリストに変換"""
    try:
        from pdf2image import convert_from_bytes
        pdf_file.seek(0)
        images = convert_from_bytes(pdf_file.read())
        return images
    except Exception as e:
        st.error(f"PDF処理エラー: {str(e)}")
        return []

def extract_info_from_multiple_files(files: List) -> Dict[str, Any]:
    """複数ファイルから情報を抽出"""
    try:
        # すべてのファイルを画像に変換
        all_images = []

        for file in files:
            file.seek(0)
            if file.type == "application/pdf":
                images = process_pdf_to_images(file)
                all_images.extend(images)
            else:
                image = Image.open(file)
                all_images.append(image)

        if not all_images:
            st.error("画像の読み込みに失敗しました")
            return None

        # 画像をBase64エンコードしてインライン画像として送信
        image_parts = []
        for img in all_images:
            # PILイメージをバイト配列に変換
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            # Base64エンコード
            img_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')
            # インライン画像パートを作成
            image_parts.append(
                types.Part.from_bytes(
                    data=base64.b64decode(img_base64),
                    mime_type="image/png"
                )
            )

        # コンテンツを作成
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=EXTRACTION_PROMPT)] + image_parts
            )
        ]

        # 生成設定
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config={"thinking_level": "HIGH"}
        )

        # コンテンツ生成
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config
        )

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
        st.error(f"ファイル処理エラー: {str(e)}")
        return None

def extract_info_from_text(text: str) -> Dict[str, Any]:
    """テキストから情報を抽出"""
    try:
        prompt = EXTRACTION_PROMPT + f"\n\n入力テキスト:\n{text}"

        # コンテンツを作成
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]

        # 生成設定
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config={"thinking_level": "HIGH"}
        )

        # コンテンツ生成
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config
        )

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

    # タブで「見やすい表示」と「テキスト生データ」を切り替え
    tab1, tab2 = st.tabs(["📋 カルテ表示", "📄 テキスト生データ"])

    with tab1:
        # === 患者基本情報 ===
        st.markdown("### 👤 患者基本情報")
        if "patient_info" in data:
            info = data["patient_info"]
            patient_info_md = f"""
- **氏名**: {info.get("name", "未記載")}
- **生年月日**: {info.get("birth_date", "未記載")}
- **年齢**: {info.get("age", "未記載")}
- **性別**: {info.get("gender", "未記載")}
"""
            st.markdown(patient_info_md)
        st.markdown("---")

        # === バイタルサイン ===
        if "vitals" in data and any(data["vitals"].values()):
            st.markdown("### 📊 バイタルサイン")
            vitals = data["vitals"]
            vital_items = []
            if vitals.get("height"):
                vital_items.append({"項目": "身長", "値": vitals["height"]})
            if vitals.get("weight"):
                vital_items.append({"項目": "体重", "値": vitals["weight"]})
            if vitals.get("blood_pressure"):
                vital_items.append({"項目": "血圧", "値": vitals["blood_pressure"]})
            if vitals.get("pulse"):
                vital_items.append({"項目": "脈拍", "値": vitals["pulse"]})
            if vitals.get("temperature"):
                vital_items.append({"項目": "体温", "値": vitals["temperature"]})
            if vitals.get("spo2"):
                vital_items.append({"項目": "SpO2", "値": vitals["spo2"]})

            if vital_items:
                st.table(pd.DataFrame(vital_items))
            st.markdown("---")

        # === 病名 ===
        if "diagnosis" in data and data["diagnosis"]:
            st.markdown("### 🏥 病名")
            for dx in data["diagnosis"]:
                st.markdown(f"- {dx}")
            st.markdown("---")

        # === SOAP ===
        st.markdown("### 📝 SOAP")
        if "soap" in data:
            soap = data["soap"]

            # S (Subjective)
            if soap.get("subjective"):
                st.markdown("**■ S (Subjective - 主訴・患者の訴え)**")
                st.markdown(f"> {soap['subjective']}")
                st.write("")

            # O (Objective)
            if "objective" in soap:
                st.markdown("**■ O (Objective - 客観的所見)**")
                obj = soap["objective"]
                obj_items = []
                if obj.get("consciousness"):
                    obj_items.append(f"- **意識レベル**: {obj['consciousness']}")
                if obj.get("general_condition"):
                    obj_items.append(f"- **全身状態**: {obj['general_condition']}")
                if obj.get("physical_exam"):
                    obj_items.append(f"- **身体所見**: {obj['physical_exam']}")
                if obj.get("test_results"):
                    obj_items.append(f"- **検査結果**: {obj['test_results']}")

                for item in obj_items:
                    st.markdown(item)
                st.write("")

            # A (Assessment)
            if soap.get("assessment"):
                st.markdown("**■ A (Assessment - 評価)**")
                st.markdown(f"> {soap['assessment']}")
                st.write("")

            # P (Plan)
            if soap.get("plan"):
                st.markdown("**■ P (Plan - 計画)**")
                st.markdown(f"> {soap['plan']}")
                st.write("")
        st.markdown("---")

        # === 経過概略 ===
        if "clinical_course" in data:
            course = data["clinical_course"]
            if any(course.values()):
                st.markdown("### 📅 経過概略")
                if course.get("onset_and_progress"):
                    st.markdown(f"**発症と経過**  \n{course['onset_and_progress']}")
                if course.get("reason_for_referral"):
                    st.markdown(f"**紹介理由**  \n{course['reason_for_referral']}")
                if course.get("recent_changes"):
                    st.markdown(f"**最近の変化**  \n{course['recent_changes']}")
                st.markdown("---")

        # === 既往歴 ===
        if "past_medical_history" in data and data["past_medical_history"]:
            st.markdown("### 🏥 既往歴")
            for history in data["past_medical_history"]:
                st.markdown(f"- {history}")
            st.markdown("---")

        # === アレルギー ===
        if "allergies" in data:
            allergies = data["allergies"]
            if any(allergies.values()):
                st.markdown("### ⚠️ アレルギー")
                allergy_items = []
                if allergies.get("drug_allergies"):
                    allergy_items.append({"種類": "薬剤", "内容": allergies["drug_allergies"]})
                if allergies.get("food_allergies"):
                    allergy_items.append({"種類": "食物", "内容": allergies["food_allergies"]})
                if allergies.get("asthma"):
                    allergy_items.append({"種類": "喘息", "内容": allergies["asthma"]})
                if allergy_items:
                    st.table(pd.DataFrame(allergy_items))
                st.markdown("---")

        # === 副作用歴 ===
        if data.get("adverse_drug_reactions"):
            st.markdown("### 💊 副作用歴")
            st.markdown(f"- {data['adverse_drug_reactions']}")
            st.markdown("---")

        # === 生活歴 ===
        if "lifestyle" in data:
            lifestyle = data["lifestyle"]
            if any(lifestyle.values()):
                st.markdown("### 🚬 生活歴")
                lifestyle_items = []
                if lifestyle.get("smoking"):
                    lifestyle_items.append({"項目": "喫煙", "内容": lifestyle["smoking"]})
                if lifestyle.get("alcohol"):
                    lifestyle_items.append({"項目": "飲酒", "内容": lifestyle["alcohol"]})
                if lifestyle.get("occupation"):
                    lifestyle_items.append({"項目": "職業", "内容": lifestyle["occupation"]})
                if lifestyle_items:
                    st.table(pd.DataFrame(lifestyle_items))
                st.markdown("---")

        # === 感染症 ===
        if data.get("infectious_disease"):
            st.markdown("### 🦠 感染症")
            st.markdown(f"- {data['infectious_disease']}")
            st.markdown("---")

        # === ADL・IADL ===
        st.markdown("### 🚶 ADL・IADL")
        if "adl" in data:
            adl = data["adl"]
            adl_items = []
            if adl.get("walking"):
                adl_items.append({"項目": "歩行", "状態": adl["walking"]})
            if adl.get("feeding"):
                adl_items.append({"項目": "食事", "状態": adl["feeding"]})
            if adl.get("excretion"):
                adl_items.append({"項目": "排泄", "状態": adl["excretion"]})
            if adl.get("bathing"):
                adl_items.append({"項目": "入浴", "状態": adl["bathing"]})
            if adl.get("dressing"):
                adl_items.append({"項目": "着衣", "状態": adl["dressing"]})
            if adl.get("daily_activities"):
                adl_items.append({"項目": "日常動作", "状態": adl["daily_activities"]})
            if adl.get("iadl"):
                adl_items.append({"項目": "IADL", "状態": adl["iadl"]})
            if adl_items:
                st.table(pd.DataFrame(adl_items))

        if data.get("independence_level"):
            st.markdown(f"**自立度**: {data['independence_level']}")
        st.markdown("---")

        # === 認知症評価 ===
        if "cognitive_status" in data:
            cog = data["cognitive_status"]
            if any(cog.values()):
                st.markdown("### 🧠 認知症評価")
                cog_items = []
                if cog.get("dementia_presence"):
                    cog_items.append({"項目": "認知症の有無", "内容": cog["dementia_presence"]})
                if cog.get("dementia_type"):
                    cog_items.append({"項目": "認知症の種類", "内容": cog["dementia_type"]})
                if cog.get("severity"):
                    cog_items.append({"項目": "重症度", "内容": cog["severity"]})
                if cog.get("mmse_score"):
                    cog_items.append({"項目": "MMSE", "内容": cog["mmse_score"]})
                if cog.get("behavioral_symptoms"):
                    cog_items.append({"項目": "周辺症状(BPSD)", "内容": cog["behavioral_symptoms"]})
                if cog_items:
                    st.table(pd.DataFrame(cog_items))
                st.markdown("---")

        # === 介護情報 ===
        if "care_info" in data:
            care = data["care_info"]
            if any([care.get("care_level"), care.get("disability_certification"),
                   care.get("family_structure"), care.get("key_person"),
                   care.get("preferred_location"), care.get("care_services")]):
                st.markdown("### 👨‍👩‍👧‍👦 介護情報")

                care_md_items = []
                if care.get("care_level"):
                    care_md_items.append(f"- **要介護度**: {care['care_level']}")
                if care.get("disability_certification"):
                    care_md_items.append(f"- **障害認定**: {care['disability_certification']}")
                if care.get("family_structure"):
                    care_md_items.append(f"- **家族構成**: {care['family_structure']}")
                if care.get("preferred_location"):
                    care_md_items.append(f"- **過ごしたい場所**: {care['preferred_location']}")

                for item in care_md_items:
                    st.markdown(item)

                if "key_person" in care:
                    kp = care["key_person"]
                    if any(kp.values()):
                        st.markdown("**キーパーソン**")
                        if kp.get("name"):
                            st.markdown(f"- 氏名: {kp['name']}")
                        if kp.get("relation"):
                            st.markdown(f"- 続柄: {kp['relation']}")
                        if kp.get("contact"):
                            st.markdown(f"- 連絡先: {kp['contact']}")

                if care.get("care_services"):
                    st.markdown("**利用中の介護サービス**")
                    for service in care["care_services"]:
                        st.markdown(f"- {service}")

                st.markdown("---")

        # === ACP ===
        if "advance_care_planning" in data:
            acp = data["advance_care_planning"]
            if any(acp.values()):
                st.markdown("### 📋 ACP（アドバンス・ケア・プランニング）")
                acp_items = []
                if acp.get("emergency_response"):
                    acp_items.append({"項目": "急変時対応", "内容": acp["emergency_response"]})
                if acp.get("life_sustaining_treatment"):
                    acp_items.append({"項目": "延命治療", "内容": acp["life_sustaining_treatment"]})
                if acp.get("tube_feeding"):
                    acp_items.append({"項目": "経管栄養・胃瘻", "内容": acp["tube_feeding"]})
                if acp.get("acute_illness_treatment"):
                    acp_items.append({"項目": "急性疾患の治療", "内容": acp["acute_illness_treatment"]})
                if acp.get("hospitalization_preference"):
                    acp_items.append({"項目": "入院の希望", "内容": acp["hospitalization_preference"]})
                if acp.get("dnr_status"):
                    acp_items.append({"項目": "DNR", "内容": acp["dnr_status"]})
                if acp.get("organ_donation"):
                    acp_items.append({"項目": "臓器提供", "内容": acp["organ_donation"]})
                if acp.get("brain_bank"):
                    acp_items.append({"項目": "ブレインバンク", "内容": acp["brain_bank"]})
                if acp.get("other_wishes"):
                    acp_items.append({"項目": "その他の希望", "内容": acp["other_wishes"]})
                if acp_items:
                    st.table(pd.DataFrame(acp_items))
                st.markdown("---")

        # === 服薬情報 ===
        if "current_medications" in data and data["current_medications"]:
            st.markdown("### 💊 定期内服薬")
            for med in data["current_medications"]:
                st.markdown(f"- {med}")
            st.markdown("---")

        if "prn_medications" in data and data["prn_medications"]:
            st.markdown("### 💊 頓服・屯用薬")
            for med in data["prn_medications"]:
                st.markdown(f"- {med}")
            st.markdown("---")

        # === 治療計画 ===
        if data.get("treatment_plan"):
            st.markdown("### 📋 治療計画")
            st.markdown(data["treatment_plan"])
            st.markdown("---")

    with tab2:
        # テキスト生データ表示（コピペしやすい形式）
        st.markdown("### 📄 テキスト生データ（コピペ用）")

        text_output = []

        # 患者基本情報
        if "patient_info" in data:
            text_output.append("=" * 60)
            text_output.append("【患者基本情報】")
            text_output.append("=" * 60)
            info = data["patient_info"]
            text_output.append(f"氏名: {info.get('name', '')}")
            text_output.append(f"生年月日: {info.get('birth_date', '')}")
            text_output.append(f"年齢: {info.get('age', '')}")
            text_output.append(f"性別: {info.get('gender', '')}")
            text_output.append("")

        # バイタルサイン
        if "vitals" in data and any(data["vitals"].values()):
            text_output.append("=" * 60)
            text_output.append("【バイタルサイン】")
            text_output.append("=" * 60)
            vitals = data["vitals"]
            if vitals.get("height"):
                text_output.append(f"身長: {vitals['height']}")
            if vitals.get("weight"):
                text_output.append(f"体重: {vitals['weight']}")
            if vitals.get("blood_pressure"):
                text_output.append(f"血圧: {vitals['blood_pressure']}")
            if vitals.get("pulse"):
                text_output.append(f"脈拍: {vitals['pulse']}")
            if vitals.get("temperature"):
                text_output.append(f"体温: {vitals['temperature']}")
            if vitals.get("spo2"):
                text_output.append(f"SpO2: {vitals['spo2']}")
            text_output.append("")

        # 病名
        if "diagnosis" in data and data["diagnosis"]:
            text_output.append("=" * 60)
            text_output.append("【病名】")
            text_output.append("=" * 60)
            for dx in data["diagnosis"]:
                text_output.append(dx)
            text_output.append("")

        # SOAP
        if "soap" in data:
            text_output.append("=" * 60)
            text_output.append("【SOAP】")
            text_output.append("=" * 60)
            soap = data["soap"]
            if soap.get("subjective"):
                text_output.append("■ S (Subjective - 主訴)")
                text_output.append(soap["subjective"])
                text_output.append("")

            if "objective" in soap:
                text_output.append("■ O (Objective - 客観的所見)")
                obj = soap["objective"]
                if obj.get("consciousness"):
                    text_output.append(f"意識レベル: {obj['consciousness']}")
                if obj.get("general_condition"):
                    text_output.append(f"全身状態: {obj['general_condition']}")
                if obj.get("physical_exam"):
                    text_output.append(f"身体所見: {obj['physical_exam']}")
                if obj.get("test_results"):
                    text_output.append(f"検査結果: {obj['test_results']}")
                text_output.append("")

            if soap.get("assessment"):
                text_output.append("■ A (Assessment - 評価)")
                text_output.append(soap["assessment"])
                text_output.append("")

            if soap.get("plan"):
                text_output.append("■ P (Plan - 計画)")
                text_output.append(soap["plan"])
                text_output.append("")

        # 経過概略
        if "clinical_course" in data:
            course = data["clinical_course"]
            if any(course.values()):
                text_output.append("=" * 60)
                text_output.append("【経過概略】")
                text_output.append("=" * 60)
                if course.get("onset_and_progress"):
                    text_output.append(f"発症と経過: {course['onset_and_progress']}")
                if course.get("reason_for_referral"):
                    text_output.append(f"紹介理由: {course['reason_for_referral']}")
                if course.get("recent_changes"):
                    text_output.append(f"最近の変化: {course['recent_changes']}")
                text_output.append("")

        # 既往歴
        if "past_medical_history" in data and data["past_medical_history"]:
            text_output.append("=" * 60)
            text_output.append("【既往歴】")
            text_output.append("=" * 60)
            for history in data["past_medical_history"]:
                text_output.append(f"- {history}")
            text_output.append("")

        # アレルギー
        if "allergies" in data and any(data["allergies"].values()):
            text_output.append("=" * 60)
            text_output.append("【アレルギー】")
            text_output.append("=" * 60)
            allergies = data["allergies"]
            if allergies.get("drug_allergies"):
                text_output.append(f"薬剤: {allergies['drug_allergies']}")
            if allergies.get("food_allergies"):
                text_output.append(f"食物: {allergies['food_allergies']}")
            if allergies.get("asthma"):
                text_output.append(f"喘息: {allergies['asthma']}")
            text_output.append("")

        # 副作用歴
        if data.get("adverse_drug_reactions"):
            text_output.append("=" * 60)
            text_output.append("【副作用歴】")
            text_output.append("=" * 60)
            text_output.append(data["adverse_drug_reactions"])
            text_output.append("")

        # 生活歴
        if "lifestyle" in data and any(data["lifestyle"].values()):
            text_output.append("=" * 60)
            text_output.append("【生活歴】")
            text_output.append("=" * 60)
            lifestyle = data["lifestyle"]
            if lifestyle.get("smoking"):
                text_output.append(f"喫煙: {lifestyle['smoking']}")
            if lifestyle.get("alcohol"):
                text_output.append(f"飲酒: {lifestyle['alcohol']}")
            if lifestyle.get("occupation"):
                text_output.append(f"職業: {lifestyle['occupation']}")
            text_output.append("")

        # 感染症
        if data.get("infectious_disease"):
            text_output.append("=" * 60)
            text_output.append("【感染症】")
            text_output.append("=" * 60)
            text_output.append(data["infectious_disease"])
            text_output.append("")

        # ADL
        if "adl" in data and any(data["adl"].values()):
            text_output.append("=" * 60)
            text_output.append("【ADL・IADL】")
            text_output.append("=" * 60)
            adl = data["adl"]
            if adl.get("walking"):
                text_output.append(f"歩行: {adl['walking']}")
            if adl.get("feeding"):
                text_output.append(f"食事: {adl['feeding']}")
            if adl.get("excretion"):
                text_output.append(f"排泄: {adl['excretion']}")
            if adl.get("bathing"):
                text_output.append(f"入浴: {adl['bathing']}")
            if adl.get("dressing"):
                text_output.append(f"着衣: {adl['dressing']}")
            if adl.get("daily_activities"):
                text_output.append(f"日常動作: {adl['daily_activities']}")
            if adl.get("iadl"):
                text_output.append(f"IADL: {adl['iadl']}")
            if data.get("independence_level"):
                text_output.append(f"自立度: {data['independence_level']}")
            text_output.append("")

        # 認知症評価
        if "cognitive_status" in data and any(data["cognitive_status"].values()):
            text_output.append("=" * 60)
            text_output.append("【認知症評価】")
            text_output.append("=" * 60)
            cog = data["cognitive_status"]
            if cog.get("dementia_presence"):
                text_output.append(f"認知症の有無: {cog['dementia_presence']}")
            if cog.get("dementia_type"):
                text_output.append(f"認知症の種類: {cog['dementia_type']}")
            if cog.get("severity"):
                text_output.append(f"重症度: {cog['severity']}")
            if cog.get("mmse_score"):
                text_output.append(f"MMSE: {cog['mmse_score']}")
            if cog.get("behavioral_symptoms"):
                text_output.append(f"周辺症状(BPSD): {cog['behavioral_symptoms']}")
            text_output.append("")

        # 介護情報
        if "care_info" in data:
            care = data["care_info"]
            if any([care.get("care_level"), care.get("disability_certification"),
                   care.get("family_structure"), care.get("key_person"),
                   care.get("preferred_location"), care.get("care_services")]):
                text_output.append("=" * 60)
                text_output.append("【介護情報】")
                text_output.append("=" * 60)
                if care.get("care_level"):
                    text_output.append(f"要介護度: {care['care_level']}")
                if care.get("disability_certification"):
                    text_output.append(f"障害認定: {care['disability_certification']}")
                if care.get("family_structure"):
                    text_output.append(f"家族構成: {care['family_structure']}")
                if care.get("preferred_location"):
                    text_output.append(f"過ごしたい場所: {care['preferred_location']}")

                if "key_person" in care and any(care["key_person"].values()):
                    text_output.append("キーパーソン:")
                    kp = care["key_person"]
                    if kp.get("name"):
                        text_output.append(f"  氏名: {kp['name']}")
                    if kp.get("relation"):
                        text_output.append(f"  続柄: {kp['relation']}")
                    if kp.get("contact"):
                        text_output.append(f"  連絡先: {kp['contact']}")

                if care.get("care_services"):
                    text_output.append("利用中の介護サービス:")
                    for service in care["care_services"]:
                        text_output.append(f"  - {service}")
                text_output.append("")

        # ACP
        if "advance_care_planning" in data and any(data["advance_care_planning"].values()):
            text_output.append("=" * 60)
            text_output.append("【ACP（アドバンス・ケア・プランニング）】")
            text_output.append("=" * 60)
            acp = data["advance_care_planning"]
            if acp.get("emergency_response"):
                text_output.append(f"急変時対応: {acp['emergency_response']}")
            if acp.get("life_sustaining_treatment"):
                text_output.append(f"延命治療: {acp['life_sustaining_treatment']}")
            if acp.get("tube_feeding"):
                text_output.append(f"経管栄養・胃瘻: {acp['tube_feeding']}")
            if acp.get("acute_illness_treatment"):
                text_output.append(f"急性疾患の治療: {acp['acute_illness_treatment']}")
            if acp.get("hospitalization_preference"):
                text_output.append(f"入院の希望: {acp['hospitalization_preference']}")
            if acp.get("dnr_status"):
                text_output.append(f"DNR: {acp['dnr_status']}")
            if acp.get("organ_donation"):
                text_output.append(f"臓器提供: {acp['organ_donation']}")
            if acp.get("brain_bank"):
                text_output.append(f"ブレインバンク: {acp['brain_bank']}")
            if acp.get("other_wishes"):
                text_output.append(f"その他の希望: {acp['other_wishes']}")
            text_output.append("")

        # 服薬情報
        if "current_medications" in data and data["current_medications"]:
            text_output.append("=" * 60)
            text_output.append("【定期内服薬】")
            text_output.append("=" * 60)
            for med in data["current_medications"]:
                text_output.append(f"- {med}")
            text_output.append("")

        if "prn_medications" in data and data["prn_medications"]:
            text_output.append("=" * 60)
            text_output.append("【頓服・屯用薬】")
            text_output.append("=" * 60)
            for med in data["prn_medications"]:
                text_output.append(f"- {med}")
            text_output.append("")

        # 治療計画
        if data.get("treatment_plan"):
            text_output.append("=" * 60)
            text_output.append("【治療計画】")
            text_output.append("=" * 60)
            text_output.append(data["treatment_plan"])
            text_output.append("")

        # テキストエリアに表示
        full_text = "\n".join(text_output)
        st.text_area("コピー可能なテキスト", value=full_text, height=600)

        # JSON形式でも表示（開発者向け）
        with st.expander("🔧 JSON形式で表示（開発者向け）"):
            st.json(data)

# メインコンテンツ
tab1, tab2 = st.tabs(["📷 画像アップロード", "📝 テキスト入力"])

with tab1:
    st.markdown("### スマートフォンで撮影した紹介状の写真をアップロード")
    st.info("💡 **複数のファイルを同時にアップロードできます！** 紹介状が複数ページに分かれている場合や、検査結果など関連資料がある場合に便利です。Ctrl/Cmd + クリックで複数選択できます。")
    
    uploaded_files = st.file_uploader(
        "画像ファイルを選択してください（複数選択可）",
        type=["jpg", "jpeg", "png", "pdf"],
        help="紹介状の写真またはPDFをアップロードしてください。Ctrl/Cmd + クリックで複数選択できます",
        accept_multiple_files=True
    )

    if uploaded_files:
        # アップロードされたファイルの情報を表示
        st.success(f"✅ {len(uploaded_files)}個のファイルがアップロードされました")
        
        with st.expander("📁 アップロードファイル一覧", expanded=True):
            for i, file in enumerate(uploaded_files, 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{i}. {file.name}**")
                with col2:
                    st.write(f"{file.size / 1024:.1f} KB")
        
        # プレビュー表示（最大6ファイルまで）
        st.markdown("### 📸 プレビュー")
        preview_files = uploaded_files[:6]
        
        if len(uploaded_files) <= 3:
            cols = st.columns(len(uploaded_files))
        else:
            cols = st.columns(3)
        
        for idx, file in enumerate(preview_files):
            col_idx = idx % 3
            with cols[col_idx]:
                if file.type == "application/pdf":
                    st.info(f"📄 PDF: {file.name}")
                else:
                    file.seek(0)
                    image = Image.open(file)
                    st.image(image, caption=file.name, use_container_width=True)
        
        if len(uploaded_files) > 6:
            st.info(f"その他 {len(uploaded_files) - 6} 個のファイル...")
        
        st.markdown("---")
        
        # 抽出ボタン
        if st.button("🔍 全ファイルから情報を抽出して初診カルテを作成", key="extract_multiple", type="primary", use_container_width=True):
            with st.spinner(f"AIが{len(uploaded_files)}個のファイルから情報を抽出中..."):
                extracted_data = extract_info_from_multiple_files(uploaded_files)
                
                if extracted_data:
                    st.success("✅ 情報抽出完了")
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

# 使い方
with st.expander("📖 複数ファイルアップロードの使い方"):
    st.markdown("""
    ### 複数ファイル対応の便利な使い方

    #### 📸 こんな場合に便利です
    
    1. **紹介状が複数ページに分かれている場合**
       - 1ページ目: 患者基本情報
       - 2ページ目: 検査結果
       - 3ページ目: 処方内容
       → すべてを一度にアップロードすれば、AIが統合して1つのカルテを作成

    2. **関連資料が複数ある場合**
       - 紹介状本体
       - 血液検査結果
       - 画像検査レポート
       - 心電図結果
       → 関連する全ての資料を一括処理

    3. **写真が複数枚に分かれている場合**
       - スマホで撮影した紹介状が複数枚
       - PDFと画像が混在
       → まとめてアップロードOK

    #### 🖱️ 複数選択の方法
    
    - **Windows**: Ctrl + クリック
    - **Mac**: Cmd + クリック
    - **連続選択**: Shift + クリック
    
    #### ⚠️ 注意事項
    
    - アップロードできるファイル形式: JPG, PNG, PDF
    - ファイルサイズは合計で20MB程度まで推奨
    - AIはすべてのファイルを読み込んで、情報を統合します
    - 同じ情報が複数回出現する場合、最も完全な情報が採用されます
    """)

# フッター
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <small>医療ハッカソン デモアプリ | 複数ファイル同時処理対応 | 実際の初診カルテ形式に準拠 | Powered by Google Gemini 2.5 Flash</small>
    </div>
    """,
    unsafe_allow_html=True
)