# 医療紹介状情報抽出アプリ

医療ハッカソンのデモアプリ - 紹介状の画像やテキストから患者情報を自動抽出

## 機能

- 📷 **画像アップロード**: スマホで撮影した紹介状の写真から情報を抽出
- 📝 **テキスト入力**: 電子カルテからコピーしたテキストから情報を抽出
- 🤖 **AI自動抽出**: Google Gemini 1.5 Flashを使用した高精度な情報抽出

## 抽出項目

- 患者氏名
- 生年月日
- 主訴
- 既往歴
- 服薬情報
- アレルギー
- ACP/患者の意向

## セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. Google Gemini APIキーの設定

1. [Google AI Studio](https://makersuite.google.com/app/apikey)でAPIキーを取得
2. `.streamlit/secrets.toml`ファイルを編集
3. `GOOGLE_API_KEY`に取得したAPIキーを設定

```toml
GOOGLE_API_KEY = "your-actual-api-key-here"
```

### 3. アプリの起動

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` が自動的に開きます。

## 使い方

### 画像から抽出

1. 「📷 画像アップロード」タブを選択
2. 紹介状の写真をアップロード
3. 「🔍 情報を抽出」ボタンをクリック
4. 右側に抽出結果が表示されます

### テキストから抽出

1. 「📝 テキスト入力」タブを選択
2. 紹介状のテキストを貼り付け
3. 「🔍 情報を抽出」ボタンをクリック
4. 右側に抽出結果が表示されます

## 技術スタック

- **Frontend**: Streamlit
- **AI Model**: Google Gemini API (gemini-1.5-flash)
- **Image Processing**: Pillow (PIL)
- **Data Display**: Pandas

## 注意事項

- このアプリはデモ用です。実際の医療現場での使用には適切なセキュリティ対策が必要です
- 個人情報の取り扱いには十分注意してください
- APIキーは外部に公開しないでください
