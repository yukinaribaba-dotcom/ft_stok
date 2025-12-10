import google.generativeai as genai

# APIキーの設定
genai.configure(api_key="AIzaSyAeM3hzeEFBCftHbenf1Y2vBR8AHeSLEYc")

print("--- 利用可能なモデル一覧 ---")
for m in genai.list_models():
    # テキスト生成（generateContent）に対応しているモデルだけを表示
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
