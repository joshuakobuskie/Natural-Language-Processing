import private_api_key
from google import genai

client = genai.Client(api_key=private_api_key.key)
response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Explain how AI works"
)

print(response.text)