import os
import json
from openai import OpenAI

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY", "MISSING_KEY"),
)

def analyze_sentiment(headline: str) -> dict:
    if not headline:
        return {"sentiment": "neutral", "confidence": 0.0, "reasoning": "No headline provided"}

    try:
        response = client.chat.completions.create(
            # The heavy hitter for deeper  gpt
            model="openai/gpt-oss-120b", 
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert financial analyst for Crude Oil (USOIL). "
                        "Analyze the sentiment of the following news headline from the perspective of a trader. "
                        "Reply ONLY with a JSON object containing three keys: "
                        "1. 'sentiment': 'positive', 'negative', or 'neutral'. "
                        "2. 'confidence': a float from 0.0 to 1.0. "
                        "3. 'reasoning': a brief one-sentence explanation of why this affects oil prices."
                    ),
                },
                {"role": "user", "content": headline},
            ],
            temperature=0.1,
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        print(f"Groq API call failed: {e}")
        return {"sentiment": "neutral", "confidence": 0.0, "reasoning": f"API Error: {str(e)}"}