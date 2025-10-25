from openai import OpenAI
import os
import time

# Set your API key as an environment variable or assign directly
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Initialize OpenAI client
client = OpenAI(
    api_key=OPENAI_API_KEY
)

config = {
    "model": "gpt-4o-mini",
    "frequency_penalty": 0,
    "presence_penalty": 0
}

RETRY_LIMIT = 5

def call_llm(prompt: str, temp: float = 0.8, top_p : float = 1.0, max_tokens: int = 16384, timeout: int = 60) -> str:
    """
    Sends a prompt to the OpenAI API endpoint and returns the generated text.
    Retries up to RETRY_LIMIT times on error.
    """
    for attempt in range(RETRY_LIMIT):
        try:
            response = client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=config["frequency_penalty"],
                presence_penalty=config["presence_penalty"],
                timeout=timeout
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < RETRY_LIMIT - 1:
                time.sleep(5)
                prompt += f"\n[Error encountered: {e}. Retrying... Attempt {attempt+2}/{RETRY_LIMIT}]"
            else:
                raise

# # For backward compatibility with the rest of the agent
# call_llm = call_model 