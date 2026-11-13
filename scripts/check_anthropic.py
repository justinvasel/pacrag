# scripts/check_anthropic.py
from anthropic import Anthropic
from pac_pipeline.config import settings

client = Anthropic(api_key=settings.anthropic_api_key)
msg = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=64,
    messages=[{"role": "user", "content": "Reply with exactly: ok"}],
)
print(msg.content[0].text)
