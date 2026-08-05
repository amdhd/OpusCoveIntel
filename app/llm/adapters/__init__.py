"""Provider adapters.

CLAUDE.md 1.4: "No silent LLM spend. Every model call goes through
app/llm/router.py, which enforces the budget guard and cache. Direct
anthropic.Anthropic() / openai.OpenAI() calls outside app/llm/adapters/
are forbidden."

Each adapter exposes a narrow async interface. The router is the only consumer.
"""

from app.llm.adapters.anthropic import AnthropicAdapter
from app.llm.adapters.openai import OpenAIAdapter
from app.llm.adapters.qwen import QwenAdapter

__all__ = ["AnthropicAdapter", "OpenAIAdapter", "QwenAdapter"]
