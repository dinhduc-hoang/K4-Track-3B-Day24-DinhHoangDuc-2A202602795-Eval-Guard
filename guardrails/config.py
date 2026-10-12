"""NeMo config hook: register a chat-completions provider for OpenAI-compatible APIs.

NeMo 0.10 maps engine "openai" to ChatOpenAI only when the model name contains
"gpt-3.5"/"gpt-4"; any other model (e.g. openai/gpt-oss-20b on Groq) falls back to the
text-completions class, which calls POST /completions — an endpoint Groq does not serve.
Registering ChatOpenAI under its own engine name keeps every self-check on /chat/completions.
NeMo executes init() before it initialises the LLM engines.
"""
from langchain_openai import ChatOpenAI
from nemoguardrails.llm.providers import register_llm_provider


def init(app):
    register_llm_provider("openai_chat", ChatOpenAI)
