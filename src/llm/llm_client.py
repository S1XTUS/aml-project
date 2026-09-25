import os

from dotenv import load_dotenv
from openai import OpenAI

from src.utils.config_loader import load_config, project_path

# Keys can live in a git-ignored .env file at the project root
load_dotenv(project_path(".env"))

_client = None


def get_llm_client() -> OpenAI:
    """Shared DeepSeek (OpenAI-compatible) client, created on first use so
    modules import cleanly without DEEPSEEK_API_KEY set."""
    global _client
    if _client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Add it to the .env file in the project root "
                "(DEEPSEEK_API_KEY=sk-...) or set it in the environment, then restart the API."
            )
        _client = OpenAI(api_key=api_key, base_url=load_config()["llm"]["base_url"])
    return _client
