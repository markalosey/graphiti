from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore


class Settings(BaseSettings):
    # OpenAI / Local Embedder (OpenAI compatible)
    openai_api_key: str | None = Field(None) # Can be dummy for local if no auth
    openai_base_url: str | None = Field(None) # For local embedder or OpenAI proxy
    openai_llm_model_name: str | None = Field(None) # If using OpenAI/Azure LLM
    embedding_name: str | None = Field(None) # For local/OpenAI embedder
    openai_embedding_dimensions: int | None = Field(None) # For local/OpenAI embedder

    # Anthropic
    anthropic_api_key: str | None = Field(None)
    anthropic_llm_model_name: str | None = Field(None)

    # Neo4j
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')


@lru_cache
def get_settings():
    return Settings()


ZepEnvDep = Annotated[Settings, Depends(get_settings)]
