import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Force debug level for this logger

from typing import Annotated

from fastapi import Depends, HTTPException
from graphiti_core import Graphiti  # type: ignore
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.errors import (
    EdgeNotFoundError,
    GroupsEdgesNotFoundError,
    NodeNotFoundError,
)

# Core LLM and Embedder components
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.openai_client import (
    OpenAIClient,
    LLMConfig as CoreLLMConfig,
)  # Aliased to avoid Pydantic conflict if any
from graphiti_core.llm_client.anthropic_client import AnthropicClient  # Added import
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

from graphiti_core.nodes import EntityNode, EpisodicNode  # type: ignore

# Import the new IdeaNodeSchema
from graphiti_core.models.nodes.custom_entity_types import IdeaNodeSchema

from pydantic import BaseModel, Field  # For defining ENTITY_TYPES here for now

from graph_service.config import ZepEnvDep, Settings  # Import Settings directly too
from graph_service.dto import FactResult


# --- Define Custom Entity Types (Copied/adapted from old MCP server) ---
class Requirement(BaseModel):
    project_name: str = Field(
        ..., description='The name of the project to which the requirement belongs.'
    )
    description: str = Field(..., description='Description of the requirement.')


class Preference(BaseModel):
    category: str = Field(
        ..., description="The category of the preference. (e.g., 'Brands', 'Food', 'Music')"
    )
    description: str = Field(..., description='Brief description of the preference.')


class Procedure(BaseModel):
    description: str = Field(..., description='Brief description of the procedure.')


ENTITY_TYPES: dict[str, type[BaseModel]] = {  # Use type[BaseModel] for better type hinting
    'Requirement': Requirement,
    'Preference': Preference,
    'Procedure': Procedure,
    'Idea': IdeaNodeSchema,  # Add the new Idea entity type
}
# --- End Custom Entity Types ---


class ZepGraphiti(Graphiti):
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        llm_client: LLMClient | None = None,
        embedder_client: EmbedderClient | None = None,
    ):
        super().__init__(uri, user, password, llm_client=llm_client, embedder=embedder_client)

    async def save_entity_node(self, name: str, uuid: str, group_id: str, summary: str = ''):
        new_node = EntityNode(
            name=name,
            uuid=uuid,
            group_id=group_id,
            summary=summary,
        )
        if self.embedder:  # Graphiti base class should have self.embedder if initialized
            await new_node.generate_name_embedding(self.embedder)
        else:
            logger.warning(
                'Embedder not available in ZepGraphiti, skipping name embedding for save_entity_node.'
            )
        await new_node.save(self.driver)
        return new_node

    async def get_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            return edge
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_group(self, group_id: str):
        try:
            edges = await EntityEdge.get_by_group_ids(self.driver, [group_id])
        except GroupsEdgesNotFoundError:
            logger.warning(f'No edges found for group {group_id}')
            edges = []

        nodes = await EntityNode.get_by_group_ids(self.driver, [group_id])

        episodes = await EpisodicNode.get_by_group_ids(self.driver, [group_id])

        for edge in edges:
            await edge.delete(self.driver)

        for node in nodes:
            await node.delete(self.driver)

        for episode in episodes:
            await episode.delete(self.driver)

    async def delete_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            await edge.delete(self.driver)
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_episodic_node(self, uuid: str):
        try:
            episode = await EpisodicNode.get_by_uuid(self.driver, uuid)
            await episode.delete(self.driver)
        except NodeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e


# This is the dependency injector that FastAPI routes will use
async def get_graphiti(settings: ZepEnvDep):
    logger.critical('!!!!!!!!!!!! GET_GRAPHITI CALLED !!!!!!!!!!!!')
    # Log all relevant settings for debugging
    logger.critical(
        f"CRITICAL_DEBUG: settings.anthropic_llm_model_name = '{settings.anthropic_llm_model_name}'"
    )
    logger.critical(
        f'CRITICAL_DEBUG: settings.anthropic_api_key IS SET = {bool(settings.anthropic_api_key)}'
    )
    logger.critical(
        f"CRITICAL_DEBUG: settings.openai_llm_model_name = '{settings.openai_llm_model_name}'"
    )
    logger.critical(f"CRITICAL_DEBUG: settings.embedding_name = '{settings.embedding_name}'")
    logger.critical(
        f"CRITICAL_DEBUG: settings.openai_embedding_dimensions = '{settings.openai_embedding_dimensions}'"
    )
    logger.critical(f"CRITICAL_DEBUG: settings.openai_base_url = '{settings.openai_base_url}'")
    logger.critical(
        f'CRITICAL_DEBUG: settings.openai_api_key IS SET: {bool(settings.openai_api_key)}'
    )

    llm_client_instance: LLMClient | None = None

    if settings.anthropic_llm_model_name and settings.anthropic_api_key:
        llm_core_config = CoreLLMConfig(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_llm_model_name,
            # Anthropic client doesn't use base_url in the same way, typically direct.
            # max_tokens can be set here if needed, or rely on AnthropicClient defaults/docker-compose vars if added
        )
        llm_client_instance = AnthropicClient(config=llm_core_config)
        logger.critical(
            f'CRITICAL_LLM_CONFIG: Anthropic LLM Client configured using model: {llm_core_config.model}'
        )
    elif (
        settings.openai_llm_model_name
    ):  # Assuming openai_base_url might be optional if hitting OpenAI directly
        llm_core_config = CoreLLMConfig(
            api_key=settings.openai_api_key or 'dummy-key',  # OpenAI API key is crucial here
            model=settings.openai_llm_model_name,
            base_url=settings.openai_base_url,  # Could be None for direct OpenAI API
        )
        llm_client_instance = OpenAIClient(config=llm_core_config)
        logger.critical(
            f'CRITICAL_LLM_CONFIG: OpenAI LLM Client configured using model: {llm_core_config.model} at {llm_core_config.base_url or "OpenAI default"}'
        )
    else:
        logger.critical(
            f'CRITICAL_LLM_CONFIG_FAIL: LLM Client NOT configured. Neither Anthropic nor OpenAI LLM settings were sufficient. Custom entity extraction might be affected.'
        )

    embedder_client_instance: EmbedderClient | None = None
    if settings.embedding_name:  # openai_base_url is critical for local model
        embedder_core_config = OpenAIEmbedderConfig(
            api_key=settings.openai_api_key or 'dummy-key',  # Usually dummy for local endpoint
            embedding_model=settings.embedding_name,
            base_url=settings.openai_base_url,  # Essential for local model
            embedding_dim=settings.openai_embedding_dimensions,  # Corrected: embedding_dim, not dimensions
        )
        embedder_client_instance = OpenAIEmbedder(config=embedder_core_config)
        logger.critical(
            f'CRITICAL_EMBEDDER_CONFIG: Embedder Client configured using model: {embedder_core_config.embedding_model} at {embedder_core_config.base_url} with dims: {embedder_core_config.embedding_dim}'  # Corrected: embedding_dim
        )
    else:
        logger.critical(
            f'CRITICAL_EMBEDDER_CONFIG_FAIL: Embedder Client NOT configured due to missing embedding_name. Embeddings will not be generated.'
        )

    client = ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        llm_client=llm_client_instance,
        embedder_client=embedder_client_instance,
    )
    try:
        yield client
    finally:
        await client.close()


async def initialize_graphiti(settings: Settings):
    logger.critical('!!!!!!!!!!!! INITIALIZE_GRAPHITI CALLED !!!!!!!!!!!!')
    llm_client_instance: LLMClient | None = None

    if settings.anthropic_llm_model_name and settings.anthropic_api_key:
        llm_core_config = CoreLLMConfig(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_llm_model_name,
        )
        llm_client_instance = AnthropicClient(config=llm_core_config)
        logger.critical(f'LLM Client for init: Anthropic ({llm_core_config.model})')
    elif settings.openai_llm_model_name:
        llm_core_config = CoreLLMConfig(
            api_key=settings.openai_api_key or 'dummy-key',
            model=settings.openai_llm_model_name,
            base_url=settings.openai_base_url,
        )
        llm_client_instance = OpenAIClient(config=llm_core_config)
        logger.critical(
            f'LLM Client for init: OpenAI ({llm_core_config.model} at {llm_core_config.base_url or "OpenAI default"})'
        )
    else:
        logger.critical(
            'LLM Client not configured for initial index build (neither Anthropic nor OpenAI settings sufficient).'
        )

    embedder_client_instance: EmbedderClient | None = None
    if settings.embedding_name:
        embedder_core_config = OpenAIEmbedderConfig(
            api_key=settings.openai_api_key or 'dummy-key',
            embedding_model=settings.embedding_name,
            base_url=settings.openai_base_url,
            embedding_dim=settings.openai_embedding_dimensions,  # Corrected: embedding_dim, not dimensions
        )
        embedder_client_instance = OpenAIEmbedder(config=embedder_core_config)
        logger.critical(
            f'Embedder Client for init: {embedder_core_config.embedding_model} at {embedder_core_config.base_url} dims: {embedder_core_config.embedding_dim}'
        )  # Corrected: embedding_dim
    else:
        logger.critical(
            'Embedder Client not configured for initial index build (embedding_name missing).'
        )

    temp_client_for_init = ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        llm_client=llm_client_instance,
        embedder_client=embedder_client_instance,
    )
    try:
        logger.info('Building indices and constraints for Graphiti API service...')
        await temp_client_for_init.build_indices_and_constraints()
        logger.info('Indices and constraints built successfully.')
    finally:
        await temp_client_for_init.close()


def get_fact_result_from_edge(edge: EntityEdge):
    return FactResult(
        uuid=edge.uuid,
        name=edge.name,
        fact=edge.fact,
        valid_at=edge.valid_at,
        invalid_at=edge.invalid_at,
        created_at=edge.created_at,
        expired_at=edge.expired_at,
    )


ZepGraphitiDep = Annotated[ZepGraphiti, Depends(get_graphiti)]
