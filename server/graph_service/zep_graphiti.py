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
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

from graphiti_core.nodes import EntityNode, EpisodicNode  # type: ignore

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
    ):  # Added embedder_client
        super().__init__(
            uri, user, password, llm_client=llm_client, embedder=embedder_client
        )  # Assumed 'embedder' kwarg

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
    logger.critical(
        f"CRITICAL_DEBUG: settings.model_name = '{settings.model_name}' (type: {type(settings.model_name)})"
    )
    logger.critical(
        f"CRITICAL_DEBUG: settings.embedding_name = '{settings.embedding_name}' (type: {type(settings.embedding_name)})"
    )
    logger.critical(
        f"CRITICAL_DEBUG: settings.openai_base_url = '{settings.openai_base_url}' (type: {type(settings.openai_base_url)})"
    )
    logger.critical(
        f'CRITICAL_DEBUG: settings.openai_api_key IS SET: {bool(settings.openai_api_key)}'
    )

    llm_client_instance: LLMClient | None = None
    if settings.model_name and settings.openai_base_url:
        llm_core_config = CoreLLMConfig(
            api_key=settings.openai_api_key or 'dummy-key',
            model=settings.model_name,
            base_url=settings.openai_base_url,
        )
        llm_client_instance = OpenAIClient(config=llm_core_config)
        logger.info(
            f'LLM Client configured for model: {settings.model_name} at {settings.openai_base_url}'
        )
    else:
        logger.warning(
            'LLM Client NOT configured due to missing model_name or openai_base_url. Custom entity extraction might be affected.'
        )

    embedder_client_instance: EmbedderClient | None = None
    if settings.embedding_name and settings.openai_base_url:
        embedder_core_config = OpenAIEmbedderConfig(
            api_key=settings.openai_api_key or 'dummy-key',
            model=settings.embedding_name,
            base_url=settings.openai_base_url,
        )
        embedder_client_instance = OpenAIEmbedder(config=embedder_core_config)
        logger.info(
            f'Embedder Client configured for model: {settings.embedding_name} at {settings.openai_base_url}'
        )
    else:
        logger.warning(
            'Embedder Client NOT configured due to missing embedding_name or openai_base_url. Embeddings will not be generated.'
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
    if settings.model_name and settings.openai_base_url:
        llm_core_config = CoreLLMConfig(
            api_key=settings.openai_api_key or 'dummy-key',
            model=settings.model_name,
            base_url=settings.openai_base_url,
        )
        llm_client_instance = OpenAIClient(config=llm_core_config)
    else:
        logger.info(
            'LLM Client not configured for initial index build (model_name or openai_base_url missing).'
        )

    embedder_client_instance: EmbedderClient | None = None
    if settings.embedding_name and settings.openai_base_url:
        embedder_core_config = OpenAIEmbedderConfig(
            api_key=settings.openai_api_key or 'dummy-key',
            model=settings.embedding_name,
            base_url=settings.openai_base_url,
        )
        embedder_client_instance = OpenAIEmbedder(config=embedder_core_config)
    else:
        logger.info(
            'Embedder Client not configured for initial index build (embedding_name or openai_base_url missing).'
        )

    temp_client_for_init = ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        llm_client=llm_client_instance,  # Pass even if None, Graphiti should handle
        embedder_client=embedder_client_instance,  # Pass even if None
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
