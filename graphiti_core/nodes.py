"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Force debug level

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from time import time
from typing import Any
from uuid import uuid4

from neo4j import AsyncDriver
from pydantic import BaseModel, Field
from typing_extensions import LiteralString

from graphiti_core.embedder import EmbedderClient
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.helpers import DEFAULT_DATABASE
from graphiti_core.models.nodes.node_db_queries import (
    COMMUNITY_NODE_SAVE,
    ENTITY_NODE_SAVE,
    EPISODIC_NODE_SAVE,
)
from graphiti_core.utils.datetime_utils import utc_now

logger = logging.getLogger(__name__)

ENTITY_NODE_RETURN: LiteralString = """
        RETURN
            n.uuid As uuid, 
            n.name AS name,
            n.group_id AS group_id,
            n.created_at AS created_at, 
            n.summary AS summary,
            labels(n) AS labels,
            properties(n) AS attributes
            """


class EpisodeType(Enum):
    """
    Enumeration of different types of episodes that can be processed.

    This enum defines the various sources or formats of episodes that the system
    can handle. It's used to categorize and potentially handle different types
    of input data differently.

    Attributes:
    -----------
    message : str
        Represents a standard message-type episode. The content for this type
        should be formatted as "actor: content". For example, "user: Hello, how are you?"
        or "assistant: I'm doing well, thank you for asking."
    json : str
        Represents an episode containing a JSON string object with structured data.
    text : str
        Represents a plain text episode.
    """

    message = 'message'
    json = 'json'
    text = 'text'

    @staticmethod
    def from_str(episode_type: str):
        if episode_type == 'message':
            return EpisodeType.message
        if episode_type == 'json':
            return EpisodeType.json
        if episode_type == 'text':
            return EpisodeType.text
        logger.error(f'Episode type: {episode_type} not implemented')
        raise NotImplementedError


class Node(BaseModel, ABC):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(description='name of the node')
    group_id: str = Field(description='partition of the graph')
    labels: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: utc_now())

    @abstractmethod
    async def save(self, driver: AsyncDriver): ...

    async def delete(self, driver: AsyncDriver):
        result = await driver.execute_query(
            """
        MATCH (n:Entity|Episodic|Community {uuid: $uuid})
        DETACH DELETE n
        """,
            uuid=self.uuid,
            database_=DEFAULT_DATABASE,
        )

        logger.debug(f'Deleted Node: {self.uuid}')

        return result

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.uuid == other.uuid
        return False

    @classmethod
    async def delete_by_group_id(cls, driver: AsyncDriver, group_id: str):
        await driver.execute_query(
            """
        MATCH (n:Entity|Episodic|Community {group_id: $group_id})
        DETACH DELETE n
        """,
            group_id=group_id,
            database_=DEFAULT_DATABASE,
        )

        return 'SUCCESS'

    @classmethod
    async def get_by_uuid(cls, driver: AsyncDriver, uuid: str): ...

    @classmethod
    async def get_by_uuids(cls, driver: AsyncDriver, uuids: list[str]): ...


class EpisodicNode(Node):
    source: EpisodeType = Field(description='The source of the episode.')
    source_description: str = Field(description='A description of the episode source.')
    content: str = Field(description='The content of the episode.')
    valid_at: datetime = Field(
        description='datetime of when the original document was created',
    )
    entity_edges: list[str] = Field(
        default_factory=list, description='A list of entity edge uuids for this episode.'
    )
    summary_text: str | None = Field(
        default=None, description='A concise summary of the episode content.'
    )

    async def save(self, driver: AsyncDriver):
        result = await driver.execute_query(
            EPISODIC_NODE_SAVE,
            uuid=self.uuid,
            name=self.name,
            group_id=self.group_id,
            source_description=self.source_description,
            content=self.content,
            entity_edges=self.entity_edges,
            created_at=self.created_at,
            valid_at=self.valid_at,
            source=self.source.value,
            database_=DEFAULT_DATABASE,
        )

        logger.debug(f'Saved Node to neo4j: {self.uuid}')

        return result

    @classmethod
    async def get_by_uuid(cls, driver: AsyncDriver, uuid: str):
        records, _, _ = await driver.execute_query(
            """
        MATCH (e:Episodic {uuid: $uuid})
            RETURN e.content AS content,
            e.created_at AS created_at,
            e.valid_at AS valid_at,
            e.uuid AS uuid,
            e.name AS name,
            e.group_id AS group_id,
            e.source_description AS source_description,
            e.source AS source,
            e.entity_edges AS entity_edges
        """,
            uuid=uuid,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        episodes = [get_episodic_node_from_record(record) for record in records]

        if len(episodes) == 0:
            raise NodeNotFoundError(uuid)

        return episodes[0]

    @classmethod
    async def get_by_uuids(cls, driver: AsyncDriver, uuids: list[str]):
        records, _, _ = await driver.execute_query(
            """
        MATCH (e:Episodic) WHERE e.uuid IN $uuids
            RETURN DISTINCT
            e.content AS content,
            e.created_at AS created_at,
            e.valid_at AS valid_at,
            e.uuid AS uuid,
            e.name AS name,
            e.group_id AS group_id,
            e.source_description AS source_description,
            e.source AS source,
            e.entity_edges AS entity_edges
        """,
            uuids=uuids,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        episodes = [get_episodic_node_from_record(record) for record in records]

        return episodes

    @classmethod
    async def get_by_group_ids(
        cls,
        driver: AsyncDriver,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ):
        cursor_query: LiteralString = 'AND e.uuid < $uuid' if uuid_cursor else ''
        limit_query: LiteralString = 'LIMIT $limit' if limit is not None else ''

        records, _, _ = await driver.execute_query(
            """
        MATCH (e:Episodic) WHERE e.group_id IN $group_ids
        """
            + cursor_query
            + """
            RETURN DISTINCT
            e.content AS content,
            e.created_at AS created_at,
            e.valid_at AS valid_at,
            e.uuid AS uuid,
            e.name AS name,
            e.group_id AS group_id,
            e.source_description AS source_description,
            e.source AS source,
            e.entity_edges AS entity_edges
        ORDER BY e.uuid DESC
        """
            + limit_query,
            group_ids=group_ids,
            uuid=uuid_cursor,
            limit=limit,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        episodes = [get_episodic_node_from_record(record) for record in records]

        return episodes

    @classmethod
    async def get_by_entity_node_uuid(cls, driver: AsyncDriver, entity_node_uuid: str):
        records, _, _ = await driver.execute_query(
            """
        MATCH (e:Episodic)-[r:MENTIONS]->(n:Entity {uuid: $entity_node_uuid})
            RETURN DISTINCT
            e.content AS content,
            e.created_at AS created_at,
            e.valid_at AS valid_at,
            e.uuid AS uuid,
            e.name AS name,
            e.group_id AS group_id,
            e.source_description AS source_description,
            e.source AS source,
            e.entity_edges AS entity_edges
        """,
            entity_node_uuid=entity_node_uuid,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        episodes = [get_episodic_node_from_record(record) for record in records]

        return episodes


class EntityNode(Node):
    name_embedding: list[float] | None = Field(default=None, description='embedding of the name')
    summary: str = Field(description='regional summary of surrounding edges', default_factory=str)
    attributes: dict[str, Any] = Field(
        default={}, description='Additional attributes of the node. Dependent on node labels'
    )

    async def generate_name_embedding(self, embedder: EmbedderClient):
        logger.critical(f'ENTITY_NODE: Attempting to generate name_embedding for: {self.name}')
        start = time()
        text = self.name.replace('\n', ' ')
        try:
            embedding_result = await embedder.create(input_data=[text])
            logger.critical(f"ENTITY_NODE: Embedding result for '{text}': {embedding_result}")
            if embedding_result:
                self.name_embedding = embedding_result
                logger.critical(f'ENTITY_NODE: Successfully set name_embedding for: {self.name}')
            else:
                logger.critical(f'ENTITY_NODE: Embedding result was None or empty for: {self.name}')
        except Exception as e:
            logger.critical(
                f"ENTITY_NODE: Error during embedder.create for '{text}': {str(e)}", exc_info=True
            )
            self.name_embedding = None  # Ensure it's None if error
        end = time()
        logger.critical(
            f'ENTITY_NODE: Time for embedding "{text}": {end - start} ms, Final name_embedding: {self.name_embedding is not None}'
        )
        return self.name_embedding

    async def load_name_embedding(self, driver: AsyncDriver):
        query: LiteralString = """
            MATCH (n:Entity {uuid: $uuid})
            RETURN n.name_embedding AS name_embedding
        """
        records, _, _ = await driver.execute_query(
            query, uuid=self.uuid, database_=DEFAULT_DATABASE, routing_='r'
        )

        if len(records) == 0:
            raise NodeNotFoundError(self.uuid)

        self.name_embedding = records[0]['name_embedding']

    async def save(self, driver: AsyncDriver):
        entity_data: dict[str, Any] = {
            'uuid': self.uuid,
            'name': self.name,
            'name_embedding': self.name_embedding,
            'group_id': self.group_id,
            'summary': self.summary,
            'created_at': self.created_at,
        }

        entity_data.update(self.attributes or {})

        result = await driver.execute_query(
            ENTITY_NODE_SAVE,
            labels=self.labels + ['Entity'],
            entity_data=entity_data,
            database_=DEFAULT_DATABASE,
        )

        logger.debug(f'Saved Node to neo4j: {self.uuid}')

        return result

    @classmethod
    async def get_by_uuid(cls, driver: AsyncDriver, uuid: str):
        query = (
            """
                                                        MATCH (n:Entity {uuid: $uuid})
                                                        """
            + ENTITY_NODE_RETURN
        )
        records, _, _ = await driver.execute_query(
            query,
            uuid=uuid,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        nodes = [get_entity_node_from_record(record) for record in records]

        if len(nodes) == 0:
            raise NodeNotFoundError(uuid)

        return nodes[0]

    @classmethod
    async def get_by_uuids(cls, driver: AsyncDriver, uuids: list[str]):
        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Entity) WHERE n.uuid IN $uuids
        """
            + ENTITY_NODE_RETURN,
            uuids=uuids,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        nodes = [get_entity_node_from_record(record) for record in records]

        return nodes

    @classmethod
    async def get_by_group_ids(
        cls,
        driver: AsyncDriver,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ):
        cursor_query: LiteralString = 'AND n.uuid < $uuid' if uuid_cursor else ''
        limit_query: LiteralString = 'LIMIT $limit' if limit is not None else ''

        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Entity) WHERE n.group_id IN $group_ids
        """
            + cursor_query
            + ENTITY_NODE_RETURN
            + """
        ORDER BY n.uuid DESC
        """
            + limit_query,
            group_ids=group_ids,
            uuid=uuid_cursor,
            limit=limit,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        nodes = [get_entity_node_from_record(record) for record in records]

        return nodes


class CommunityNode(Node):
    name_embedding: list[float] | None = Field(default=None, description='embedding of the name')
    summary: str = Field(description='region summary of member nodes', default_factory=str)

    async def save(self, driver: AsyncDriver):
        result = await driver.execute_query(
            COMMUNITY_NODE_SAVE,
            uuid=self.uuid,
            name=self.name,
            group_id=self.group_id,
            summary=self.summary,
            name_embedding=self.name_embedding,
            created_at=self.created_at,
            database_=DEFAULT_DATABASE,
        )

        logger.debug(f'Saved Node to neo4j: {self.uuid}')

        return result

    async def generate_name_embedding(self, embedder: EmbedderClient):
        logger.critical(
            f'COMMUNITY_NODE: Attempting to generate name_embedding for: {self.name}'
        )  # Added for community
        start = time()
        text = self.name.replace('\n', ' ')
        try:
            embedding_result = await embedder.create(input_data=[text])
            logger.critical(f"COMMUNITY_NODE: Embedding result for '{text}': {embedding_result}")
            if embedding_result:
                self.name_embedding = embedding_result
                logger.critical(f'COMMUNITY_NODE: Successfully set name_embedding for: {self.name}')
            else:
                logger.critical(
                    f'COMMUNITY_NODE: Embedding result was None or empty for: {self.name}'
                )
        except Exception as e:
            logger.critical(
                f"COMMUNITY_NODE: Error during embedder.create for '{text}': {str(e)}",
                exc_info=True,
            )
            self.name_embedding = None
        end = time()
        logger.critical(
            f'COMMUNITY_NODE: Time for embedding "{text}": {end - start} ms, Final name_embedding: {self.name_embedding is not None}'
        )
        return self.name_embedding

    async def load_name_embedding(self, driver: AsyncDriver):
        query: LiteralString = """
            MATCH (c:Community {uuid: $uuid})
            RETURN c.name_embedding AS name_embedding
        """
        records, _, _ = await driver.execute_query(
            query, uuid=self.uuid, database_=DEFAULT_DATABASE, routing_='r'
        )

        if len(records) == 0:
            raise NodeNotFoundError(self.uuid)

        self.name_embedding = records[0]['name_embedding']

    @classmethod
    async def get_by_uuid(cls, driver: AsyncDriver, uuid: str):
        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Community {uuid: $uuid})
        RETURN
            n.uuid As uuid, 
            n.name AS name,
            n.group_id AS group_id,
            n.created_at AS created_at, 
            n.summary AS summary
        """,
            uuid=uuid,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        nodes = [get_community_node_from_record(record) for record in records]

        if len(nodes) == 0:
            raise NodeNotFoundError(uuid)

        return nodes[0]

    @classmethod
    async def get_by_uuids(cls, driver: AsyncDriver, uuids: list[str]):
        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Community) WHERE n.uuid IN $uuids
        RETURN
            n.uuid As uuid, 
            n.name AS name,
            n.group_id AS group_id,
            n.created_at AS created_at, 
            n.summary AS summary
        """,
            uuids=uuids,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        communities = [get_community_node_from_record(record) for record in records]

        return communities

    @classmethod
    async def get_by_group_ids(
        cls,
        driver: AsyncDriver,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ):
        cursor_query: LiteralString = 'AND n.uuid < $uuid' if uuid_cursor else ''
        limit_query: LiteralString = 'LIMIT $limit' if limit is not None else ''

        records, _, _ = await driver.execute_query(
            """
        MATCH (n:Community) WHERE n.group_id IN $group_ids
        """
            + cursor_query
            + """
            RETURN
            n.uuid As uuid, 
            n.name AS name,
            n.group_id AS group_id,
            n.created_at AS created_at, 
            n.summary AS summary
        ORDER BY n.uuid DESC
        """
            + limit_query,
            group_ids=group_ids,
            uuid=uuid_cursor,
            limit=limit,
            database_=DEFAULT_DATABASE,
            routing_='r',
        )

        communities = [get_community_node_from_record(record) for record in records]

        return communities


# Node helpers
def get_episodic_node_from_record(record: Any) -> EpisodicNode:
    return EpisodicNode(
        uuid=record['uuid'],
        name=record['name'],
        group_id=record['group_id'],
        source_description=record['source_description'],
        content=record['content'],
        source=record['source'],
        entity_edges=record['entity_edges'],
        created_at=record['created_at'].to_native()
        if hasattr(record['created_at'], 'to_native')
        else record['created_at'],
        valid_at=record['valid_at'].to_native()
        if hasattr(record['valid_at'], 'to_native')
        else record['valid_at'],
        summary_text=record.get('summary_text'),
    )


def get_entity_node_from_record(record: Any) -> EntityNode:
    created_at_val = record['created_at']
    created_at_native = (
        created_at_val.to_native() if hasattr(created_at_val, 'to_native') else created_at_val
    )

    name_embedding_val = record.get('name_embedding')
    if name_embedding_val is None and record.get('attributes'):
        name_embedding_val = record['attributes'].get('name_embedding')

    return EntityNode(
        uuid=record['uuid'],
        name=record['name'],
        group_id=record['group_id'],
        summary=record['summary'],
        labels=record['labels'],
        attributes=record['attributes'],
        created_at=created_at_native,
        name_embedding=name_embedding_val,
    )


def get_community_node_from_record(record: Any) -> CommunityNode:
    created_at_val = record['created_at']
    created_at_native = (
        created_at_val.to_native() if hasattr(created_at_val, 'to_native') else created_at_val
    )

    name_embedding_val = record.get('name_embedding')
    if name_embedding_val is None and record.get('attributes'):
        name_embedding_val = record['attributes'].get('name_embedding')

    return CommunityNode(
        uuid=record['uuid'],
        name=record['name'],
        group_id=record['group_id'],
        summary=record['summary'],
        created_at=created_at_native,
        name_embedding=name_embedding_val,
    )


async def create_entity_node_embeddings(embedder: EmbedderClient, nodes: list[EntityNode]):
    """Generate name embeddings for a list of entity nodes in place."""
    texts_to_embed = [node.name.replace('\n', ' ') for node in nodes if node.name_embedding is None]
    if not texts_to_embed:
        return

    logger.critical(
        f'BULK_NODE_EMBED: Generating embeddings for {len(texts_to_embed)} node names via create_batch.'
    )
    list_of_embeddings = await embedder.create_batch(texts_to_embed)

    embed_idx = 0
    for node in nodes:
        if node.name_embedding is None:
            if embed_idx < len(list_of_embeddings):
                node.name_embedding = list_of_embeddings[embed_idx]
                logger.critical(f"BULK_NODE_EMBED: Set name_embedding for node '{node.name}'")
                embed_idx += 1
            else:
                logger.error(
                    f"BULK_NODE_EMBED: Mismatch in embedding results for node '{node.name}'"
                )
