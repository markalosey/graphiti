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
from contextlib import suppress
from time import time
from typing import Any, List, Dict, Optional, Tuple
from uuid import uuid4
import re

import pydantic
from pydantic import BaseModel, Field
from neo4j.exceptions import ClientError

from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import MAX_REFLEXION_ITERATIONS, semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode, create_entity_node_embeddings
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_nodes import NodeDuplicate, NodeResolutions
from graphiti_core.prompts.extract_nodes import (
    ExtractedEntities,
    ExtractedEntity,
    MissedEntities,
)
from graphiti_core.search.search import search
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.pattern_library import get_all_patterns
from graphiti_core.utils.entity_extractor import RegexEntityExtractor, EntityMatch
from graphiti_core.llm_client.response_models import EpisodeSummary

logger = logging.getLogger(__name__)

MAX_DUPLICATION_CANDIDATES_PER_NODE = 3  # Reduced from 5
SUMMARY_TRUNCATION_LENGTH = 75  # Reduced from 150


async def extract_nodes_reflexion(
    llm_client: LLMClient,
    episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
    node_names: list[str],
) -> list[str]:
    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'previous_episode_summaries': [
            ep.summary_text for ep in previous_episodes if ep.summary_text
        ],
        'extracted_entities': node_names,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes['reflexion'](context), MissedEntities
    )
    missed_entities = llm_response.get('missed_entities', [])

    return missed_entities


async def extract_nodes(
    clients: GraphitiClients,
    episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
    entity_types: dict[str, BaseModel] | None = None,
) -> list[EntityNode]:
    start = time()
    llm_client = clients.llm_client
    llm_response = {}
    entities_missed = True
    reflexion_iterations = 0

    # Initialize the RegexEntityExtractor with patterns
    regex_extractor = RegexEntityExtractor(get_all_patterns())

    # Try regex-based extraction first
    regex_extracted_entities: List[EntityMatch] = []
    try:
        # Only attempt regex extraction if we have content to process
        if episode.content:
            logger.info('NODE_EXTRACTION: Attempting regex-based entity extraction')
            regex_extracted_entities = await regex_extractor.extract_entities(episode.content)

            # Filter out overlapping matches
            regex_extracted_entities = regex_extractor.filter_overlapping_matches(
                regex_extracted_entities
            )

            logger.info(
                f'NODE_EXTRACTION: Regex extraction found {len(regex_extracted_entities)} entities'
            )

            # If we found entities with regex, we can convert and return directly
            if regex_extracted_entities:
                # Convert regex matches to EntityNode objects
                extracted_nodes_list = []
                for match in regex_extracted_entities:
                    # Only include matches with confidence above threshold
                    if match.confidence >= 0.85:
                        # Map entity_type from regex patterns to Graphiti entity types
                        entity_type_name = map_regex_entity_type_to_graphiti(
                            match.entity_type, entity_types.keys() if entity_types else []
                        )

                        labels: list[str] = list({'Entity', str(entity_type_name)})

                        new_node = EntityNode(
                            name=match.name,
                            group_id=episode.group_id,
                            labels=labels,
                            summary='',
                            created_at=utc_now(),
                        )
                        extracted_nodes_list.append(new_node)
                        logger.debug(
                            f'Created new node from regex: {new_node.name} (UUID: {new_node.uuid})'
                        )

                # If we found sufficient entities, return them without LLM extraction
                if len(extracted_nodes_list) > 0:
                    logger.info(
                        f'NODE_EXTRACTION: Successfully extracted {len(extracted_nodes_list)} nodes using regex patterns'
                    )
                    end = time()
                    logger.debug(f'Extracted new nodes using regex in {(end - start) * 1000} ms')
                    return extracted_nodes_list
                else:
                    logger.info(
                        'NODE_EXTRACTION: Regex extraction found no high-confidence matches, falling back to LLM'
                    )
            else:
                logger.info(
                    'NODE_EXTRACTION: No entities found with regex extraction, falling back to LLM'
                )
    except Exception as e:
        logger.error(f'NODE_EXTRACTION: Error during regex extraction: {e}', exc_info=True)
        # Continue with LLM extraction as fallback

    # If regex extraction didn't return sufficient results, proceed with LLM extraction
    logger.info('NODE_EXTRACTION: Proceeding with LLM-based entity extraction')

    entity_types_context = [
        {
            'entity_type_id': 0,
            'entity_type_name': 'Entity',
            'entity_type_description': 'Default entity classification. Use this entity type if the entity is not one of the other listed types.',
        }
    ]
    entity_types_context += (
        [
            {
                'entity_type_id': i + 1,
                'entity_type_name': type_name,
                'entity_type_description': type_model.__doc__,
            }
            for i, (type_name, type_model) in enumerate(entity_types.items())
        ]
        if entity_types is not None
        else []
    )

    # Initial context for the first pass of extraction
    # Make a mutable copy for the loop if custom_prompt needs to be updated
    current_context = {
        'episode_content': episode.content,
        'episode_timestamp': episode.valid_at.isoformat(),
        'previous_episode_summaries': [],
        'custom_prompt': '',
        'entity_types': entity_types_context,
        'source_description': episode.source_description,
    }

    # Add any regex findings as hints to the LLM
    if regex_extracted_entities:
        regex_hints = [f'{match.name} ({match.entity_type})' for match in regex_extracted_entities]
        current_context['custom_prompt'] = (
            'Consider these potential entities from preliminary analysis: ' + ', '.join(regex_hints)
        )

    while entities_missed and reflexion_iterations <= MAX_REFLEXION_ITERATIONS:
        logger.info(f'NODE_EXTRACTION: Iteration {reflexion_iterations + 1}')
        logger.debug(
            f'NODE_EXTRACTION: Context for LLM call: {current_context}'
        )  # Log the context being sent

        if episode.source == EpisodeType.message:
            llm_response = await llm_client.generate_response(
                prompt_library.extract_nodes['extract_message'](current_context),
                response_model=ExtractedEntities,
            )
        elif episode.source == EpisodeType.text:
            llm_response = await llm_client.generate_response(
                prompt_library.extract_nodes['extract_text'](current_context),
                response_model=ExtractedEntities,
            )
        elif episode.source == EpisodeType.json:
            llm_response = await llm_client.generate_response(
                prompt_library.extract_nodes['extract_json'](current_context),
                response_model=ExtractedEntities,
            )
        else:
            logger.error(f'Unknown episode source type: {episode.source} for node extraction.')
            return []  # Or raise error

        logger.debug(f'NODE_EXTRACTION: LLM Response for entities: {llm_response}')

        extracted_entities_data = llm_response.get('extracted_entities', [])
        extracted_entities: list[ExtractedEntity] = []
        if isinstance(extracted_entities_data, list):
            for entity_data_from_llm in extracted_entities_data:
                if isinstance(entity_data_from_llm, dict):
                    try:
                        extracted_entities.append(ExtractedEntity(**entity_data_from_llm))
                    except Exception as e:
                        logger.error(
                            f'Error instantiating ExtractedEntity from LLM data: {entity_data_from_llm}, Error: {e}'
                        )
                else:
                    logger.warning(
                        f'LLM returned non-dict item in extracted_entities: {entity_data_from_llm}'
                    )
        else:
            logger.warning(
                f'LLM response for extracted_entities was not a list: {extracted_entities_data}'
            )

        reflexion_iterations += 1
        if (
            reflexion_iterations < MAX_REFLEXION_ITERATIONS
        ):  # Only do reflexion if enabled and not last iteration
            logger.info('NODE_EXTRACTION: Performing reflexion step.')
            missing_entities = await extract_nodes_reflexion(
                llm_client,
                episode,
                previous_episodes,
                [entity.name for entity in extracted_entities],
            )
            logger.info(
                f'NODE_EXTRACTION: Reflexion found {len(missing_entities)} missing entities: {missing_entities}'
            )

            entities_missed = len(missing_entities) != 0
            if entities_missed:
                current_context['custom_prompt'] = (
                    'Make sure that the following entities are extracted: '
                    + ', '.join(missing_entities)
                )
            else:
                current_context['custom_prompt'] = ''  # Clear custom prompt if no missed entities
        else:
            entities_missed = False  # Stop loop if max iterations reached or reflexion disabled

    filtered_extracted_entities = [entity for entity in extracted_entities if entity.name.strip()]
    end = time()
    logger.debug(f'Extracted new nodes: {filtered_extracted_entities} in {(end - start) * 1000} ms')
    extracted_nodes_list = []
    for extracted_entity in filtered_extracted_entities:
        entity_type_id_from_llm = extracted_entity.entity_type_id
        entity_type_name = 'Entity'
        if 0 <= entity_type_id_from_llm < len(entity_types_context):
            entity_type_name = entity_types_context[entity_type_id_from_llm].get(
                'entity_type_name', 'Entity'
            )
        else:
            logger.warning(
                f"Invalid entity_type_id {entity_type_id_from_llm} from LLM. Defaulting to 'Entity'."
            )

        labels: list[str] = list({'Entity', str(entity_type_name)})

        new_node = EntityNode(
            name=extracted_entity.name,
            group_id=episode.group_id,
            labels=labels,
            summary='',
            created_at=utc_now(),
        )
        extracted_nodes_list.append(new_node)
        logger.debug(f'Created new node: {new_node.name} (UUID: {new_node.uuid})')

    logger.debug(f'Extracted nodes: {[(n.name, n.uuid) for n in extracted_nodes_list]}')
    return extracted_nodes_list


async def dedupe_extracted_nodes(
    llm_client: LLMClient,
    extracted_nodes: list[EntityNode],
    existing_nodes: list[EntityNode],
) -> tuple[list[EntityNode], dict[str, str]]:
    start = time()

    # build existing node map
    node_map: dict[str, EntityNode] = {}
    for node in existing_nodes:
        node_map[node.uuid] = node

    # Prepare context for LLM
    existing_nodes_context = [
        {'uuid': node.uuid, 'name': node.name, 'summary': node.summary} for node in existing_nodes
    ]

    extracted_nodes_context = [
        {'uuid': node.uuid, 'name': node.name, 'summary': node.summary} for node in extracted_nodes
    ]

    context = {
        'existing_nodes': existing_nodes_context,
        'extracted_nodes': extracted_nodes_context,
    }

    llm_response = await llm_client.generate_response(prompt_library.dedupe_nodes['node'](context))

    duplicate_data = llm_response.get('duplicates', [])

    end = time()
    logger.debug(f'Deduplicated nodes: {duplicate_data} in {(end - start) * 1000} ms')

    uuid_map: dict[str, str] = {}
    for duplicate in duplicate_data:
        uuid_value = duplicate['duplicate_of']
        uuid_map[duplicate['uuid']] = uuid_value

    nodes: list[EntityNode] = []
    for node in extracted_nodes:
        if node.uuid in uuid_map:
            existing_uuid = uuid_map[node.uuid]
            existing_node = node_map[existing_uuid]
            nodes.append(existing_node)
        else:
            nodes.append(node)

    return nodes, uuid_map


async def resolve_extracted_nodes(
    clients: GraphitiClients,
    extracted_nodes: list[EntityNode],
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_types: dict[str, BaseModel] | None = None,
) -> tuple[list[EntityNode], dict[str, str]]:
    llm_client = clients.llm_client

    search_results: list[SearchResults] = await semaphore_gather(
        *[
            search(
                clients=clients,
                query=node.name,
                group_ids=[node.group_id],
                search_filter=SearchFilters(),
                config=NODE_HYBRID_SEARCH_RRF,
            )
            for node in extracted_nodes
        ]
    )

    existing_nodes_lists: list[list[EntityNode]] = [result.nodes for result in search_results]

    entity_types_dict: dict[str, BaseModel] = entity_types if entity_types is not None else {}

    extracted_nodes_context = []
    for i, node in enumerate(extracted_nodes):
        candidates_for_node = existing_nodes_lists[i][:MAX_DUPLICATION_CANDIDATES_PER_NODE]
        duplication_candidates_payload = []
        for j, candidate in enumerate(candidates_for_node):
            truncated_summary = (
                (candidate.summary[:SUMMARY_TRUNCATION_LENGTH] + '...')
                if candidate.summary and len(candidate.summary) > SUMMARY_TRUNCATION_LENGTH
                else candidate.summary
            )
            # Explicitly construct the candidate dictionary to ensure only desired fields are included
            candidate_data = {
                'idx': j,
                'name': candidate.name,
                'entity_types': candidate.labels,
                'summary': truncated_summary,
                # 'attributes': {k: v for k, v in candidate.attributes.items() if k not in ['name_embedding', 'uuid']} # Omit all attributes for now
            }
            duplication_candidates_payload.append(candidate_data)

        node_context_item = {
            'id': i,
            'name': node.name,
            'entity_type': node.labels,
            'entity_type_description': entity_types_dict.get(
                next((item for item in node.labels if item != 'Entity'), '')
            ).__doc__
            or 'Default Entity Type',
            'duplication_candidates': duplication_candidates_payload,
        }
        extracted_nodes_context.append(node_context_item)

    context = {
        'extracted_nodes': extracted_nodes_context,
        'episode_content': episode.content if episode is not None else '',
        'previous_episode_summaries': [],
    }

    llm_response = await llm_client.generate_response(
        prompt_library.dedupe_nodes['nodes'](context),
        response_model=NodeResolutions,
    )

    node_resolutions: list = llm_response.get('entity_resolutions', [])

    resolved_nodes: list[EntityNode] = []
    uuid_map: dict[str, str] = {}
    for resolution in node_resolutions:
        resolution_id = resolution.get('id', -1)
        duplicate_idx = resolution.get('duplicate_idx', -1)

        # Ensure resolution_id is within bounds of extracted_nodes
        if not (0 <= resolution_id < len(extracted_nodes)):
            logger.warning(f'Resolved node id {resolution_id} out of bounds.')
            continue  # Skip this resolution if id is invalid

        extracted_node = extracted_nodes[resolution_id]

        resolved_node = (
            existing_nodes_lists[resolution_id][duplicate_idx]
            if 0 <= duplicate_idx < len(existing_nodes_lists[resolution_id])
            else extracted_node
        )

        # Ensure resolved_node is not None before trying to access attributes
        if resolved_node:
            new_name = resolution.get('name')
            if new_name:
                resolved_node.name = new_name
            else:
                logger.warning(
                    f'LLM did not return a name for resolved node based on id {resolution_id}, keeping original: {resolved_node.name}'
                )

            resolved_nodes.append(resolved_node)
            uuid_map[extracted_node.uuid] = resolved_node.uuid
        else:
            logger.warning(
                f'Could not determine resolved_node for id {resolution_id} and duplicate_idx {duplicate_idx}. Original extracted node: {extracted_node.name}'
            )

    logger.debug(f'Resolved nodes: {[(n.name, n.uuid) for n in resolved_nodes]}')

    return resolved_nodes, uuid_map


async def resolve_extracted_node(
    llm_client: LLMClient,
    extracted_node: EntityNode,
    existing_nodes: list[EntityNode],
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_type: BaseModel | None = None,
) -> EntityNode:
    start = time()
    if len(existing_nodes) == 0:
        return extracted_node

    # Prepare context for LLM
    existing_nodes_context = [
        {
            **{
                'id': i,
                'name': node.name,
                'entity_types': node.labels,
            },
            **node.attributes,
        }
        for i, node in enumerate(existing_nodes)
    ]

    extracted_node_context = {
        'name': extracted_node.name,
        'entity_type': entity_type.__name__ if entity_type is not None else 'Entity',  # type: ignore
    }

    context = {
        'existing_nodes': existing_nodes_context,
        'extracted_node': extracted_node_context,
        'entity_type_description': entity_type.__doc__
        if entity_type is not None
        else 'Default Entity Type',
        'episode_content': episode.content if episode is not None else '',
        'previous_episode_summaries': [
            ep.summary_text
            for ep in previous_episodes
            if ep.summary_text and previous_episodes is not None
        ],
    }

    llm_response = await llm_client.generate_response(
        prompt_library.dedupe_nodes['node'](context),
        response_model=NodeDuplicate,
        model_size=ModelSize.small,
    )

    duplicate_id: int = llm_response.get('duplicate_node_id', -1)

    node = (
        existing_nodes[duplicate_id] if 0 <= duplicate_id < len(existing_nodes) else extracted_node
    )

    node.name = llm_response.get('name', '')

    end = time()
    logger.debug(
        f'Resolved node: {extracted_node.name} is {node.name}, in {(end - start) * 1000} ms'
    )

    return node


async def extract_attributes_from_nodes(
    clients: GraphitiClients,
    nodes: list[EntityNode],
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_types: dict[str, BaseModel] | None = None,
) -> list[EntityNode]:
    llm_client = clients.llm_client
    embedder = clients.embedder

    updated_nodes: list[EntityNode] = await semaphore_gather(
        *[
            extract_attributes_from_node(
                llm_client,
                node,
                episode,
                previous_episodes,
                entity_types.get(next((item for item in node.labels if item != 'Entity'), ''))
                if entity_types is not None
                else None,
            )
            for node in nodes
        ]
    )

    await create_entity_node_embeddings(embedder, updated_nodes)

    return updated_nodes


async def extract_attributes_from_node(
    llm_client: LLMClient,
    node: EntityNode,
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_type: BaseModel | None = None,
) -> EntityNode:
    node_context: dict[str, Any] = {
        'name': node.name,
        'summary': node.summary,
        'entity_types': node.labels,
        'attributes': node.attributes,
    }

    attributes_definitions: dict[str, Any] = {
        'summary': (
            str,
            Field(
                description='Summary containing the important information about the entity. Under 250 words',
            ),
        )
    }

    if entity_type is not None:
        for field_name, field_info in entity_type.model_fields.items():
            attributes_definitions[field_name] = (
                field_info.annotation,
                Field(description=field_info.description),
            )

    unique_model_name = f'EntityAttributes_{uuid4().hex}'
    entity_attributes_model = pydantic.create_model(unique_model_name, **attributes_definitions)

    summary_context: dict[str, Any] = {
        'node': node_context,
        'episode_content': episode.content if episode is not None else '',
        'previous_episode_summaries': [
            ep.summary_text
            for ep in previous_episodes
            if ep.summary_text and previous_episodes is not None
        ],
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes['extract_attributes'](summary_context),
        response_model=entity_attributes_model,
        model_size=ModelSize.small,
    )

    node.summary = llm_response.get('summary', node.summary)
    node_attributes = {key: value for key, value in llm_response.items()}

    with suppress(KeyError):
        del node_attributes['summary']

    node.attributes.update(node_attributes)

    return node


async def dedupe_node_list(
    llm_client: LLMClient,
    nodes: list[EntityNode],
) -> tuple[list[EntityNode], dict[str, str]]:
    start = time()

    # build node map
    node_map = {}
    for node in nodes:
        node_map[node.uuid] = node

    # Prepare context for LLM
    nodes_context = [{'uuid': node.uuid, 'name': node.name, **node.attributes} for node in nodes]

    context = {
        'nodes': nodes_context,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.dedupe_nodes['node_list'](context)
    )

    nodes_data = llm_response.get('nodes', [])

    end = time()
    logger.debug(f'Deduplicated nodes: {nodes_data} in {(end - start) * 1000} ms')

    # Get full node data
    unique_nodes = []
    uuid_map: dict[str, str] = {}
    for node_data in nodes_data:
        node_instance: EntityNode | None = node_map.get(node_data['uuids'][0])
        if node_instance is None:
            logger.warning(f'Node {node_data["uuids"][0]} not found in node map')
            continue
        node_instance.summary = node_data['summary']
        unique_nodes.append(node_instance)

        for uuid in node_data['uuids'][1:]:
            uuid_value = node_map[node_data['uuids'][0]].uuid
            uuid_map[uuid] = uuid_value

    return unique_nodes, uuid_map


def map_regex_entity_type_to_graphiti(
    regex_entity_type: str, available_entity_types: list[str]
) -> str:
    """
    Map entity type from regex patterns to available Graphiti entity types.

    Args:
        regex_entity_type: Entity type from regex pattern
        available_entity_types: List of available entity types in Graphiti

    Returns:
        Mapped entity type or 'Entity' as default
    """
    # Handle special EntityType and EntityName patterns from the pattern library
    if regex_entity_type == 'EntityType':
        return 'Entity'

    # Direct matches (case-sensitive)
    if regex_entity_type in available_entity_types:
        return regex_entity_type

    # Case-insensitive matches
    for available_type in available_entity_types:
        if regex_entity_type.lower() == available_type.lower():
            return available_type

    # Default to generic Entity
    return 'Entity'
