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
from datetime import datetime
from time import time
from typing import Optional

from pydantic import BaseModel

from graphiti_core.edges import (
    CommunityEdge,
    EntityEdge,
    EpisodicEdge,
    create_entity_edge_embeddings,
)
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import MAX_REFLEXION_ITERATIONS, semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_edges import EdgeDuplicate, UniqueFacts
from graphiti_core.prompts.extract_edges import ExtractedEdges, MissingFacts
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import get_edge_invalidation_candidates, get_relevant_edges
from graphiti_core.utils.datetime_utils import ensure_utc, utc_now

logger = logging.getLogger(__name__)


def build_episodic_edges(
    entity_nodes: list[EntityNode],
    episode: EpisodicNode,
    created_at: datetime,
) -> list[EpisodicEdge]:
    episodic_edges: list[EpisodicEdge] = [
        EpisodicEdge(
            source_node_uuid=episode.uuid,
            target_node_uuid=node.uuid,
            created_at=created_at,
            group_id=episode.group_id,
        )
        for node in entity_nodes
    ]

    logger.debug(f'Built episodic edges: {episodic_edges}')

    return episodic_edges


def build_community_edges(
    entity_nodes: list[EntityNode],
    community_node: CommunityNode,
    created_at: datetime,
) -> list[CommunityEdge]:
    edges: list[CommunityEdge] = [
        CommunityEdge(
            source_node_uuid=community_node.uuid,
            target_node_uuid=node.uuid,
            created_at=created_at,
            group_id=community_node.group_id,
        )
        for node in entity_nodes
    ]

    return edges


async def extract_edges(
    clients: GraphitiClients,
    episode: EpisodicNode,
    nodes: list[EntityNode],
    previous_episodes: list[EpisodicNode],
    group_id: str = '',
    edge_types: dict[str, BaseModel] | None = None,
) -> list[EntityEdge]:
    start = time()

    extract_edges_max_tokens = 16384
    llm_client = clients.llm_client

    node_uuids_by_name_map = {node.name: node.uuid for node in nodes}

    edge_types_context = (
        [
            {
                'fact_type_name': type_name,
                'fact_type_description': type_model.__doc__,
            }
            for type_name, type_model in edge_types.items()
        ]
        if edge_types is not None
        else []
    )

    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'nodes': [node.name for node in nodes],
        'previous_episode_summaries': [
            ep.summary_text for ep in previous_episodes if ep.summary_text
        ],
        'reference_time': episode.valid_at,
        'edge_types': edge_types_context,
        'custom_prompt': '',
    }

    facts_missed = True
    reflexion_iterations = 0
    while facts_missed and reflexion_iterations <= MAX_REFLEXION_ITERATIONS:
        llm_response = await llm_client.generate_response(
            prompt_library.extract_edges['edge'](context),
            response_model=ExtractedEdges,
            max_tokens=extract_edges_max_tokens,
        )
        edges_data = llm_response.get('edges', [])

        context['extracted_facts'] = [edge_data.get('fact', '') for edge_data in edges_data]

        reflexion_iterations += 1
        if reflexion_iterations < MAX_REFLEXION_ITERATIONS:
            reflexion_response = await llm_client.generate_response(
                prompt_library.extract_edges['reflexion'](context),
                response_model=MissingFacts,
                max_tokens=extract_edges_max_tokens,
            )

            missing_facts = reflexion_response.get('missing_facts', [])

            custom_prompt = 'The following facts were missed in a previous extraction: '
            for fact in missing_facts:
                custom_prompt += f'\n{fact},'

            context['custom_prompt'] = custom_prompt

            facts_missed = len(missing_facts) != 0

    end = time()
    logger.debug(f'Extracted new edges: {edges_data} in {(end - start) * 1000} ms')

    if len(edges_data) == 0:
        return []

    # Convert the extracted data into EntityEdge objects
    edges = []
    for edge_data in edges_data:
        # Validate Edge Date information
        valid_at = edge_data.get('valid_at', None)
        invalid_at = edge_data.get('invalid_at', None)
        valid_at_datetime = None
        invalid_at_datetime = None

        if valid_at:
            try:
                valid_at_datetime = ensure_utc(
                    datetime.fromisoformat(valid_at.replace('Z', '+00:00'))
                )
            except ValueError as e:
                logger.warning(f'WARNING: Error parsing valid_at date: {e}. Input: {valid_at}')

        if invalid_at:
            try:
                invalid_at_datetime = ensure_utc(
                    datetime.fromisoformat(invalid_at.replace('Z', '+00:00'))
                )
            except ValueError as e:
                logger.warning(f'WARNING: Error parsing invalid_at date: {e}. Input: {invalid_at}')

        source_entity_name = edge_data.get('source_entity_name', '')
        target_entity_name = edge_data.get('target_entity_name', '')

        source_node_uuid = node_uuids_by_name_map.get(source_entity_name)
        target_node_uuid = node_uuids_by_name_map.get(target_entity_name)

        if not source_node_uuid:
            logger.warning(
                f"Skipping edge creation: Source entity name '{source_entity_name}' "
                f"from LLM (fact: '{edge_data.get('fact', '')}') not found in provided node list. "
                f'Available node names: {list(node_uuids_by_name_map.keys())}'
            )
            continue

        if not target_node_uuid:
            logger.warning(
                f"Skipping edge creation: Target entity name '{target_entity_name}' "
                f"from LLM (fact: '{edge_data.get('fact', '')}') not found in provided node list. "
                f"Source was '{source_entity_name}'. Available node names: {list(node_uuids_by_name_map.keys())}"
            )
            continue

        # If we reach here, both source and target UUIDs are valid (not None from .get())
        edge = EntityEdge(
            source_node_uuid=source_node_uuid,  # Now guaranteed to be a valid UUID
            target_node_uuid=target_node_uuid,  # Now guaranteed to be a valid UUID
            name=edge_data.get('relation_type', ''),
            group_id=group_id,
            fact=edge_data.get('fact', ''),
            episodes=[episode.uuid],
            created_at=utc_now(),
            valid_at=valid_at_datetime,
            invalid_at=invalid_at_datetime,
        )
        edges.append(edge)
        logger.debug(
            f'Created new edge: {edge.name} from (UUID: {edge.source_node_uuid}) to (UUID: {edge.target_node_uuid})'
        )

    logger.debug(f'Extracted edges: {[(e.name, e.uuid) for e in edges]}')

    return edges


async def dedupe_extracted_edges(
    llm_client: LLMClient,
    extracted_edges: list[EntityEdge],
    existing_edges: list[EntityEdge],
) -> list[EntityEdge]:
    # Create edge map
    edge_map: dict[str, EntityEdge] = {}
    for edge in existing_edges:
        edge_map[edge.uuid] = edge

    # Prepare context for LLM
    context = {
        'extracted_edges': [
            {'uuid': edge.uuid, 'name': edge.name, 'fact': edge.fact} for edge in extracted_edges
        ],
        'existing_edges': [
            {'uuid': edge.uuid, 'name': edge.name, 'fact': edge.fact} for edge in existing_edges
        ],
    }

    llm_response = await llm_client.generate_response(prompt_library.dedupe_edges['edge'](context))
    duplicate_data = llm_response.get('duplicates', [])
    logger.debug(f'Extracted unique edges: {duplicate_data}')

    duplicate_uuid_map: dict[str, str] = {}
    for duplicate in duplicate_data:
        uuid_value = duplicate['duplicate_of']
        duplicate_uuid_map[duplicate['uuid']] = uuid_value

    # Get full edge data
    edges: list[EntityEdge] = []
    for edge in extracted_edges:
        if edge.uuid in duplicate_uuid_map:
            existing_uuid = duplicate_uuid_map[edge.uuid]
            existing_edge = edge_map[existing_uuid]
            # Add current episode to the episodes list
            existing_edge.episodes += edge.episodes
            edges.append(existing_edge)
        else:
            edges.append(edge)

    return edges


async def resolve_extracted_edges(
    clients: GraphitiClients,
    extracted_edges: list[EntityEdge],
    episode: EpisodicNode,
    entities: list[EntityNode],
    edge_types: dict[str, BaseModel],
    edge_type_map: dict[tuple[str, str], list[str]],
) -> tuple[list[EntityEdge], list[EntityEdge]]:
    driver = clients.driver
    llm_client = clients.llm_client
    embedder = clients.embedder

    await create_entity_edge_embeddings(embedder, extracted_edges)

    search_results: tuple[list[list[EntityEdge]], list[list[EntityEdge]]] = await semaphore_gather(
        get_relevant_edges(driver, extracted_edges, SearchFilters()),
        get_edge_invalidation_candidates(driver, extracted_edges, SearchFilters(), 0.2),
    )

    related_edges_lists, edge_invalidation_candidates = search_results

    logger.debug(
        f'Related edges lists: {[(e.name, e.uuid) for edges_lst in related_edges_lists for e in edges_lst]}'
    )

    # Build entity hash table
    uuid_entity_map: dict[str, EntityNode] = {entity.uuid: entity for entity in entities}

    # Determine which edge types are relevant for each edge
    edge_types_lst: list[dict[str, BaseModel]] = []
    for extracted_edge in extracted_edges:
        source_node_labels = uuid_entity_map[extracted_edge.source_node_uuid].labels
        target_node_labels = uuid_entity_map[extracted_edge.target_node_uuid].labels
        label_tuples = [
            (source_label, target_label)
            for source_label in source_node_labels
            for target_label in target_node_labels
        ]

        extracted_edge_types = {}
        for label_tuple in label_tuples:
            type_names = edge_type_map.get(label_tuple, [])
            for type_name in type_names:
                type_model = edge_types.get(type_name)
                if type_model is None:
                    continue

                extracted_edge_types[type_name] = type_model

        edge_types_lst.append(extracted_edge_types)

    # resolve edges with related edges in the graph and find invalidation candidates
    results: list[tuple[EntityEdge, list[EntityEdge]]] = list(
        await semaphore_gather(
            *[
                resolve_extracted_edge(
                    llm_client,
                    extracted_edge,
                    related_edges,
                    existing_edges,
                    episode,
                    extracted_edge_types,
                )
                for extracted_edge, related_edges, existing_edges, extracted_edge_types in zip(
                    extracted_edges,
                    related_edges_lists,
                    edge_invalidation_candidates,
                    edge_types_lst,
                    strict=True,
                )
            ]
        )
    )

    resolved_edges: list[EntityEdge] = []
    invalidated_edges: list[EntityEdge] = []
    for result in results:
        resolved_edge = result[0]
        invalidated_edge_chunk = result[1]

        resolved_edges.append(resolved_edge)
        invalidated_edges.extend(invalidated_edge_chunk)

    logger.debug(f'Resolved edges: {[(e.name, e.uuid) for e in resolved_edges]}')

    await semaphore_gather(
        create_entity_edge_embeddings(embedder, resolved_edges),
        create_entity_edge_embeddings(embedder, invalidated_edges),
    )

    return resolved_edges, invalidated_edges


def resolve_edge_contradictions(
    resolved_edge: EntityEdge, invalidation_candidates: list[EntityEdge]
) -> list[EntityEdge]:
    if len(invalidation_candidates) == 0:
        return []

    # Determine which contradictory edges need to be expired
    invalidated_edges: list[EntityEdge] = []
    for edge in invalidation_candidates:
        # (Edge invalid before new edge becomes valid) or (new edge invalid before edge becomes valid)
        if (
            edge.invalid_at is not None
            and resolved_edge.valid_at is not None
            and edge.invalid_at <= resolved_edge.valid_at
        ) or (
            edge.valid_at is not None
            and resolved_edge.invalid_at is not None
            and resolved_edge.invalid_at <= edge.valid_at
        ):
            continue
        # New edge invalidates edge
        elif (
            edge.valid_at is not None
            and resolved_edge.valid_at is not None
            and edge.valid_at < resolved_edge.valid_at
        ):
            edge.invalid_at = resolved_edge.valid_at
            edge.expired_at = edge.expired_at if edge.expired_at is not None else utc_now()
            invalidated_edges.append(edge)

    return invalidated_edges


async def resolve_extracted_edge(
    llm_client: LLMClient,
    extracted_edge: EntityEdge,
    related_edges: list[EntityEdge],
    existing_edges: list[EntityEdge],
    episode: EpisodicNode,
    edge_types: dict[str, BaseModel] | None = None,
) -> tuple[EntityEdge, list[EntityEdge]]:
    if len(related_edges) == 0 and len(existing_edges) == 0:
        return extracted_edge, []

    start = time()

    # Prepare context for LLM
    related_edges_context = [
        {'id': edge.uuid, 'fact': edge.fact} for i, edge in enumerate(related_edges)
    ]

    invalidation_edge_candidates_context = [
        {'id': i, 'fact': existing_edge.fact} for i, existing_edge in enumerate(existing_edges)
    ]

    edge_types_context = (
        [
            {
                'fact_type_id': i,
                'fact_type_name': type_name,
                'fact_type_description': type_model.__doc__,
            }
            for i, (type_name, type_model) in enumerate(edge_types.items())
        ]
        if edge_types is not None
        else []
    )

    context = {
        'existing_edges': related_edges_context,
        'new_edge': extracted_edge.fact,
        'edge_invalidation_candidates': invalidation_edge_candidates_context,
        'edge_types': edge_types_context,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.dedupe_edges['resolve_edge'](context),
        response_model=EdgeDuplicate,
        model_size=ModelSize.small,
    )

    duplicate_fact_uuid: Optional[str] = llm_response.get('duplicate_fact_id')
    contradicted_fact_indices: list[int] = llm_response.get('contradicted_facts', [])
    resolved_fact_type: str = llm_response.get('fact_type', 'DEFAULT')

    edge: EntityEdge
    if duplicate_fact_uuid:
        found_duplicate = False
        for existing_edge_dict in context.get('existing_edges', []):
            if existing_edge_dict.get('id') == duplicate_fact_uuid:
                matching_related_edge = next(
                    (re for re in related_edges if re.uuid == duplicate_fact_uuid), None
                )
                if matching_related_edge:
                    edge = matching_related_edge
                    logger.info(
                        f'New edge fact "{extracted_edge.fact}" is a duplicate of existing edge UUID: {duplicate_fact_uuid}, Name: {edge.name}'
                    )
                    found_duplicate = True
                else:
                    logger.warning(
                        f'LLM identified duplicate UUID {duplicate_fact_uuid} but not found in provided related_edges objects. Treating as new.'
                    )
                    edge = extracted_edge
                break
        if not found_duplicate:
            logger.warning(
                f'LLM identified duplicate UUID {duplicate_fact_uuid} from prompt context, but it was not actionable with current related_edges objects. Treating new_edge as unique.'
            )
            edge = extracted_edge
    else:
        edge = extracted_edge
        logger.info(f'New edge fact "{edge.fact}" is considered unique.')

    edge.name = resolved_fact_type

    invalidated_edge_uuids: list[str] = []
    if existing_edges and contradicted_fact_indices:
        for idx in contradicted_fact_indices:
            if 0 <= idx < len(existing_edges):
                candidate_to_invalidate = existing_edges[idx]
                invalidated_uuid = candidate_to_invalidate.uuid
                if invalidated_uuid:
                    invalidated_edge_uuids.append(invalidated_uuid)
                    logger.info(
                        f'Edge UUID {invalidated_uuid} (Fact: "{candidate_to_invalidate.fact}") marked for invalidation due to contradiction by new edge fact "{extracted_edge.fact}".'
                    )
            else:
                logger.warning(f'LLM returned out-of-bounds contradicted_fact index: {idx}')

    end = time()
    logger.debug(
        f'Resolved edge: {extracted_edge.fact} -> {edge.fact} ({edge.name}) in {(end - start) * 1000} ms. Invalidated: {invalidated_edge_uuids}'
    )
    return edge, invalidated_edge_uuids


async def dedupe_extracted_edge(
    llm_client: LLMClient,
    extracted_edge: EntityEdge,
    related_edges: list[EntityEdge],
    episode: EpisodicNode | None = None,
) -> EntityEdge:
    if len(related_edges) == 0:
        return extracted_edge

    start = time()

    # Prepare context for LLM
    related_edges_context = [
        {'id': edge.uuid, 'fact': edge.fact} for i, edge in enumerate(related_edges)
    ]

    extracted_edge_context = {
        'fact': extracted_edge.fact,
    }

    context = {
        'related_edges': related_edges_context,
        'extracted_edges': extracted_edge_context,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.dedupe_edges['edge'](context),
        response_model=EdgeDuplicate,
        model_size=ModelSize.small,
    )

    duplicate_fact_id: int = llm_response.get('duplicate_fact_id', -1)

    edge = (
        related_edges[duplicate_fact_id]
        if 0 <= duplicate_fact_id < len(related_edges)
        else extracted_edge
    )

    if duplicate_fact_id >= 0 and episode is not None:
        edge.episodes.append(episode.uuid)

    end = time()
    logger.debug(
        f'Resolved Edge: {extracted_edge.name} is {edge.name}, in {(end - start) * 1000} ms'
    )

    return edge


async def dedupe_edge_list(
    llm_client: LLMClient,
    edges: list[EntityEdge],
) -> list[EntityEdge]:
    start = time()

    # Create edge map
    edge_map = {}
    for edge in edges:
        edge_map[edge.uuid] = edge

    # Prepare context for LLM
    context = {'edges': [{'uuid': edge.uuid, 'fact': edge.fact} for edge in edges]}

    llm_response = await llm_client.generate_response(
        prompt_library.dedupe_edges['edge_list'](context), response_model=UniqueFacts
    )
    unique_edges_data = llm_response.get('unique_facts', [])

    end = time()
    logger.debug(f'Extracted edge duplicates: {unique_edges_data} in {(end - start) * 1000} ms ')

    # Get full edge data
    unique_edges = []
    for edge_data in unique_edges_data:
        uuid = edge_data['uuid']
        edge = edge_map[uuid]
        edge.fact = edge_data['fact']
        unique_edges.append(edge)

    return unique_edges
