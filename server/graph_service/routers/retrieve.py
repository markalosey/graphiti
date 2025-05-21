from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, status
from neo4j.time import DateTime as Neo4jDateTime

from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_config import SearchConfig, SearchResults as CoreSearchResults
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

from graph_service.dto import (
    GetMemoryRequest,
    GetMemoryResponse,
    Message,
    FactSearchQuery,
    NodeSearchQuery,
    SearchResults,
    NodeSearchResultItem,
    SearchNodesResponse,
)
from graph_service.zep_graphiti import ZepGraphitiDep, get_fact_result_from_edge

router = APIRouter()


def sanitize_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    sanitized = {}
    if attributes:
        for key, value in attributes.items():
            if isinstance(value, Neo4jDateTime):
                sanitized[key] = value.to_native().astimezone(timezone.utc).isoformat()
            elif isinstance(value, datetime):
                sanitized[key] = value.astimezone(timezone.utc).isoformat()
            else:
                sanitized[key] = value
    return sanitized


@router.post('/search', status_code=status.HTTP_200_OK, response_model=SearchResults)
async def search_facts_endpoint(query: FactSearchQuery, graphiti: ZepGraphitiDep):
    search_filters = None
    if query.entity_filter:
        search_filters = SearchFilters(node_labels=[query.entity_filter])

    relevant_edges = await graphiti.search(
        group_ids=query.group_ids,
        query=query.query,
        num_results=query.max_results,
        center_node_uuid=query.center_node_uuid,
        search_filter=search_filters,
    )
    facts = [get_fact_result_from_edge(edge) for edge in relevant_edges]
    return SearchResults(
        facts=facts,
    )


@router.post('/search-nodes', status_code=status.HTTP_200_OK, response_model=SearchNodesResponse)
async def search_nodes_endpoint(query: NodeSearchQuery, graphiti: ZepGraphitiDep):
    search_filters = None
    if query.entity_filter:
        search_filters = SearchFilters(node_labels=[query.entity_filter])

    core_search_results: CoreSearchResults = await graphiti.search_(
        group_ids=query.group_ids,
        query=query.query,
        center_node_uuid=query.center_node_uuid,
        search_filter=search_filters,
        config=NODE_HYBRID_SEARCH_RRF.model_copy(update={'limit': query.max_results}),
    )

    nodes_payload = [
        NodeSearchResultItem(
            uuid=node.uuid,
            name=node.name,
            summary=node.summary,
            labels=node.labels,
            group_id=node.group_id,
            created_at=node.created_at,
            attributes=sanitize_attributes(node.attributes),
        )
        for node in core_search_results.nodes
    ]
    return SearchNodesResponse(nodes=nodes_payload, message='Nodes retrieved successfully')


@router.get('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def get_entity_edge(uuid: str, graphiti: ZepGraphitiDep):
    entity_edge = await graphiti.get_entity_edge(uuid)
    return get_fact_result_from_edge(entity_edge)


@router.get('/episodes/{group_id}', status_code=status.HTTP_200_OK)
async def get_episodes(group_id: str, last_n: int, graphiti: ZepGraphitiDep):
    episodes = await graphiti.retrieve_episodes(
        group_ids=[group_id], last_n=last_n, reference_time=datetime.now(timezone.utc)
    )
    return episodes


@router.post('/get-memory', status_code=status.HTTP_200_OK)
async def get_memory(
    request: GetMemoryRequest,
    graphiti: ZepGraphitiDep,
):
    combined_query = compose_query_from_messages(request.messages)
    result = await graphiti.search(
        group_ids=[request.group_id],
        query=combined_query,
        num_results=request.max_facts,
    )
    facts = [get_fact_result_from_edge(edge) for edge in result]
    return GetMemoryResponse(facts=facts)


def compose_query_from_messages(messages: list[Message]):
    combined_query = ''
    for message in messages:
        combined_query += f'{message.role_type or ""}({message.role or ""}): {message.content}\n'
    return combined_query
