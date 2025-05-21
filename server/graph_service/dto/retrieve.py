from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from graph_service.dto.common import Message


class SearchQuery(BaseModel):
    group_ids: list[str] | None = Field(
        None, description='The group ids for the memories to search'
    )
    query: str
    max_facts: int = Field(default=10, description='The maximum number of facts to retrieve')
    center_node_uuid: str | None = Field(
        default=None, description='Optional UUID of a node to center the search around'
    )
    entity_filter: str | None = Field(
        default=None,
        description='Optional entity type to filter results (e.g., Preference, Procedure)',
    )


class BaseSearchQuery(BaseModel):
    group_ids: list[str] | None = Field(
        None, description='The group ids for the memories to search'
    )
    query: str
    center_node_uuid: str | None = Field(
        default=None, description='Optional UUID of a node to center the search around'
    )
    entity_filter: str | None = Field(
        default=None,
        description='Optional entity type to filter results (e.g., Preference, Procedure)',
    )


class FactSearchQuery(BaseSearchQuery):
    max_results: int = Field(default=10, description='The maximum number of facts to retrieve')


class NodeSearchQuery(BaseSearchQuery):
    max_results: int = Field(default=10, description='The maximum number of nodes to retrieve')


class FactResult(BaseModel):
    uuid: str
    name: str
    fact: str
    valid_at: datetime | None
    invalid_at: datetime | None
    created_at: datetime
    expired_at: datetime | None

    class Config:
        json_encoders = {datetime: lambda v: v.astimezone(timezone.utc).isoformat()}


class SearchResults(BaseModel):
    facts: list[FactResult]


class NodeSearchResultItem(BaseModel):
    uuid: str
    name: str
    summary: str | None = None
    labels: list[str] = []
    group_id: str
    created_at: datetime
    attributes: dict[str, Any] = {}

    class Config:
        json_encoders = {datetime: lambda v: v.astimezone(timezone.utc).isoformat()}


class SearchNodesResponse(BaseModel):
    nodes: list[NodeSearchResultItem]
    message: str | None = None


class GetMemoryRequest(BaseModel):
    group_id: str = Field(..., description='The group id of the memory to get')
    max_facts: int = Field(default=10, description='The maximum number of facts to retrieve')
    center_node_uuid: str | None = Field(
        ..., description='The uuid of the node to center the retrieval on'
    )
    messages: list[Message] = Field(
        ..., description='The messages to build the retrieval query from '
    )


class GetMemoryResponse(BaseModel):
    facts: list[FactResult] = Field(..., description='The facts that were retrieved from the graph')
