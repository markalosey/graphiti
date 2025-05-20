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

import json
from typing import Any, Protocol, TypedDict
from datetime import datetime
from neo4j import time as neo4j_time

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion


# Helper to prepare context data for json.dumps by using model_dump(mode='json')
def _prepare_for_json(data: Any) -> Any:
    if isinstance(data, list):
        return [_prepare_for_json(item) for item in data]
    elif isinstance(data, dict):
        return {key: _prepare_for_json(value) for key, value in data.items()}
    elif hasattr(data, 'model_dump'):  # Check if it's a Pydantic model instance
        return data.model_dump(mode='json')
    elif (
        isinstance(data, neo4j_time.DateTime)
        or isinstance(data, neo4j_time.Date)
        or isinstance(data, neo4j_time.Time)
        or isinstance(data, neo4j_time.Duration)
    ):
        # For Neo4j temporal types, convert to native Python types then to ISO string
        native_dt = data.to_native() if hasattr(data, 'to_native') else data
        if hasattr(native_dt, 'isoformat'):
            return native_dt.isoformat()
        return str(native_dt)  # Fallback to string if no isoformat
    elif isinstance(data, datetime):  # Python datetime
        return data.isoformat()
    # Add other type conversions if necessary (e.g., Decimal, UUID)
    return data


class NodeDuplicate(BaseModel):
    id: int = Field(..., description='integer id of the entity')
    duplicate_idx: int = Field(
        ...,
        description='idx of the duplicate node. If no duplicate nodes are found, default to -1.',
    )
    name: str = Field(
        ...,
        description='Name of the entity. Should be the most complete and descriptive name possible.',
    )


class NodeResolutions(BaseModel):
    entity_resolutions: list[NodeDuplicate] = Field(..., description='List of resolved nodes')


class Prompt(Protocol):
    node: PromptVersion
    node_list: PromptVersion
    nodes: PromptVersion


class Versions(TypedDict):
    node: PromptFunction
    node_list: PromptFunction
    nodes: PromptFunction


def node(context: dict[str, Any]) -> list[Message]:  # For singular node deduplication
    sys_prompt = (
        'You are an AI assistant that determines if a NEW_ENTITY is a duplicate of any EXISTING_ENTITIES, '
        'considering the conversational context. The goal is to consolidate references to the same real-world object or concept.'
    )

    user_prompt_content = f"""
<CONTEXT_INFORMATION>
Previous Summaries: {json.dumps(_prepare_for_json(context.get('previous_episode_summaries', [])), indent=2)}
Current Message: {context.get('episode_content', '')}
</CONTEXT_INFORMATION>

<NEW_ENTITY_TO_EVALUATE>
{json.dumps(_prepare_for_json(context.get('extracted_node')), indent=2)}
</NEW_ENTITY_TO_EVALUATE>

<EXISTING_CANDIDATE_ENTITIES>
{json.dumps(_prepare_for_json(context.get('existing_nodes', [])), indent=2)}
</EXISTING_CANDIDATE_ENTITIES>

TASK: Compare NEW_ENTITY_TO_EVALUATE with each entity in EXISTING_CANDIDATE_ENTITIES.
- If NEW_ENTITY refers to the *same real-world object or concept* as an EXISTING_ENTITY, set 'duplicate_entity_id' to the ID of that existing entity.
- Otherwise, set 'duplicate_entity_id' to -1 (indicating it's a new, distinct entity).
- Also, provide the best consolidated 'name' for the entity (whether new or existing).

KEY_POINTS:
- Consider entity type descriptions (implicitly provided by schema) if available.
- Do NOT merge if related but distinct, or similar names for separate concepts.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt_content),
    ]


def nodes(context: dict[str, Any]) -> list[Message]:  # For plural node deduplication (batch)
    sys_prompt = (
        'You are an AI assistant. For each ENTITY provided (extracted from CURRENT_MESSAGE), '
        "determine if it's a duplicate of any of its listed DUPLICATION_CANDIDATES."
    )

    user_prompt_content = f"""
<CONTEXT_INFORMATION>
Previous Summaries: {json.dumps(_prepare_for_json(context.get('previous_episode_summaries', [])), indent=2)}
Current Message: {context.get('episode_content', '')}
</CONTEXT_INFORMATION>

<ENTITIES_FOR_RESOLUTION>
{json.dumps(_prepare_for_json(context.get('extracted_nodes')), indent=2)}
Each entity above has an 'id' (its temporary ID for this request) and a list of 'duplication_candidates' (potential existing matches from the graph, each with an 'idx').
</ENTITIES_FOR_RESOLUTION>

TASK: For each entity in ENTITIES_FOR_RESOLUTION:
- Output its original 'id'.
- Determine the best consolidated 'name'.
- Set 'duplicate_idx' to the 'idx' of a candidate if it's a true duplicate (same real-world concept).
- Set 'duplicate_idx' to -1 if it's not a duplicate of any listed candidates.

KEY_POINTS:
- Entities are duplicates if they refer to the *same real-world object or concept*.
- Do NOT mark as duplicates if related but distinct, or similar names for separate instances.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt_content),
    ]


def node_list(
    context: dict[str, Any],
) -> list[Message]:  # For deduplicating within a given list of nodes
    sys_prompt = 'You are an AI assistant that deduplicates a provided list of NODES.'

    user_prompt_content = f"""
<NODES_TO_DEDUPLICATE>
{json.dumps(_prepare_for_json(context.get('nodes')), indent=2)}
</NODES_TO_DEDUPLICATE>

TASK: Group nodes in NODES_TO_DEDUPLICATE that refer to the same real-world entity.

OUTPUT_FORMAT:
Respond with a JSON object: {{"nodes": [ {{"uuids": ["uuid1", "uuid2_if_duplicate_of_uuid1"], "summary": "Synthesized brief summary."}}, ... ]}}
- Each inner object represents a unique entity.
- 'uuids': lists all UUIDs from the input that map to this unique entity.
- 'summary': a new, synthesized summary for the unique entity.

KEY_POINTS:
- Every input UUID must appear in exactly one 'uuids' list in your output.
- If a node has no duplicates in the input list, its 'uuids' list will contain only itself.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt_content),
    ]


versions: Versions = {'node': node, 'node_list': node_list, 'nodes': nodes}
