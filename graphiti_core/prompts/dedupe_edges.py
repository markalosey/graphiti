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
from typing import Any, Protocol, TypedDict, Optional

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion


class EdgeDuplicate(BaseModel):
    duplicate_fact_id: Optional[str] = Field(
        default=None,
        description='UUID of the duplicate fact. If no duplicate facts are found, this should be null or omitted.',
    )
    contradicted_facts: list[int] = Field(
        ...,
        description='List of ids of facts that should be invalidated. If no facts should be invalidated, the list should be empty.',
    )
    fact_type: str = Field(..., description='One of the provided fact types or DEFAULT')


class UniqueFact(BaseModel):
    uuid: str = Field(..., description='unique identifier of the fact')
    fact: str = Field(..., description='fact of a unique edge')


class UniqueFacts(BaseModel):
    unique_facts: list[UniqueFact]


class Prompt(Protocol):
    edge: PromptVersion
    edge_list: PromptVersion
    resolve_edge: PromptVersion


class Versions(TypedDict):
    edge: PromptFunction
    edge_list: PromptFunction
    resolve_edge: PromptFunction


def edge(context: dict[str, Any]) -> list[Message]:
    sys_prompt = 'You are an AI assistant that determines if a NEW_EDGE is a duplicate of any EXISTING_EDGES.'

    user_prompt_content = f"""
<NEW_EDGE_FACT_TO_EVALUATE>
{context.get('extracted_edges', {}).get('fact', '')} 
</NEW_EDGE_FACT_TO_EVALUATE>

<EXISTING_RELATED_EDGES>
{json.dumps(context.get('related_edges', []), indent=2)} 
</EXISTING_RELATED_EDGES>

TASK: Compare NEW_EDGE_FACT_TO_EVALUATE with each fact in EXISTING_RELATED_EDGES.
- If it's a duplicate (represents the same factual statement), set 'duplicate_fact_id' to the UUID of the existing edge.
- Otherwise, set 'duplicate_fact_id' to -1.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt_content),
    ]


def edge_list(context: dict[str, Any]) -> list[Message]:
    sys_prompt = "You are an AI assistant that deduplicates a provided list of EDGES based on their 'fact' text."

    user_prompt_content = f"""
<EDGES_TO_DEDUPLICATE>
{json.dumps(context.get('edges', []), indent=2)} # Each edge has 'uuid' and 'fact'
</EDGES_TO_DEDUPLICATE>

TASK: Identify unique facts from the EDGES_TO_DEDUPLICATE.

OUTPUT_FORMAT:
Respond with a JSON object: {{"unique_facts": [ {{"uuid": "uuid_of_chosen_representative_edge", "fact": "the_unique_fact_text"}} ]}}
- Each inner object represents one unique fact.
- 'uuid' should be the UUID of one of the input edges that best represents this unique fact.
- 'fact' should be the canonical text of that unique fact.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt_content),
    ]


def resolve_edge(context: dict[str, Any]) -> list[Message]:
    sys_prompt = (
        'You are an AI assistant that processes a NEW_EDGE. Determine if it duplicates an EXISTING_EDGE, '
        'identify any EDGE_INVALIDATION_CANDIDATES it contradicts, and classify its FACT_TYPE.'
    )

    user_prompt_content = f"""
<NEW_EDGE_FACT>
{context.get('new_edge', '')}
</NEW_EDGE_FACT>

<EXISTING_EDGES_FOR_DUPLICATION_CHECK>
{json.dumps(context.get('existing_edges', []), indent=2)} 
</EXISTING_EDGES_FOR_DUPLICATION_CHECK>

<EDGE_INVALIDATION_CANDIDATES>
{json.dumps(context.get('edge_invalidation_candidates', []), indent=2)}
</EDGE_INVALIDATION_CANDIDATES>

<AVAILABLE_FACT_TYPES>
{json.dumps(context.get('edge_types', []), indent=2)} # List of {'fact_type_id', 'fact_type_name', 'fact_type_description'}
</AVAILABLE_FACT_TYPES>

TASK:
1.  **Duplication Check**: If NEW_EDGE_FACT is a semantic duplicate of any fact in EXISTING_EDGES_FOR_DUPLICATION_CHECK, return the 'id' (UUID) of that existing edge as 'duplicate_fact_id'. Otherwise, omit 'duplicate_fact_id' or set its value to null.
2.  **Contradiction Check**: Identify IDs (indices from the input list) of any facts in EDGE_INVALIDATION_CANDIDATES that are directly contradicted by NEW_EDGE_FACT. Return these as a list of integers in 'contradicted_facts'.
3.  **Fact Classification**: Classify NEW_EDGE_FACT using one of the FACT_TYPE_NAMEs from AVAILABLE_FACT_TYPES. Return this as 'fact_type'. If no specific type fits well, use "DEFAULT".
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt_content),
    ]


versions: Versions = {
    'edge': edge,
    'resolve_edge': resolve_edge,
    'edge_list': edge_list,
}
