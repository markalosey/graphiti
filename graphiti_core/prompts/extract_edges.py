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

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion


class Edge(BaseModel):
    relation_type: str = Field(..., description='FACT_PREDICATE_IN_SCREAMING_SNAKE_CASE')
    source_entity_name: str = Field(..., description='The name of the source entity of the fact.')
    target_entity_name: str = Field(..., description='The name of the target entity of the fact.')
    fact: str = Field(..., description='')
    valid_at: str | None = Field(
        None,
        description='The date and time when the relationship described by the edge fact became true or was established. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ)',
    )
    invalid_at: str | None = Field(
        None,
        description='The date and time when the relationship described by the edge fact stopped being true or ended. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ)',
    )


class ExtractedEdges(BaseModel):
    edges: list[Edge]


class MissingFacts(BaseModel):
    missing_facts: list[str] = Field(..., description="facts that weren't extracted")


class Prompt(Protocol):
    edge: PromptVersion
    reflexion: PromptVersion
    extract_attributes: PromptVersion


class Versions(TypedDict):
    edge: PromptFunction
    reflexion: PromptFunction
    extract_attributes: PromptFunction


def edge(context: dict[str, Any]) -> list[Message]:
    sys_prompt = (
        'You are an expert fact extractor. Extract fact triples (subject-predicate-object) from CURRENT_MESSAGE. '
        'Include relevant date information (valid_at, invalid_at in ISO 8601 UTC) if explicitly tied to the fact. '
        'Use REFERENCE_TIME as current time. Focus on relationships between given ENTITIES.'
    )

    user_prompt_content = f"""
<PREVIOUS_EPISODE_SUMMARIES>
{json.dumps([s for s in context.get('previous_episode_summaries', [])], indent=2)}
</PREVIOUS_EPISODE_SUMMARIES>

<CURRENT_MESSAGE>
{context['episode_content']}
</CURRENT_MESSAGE>

<ENTITIES>
{json.dumps(context.get('nodes', []))} # List of entity names
</ENTITIES>

<REFERENCE_TIME>
{context['reference_time']}
</REFERENCE_TIME>

<FACT_TYPES_TO_CONSIDER>
{json.dumps(context.get('edge_types', []))} # List of {{'fact_type_name': ..., 'fact_type_description': ...}}
</FACT_TYPES_TO_CONSIDER>

TASK: Extract factual relationships from CURRENT_MESSAGE involving two distinct entities from the ENTITIES list.

KEY_POINTS:
- Output relation_type in SCREAMING_SNAKE_CASE.
- Ensure fact_text quotes or paraphrases source.
- For dates: ISO 8601 UTC (YYYY-MM-DDTHH:MM:SSZ). If ongoing, valid_at = REFERENCE_TIME. If ended, set invalid_at. Null if no explicit date for the fact itself.

{context.get('custom_prompt', '')}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt_content),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = 'You are an AI assistant that determines which facts have not been extracted from the given context.'

    user_prompt_content = f"""
<PREVIOUS_EPISODE_SUMMARIES>
{json.dumps([s for s in context.get('previous_episode_summaries', [])], indent=2)}
</PREVIOUS_EPISODE_SUMMARIES>
<CURRENT_MESSAGE>
{context['episode_content']}
</CURRENT_MESSAGE>
<EXTRACTED_ENTITIES>
{context.get('nodes', [])}
</EXTRACTED_ENTITIES>
<EXTRACTED_FACTS>
{context.get('extracted_facts', [])}
</EXTRACTED_FACTS>

TASK: Review CURRENT_MESSAGE and related context. Identify any factual relationships between EXTRACTED_ENTITIES that were NOT in EXTRACTED_FACTS.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt_content),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    sys_prompt = (
        'You are an AI assistant that extracts attributes for a given FACT based on a MESSAGE.'
    )
    user_prompt_content = f"""
<MESSAGE>
{json.dumps(context['episode_content'], indent=2)}
</MESSAGE>
<REFERENCE_TIME>
{context['reference_time']}
</REFERENCE_TIME>
<FACT_TO_ENRICH>
{context['fact']}
</FACT_TO_ENRICH>

TASK: Based on the MESSAGE and REFERENCE_TIME, extract additional attributes for the FACT_TO_ENRICH, as defined by the expected response schema (which includes date fields like valid_at, invalid_at).
KEY_POINTS:
- Only use information from MESSAGE.
- Adhere to ISO 8601 UTC for any dates.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt_content),
    ]


versions: Versions = {
    'edge': edge,
    'reflexion': reflexion,
    'extract_attributes': extract_attributes,
}
