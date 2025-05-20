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


class ExtractedEntity(BaseModel):
    name: str = Field(..., description='Name of the extracted entity')
    entity_type_id: int = Field(
        description='ID of the classified entity type. '
        'Must be one of the provided entity_type_id integers.',
    )


class ExtractedEntities(BaseModel):
    extracted_entities: list[ExtractedEntity] = Field(..., description='List of extracted entities')


class MissedEntities(BaseModel):
    missed_entities: list[str] = Field(..., description="Names of entities that weren't extracted")


class EntityClassificationTriple(BaseModel):
    uuid: str = Field(description='UUID of the entity')
    name: str = Field(description='Name of the entity')
    entity_type: str | None = Field(
        default=None, description='Type of the entity. Must be one of the provided types or None'
    )


class EntityClassification(BaseModel):
    entity_classifications: list[EntityClassificationTriple] = Field(
        ..., description='List of entities classification triples.'
    )


class Prompt(Protocol):
    extract_message: PromptVersion
    extract_json: PromptVersion
    extract_text: PromptVersion
    reflexion: PromptVersion
    classify_nodes: PromptVersion
    extract_attributes: PromptVersion


class Versions(TypedDict):
    extract_message: PromptFunction
    extract_json: PromptFunction
    extract_text: PromptFunction
    reflexion: PromptFunction
    classify_nodes: PromptFunction
    extract_attributes: PromptFunction


def extract_message(context: dict[str, Any]) -> list[Message]:
    sys_prompt = (
        'You are an AI assistant that extracts and classifies entity nodes from conversational messages. '
        'Focus on the speaker and other significant entities in the CURRENT MESSAGE.'
    )

    # Condensed instructions, relying more on schema and model capability
    user_prompt_content = f"""
<PREVIOUS_EPISODE_SUMMARIES>
{json.dumps([s for s in context.get('previous_episode_summaries', [])], indent=2)}
</PREVIOUS_EPISODE_SUMMARIES>

<CURRENT_MESSAGE>
{context['episode_content']}
</CURRENT_MESSAGE>

<ENTITY_TYPES>
{context['entity_types']} 
</ENTITY_TYPES>

TASK: Extract entity nodes from CURRENT_MESSAGE. For each entity, provide its name and classify it using one of the provided ENTITY_TYPES (by entity_type_id).

KEY_POINTS:
- Always extract the speaker (before the colon ':') as the first entity.
- Extract significant entities, concepts, or actors mentioned in CURRENT_MESSAGE.
- Exclude entities only mentioned in PREVIOUS_EPISODE_SUMMARIES.
- Do NOT extract relationships, actions, dates, or times as entities.
- Provide full, unambiguous names for entities.

{context.get('custom_prompt', '')} 
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt_content),
    ]


def extract_json(context: dict[str, Any]) -> list[Message]:
    sys_prompt = (
        'You are an AI assistant that extracts and classifies relevant entity nodes from JSON content, '
        'guided by a source description.'
    )

    user_prompt_content = f"""
<SOURCE_DESCRIPTION>
{context['source_description']}
</SOURCE_DESCRIPTION>
<JSON_CONTENT>
{context['episode_content']}
</JSON_CONTENT>
<ENTITY_TYPES>
{context['entity_types']}
</ENTITY_TYPES>

TASK: From the JSON_CONTENT, extract relevant entities. Classify each using an entity_type_id from ENTITY_TYPES.

KEY_POINTS:
- Use SOURCE_DESCRIPTION for context.
- Extract entities the JSON represents (e.g., from "name" or "user" fields).
- Do NOT extract properties containing only dates.

{context.get('custom_prompt', '')}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt_content),
    ]


def extract_text(context: dict[str, Any]) -> list[Message]:
    sys_prompt = (
        'You are an AI assistant that extracts and classifies entity nodes from text. '
        'Focus on significant entities explicitly or implicitly mentioned.'
    )

    user_prompt_content = f"""
<TEXT_CONTENT>
{context['episode_content']}
</TEXT_CONTENT>
<ENTITY_TYPES>
{context['entity_types']}
</ENTITY_TYPES>

TASK: Extract entity nodes from TEXT_CONTENT. Classify each using an entity_type_id from ENTITY_TYPES.

KEY_POINTS:
- Extract significant entities, concepts, or actors.
- Do NOT extract relationships, actions, or temporal information (dates, times) as entities.
- Provide full, unambiguous names.

{context.get('custom_prompt', '')}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt_content),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = (
        'You are an AI assistant that identifies entities missed in a previous extraction pass.'
    )

    user_prompt_content = f"""
<PREVIOUS_EPISODE_SUMMARIES>
{json.dumps([s for s in context.get('previous_episode_summaries', [])], indent=2)}
</PREVIOUS_EPISODE_SUMMARIES>
<CURRENT_MESSAGE>
{context['episode_content']}
</CURRENT_MESSAGE>
<EXTRACTED_ENTITIES>
{context.get('extracted_entities', [])}
</EXTRACTED_ENTITIES>

TASK: Review the CURRENT_MESSAGE and PREVIOUS_EPISODE_SUMMARIES. Identify any significant entities from CURRENT_MESSAGE that are NOT in EXTRACTED_ENTITIES.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt_content),
    ]


def classify_nodes(context: dict[str, Any]) -> list[Message]:
    sys_prompt = 'You are an AI assistant that classifies given entity nodes based on provided context and entity types.'

    user_prompt_content = f"""
<PREVIOUS_EPISODE_SUMMARIES>
{json.dumps([s for s in context.get('previous_episode_summaries', [])], indent=2)}
</PREVIOUS_EPISODE_SUMMARIES>
<CURRENT_MESSAGE>
{context['episode_content']}
</CURRENT_MESSAGE>
<EXTRACTED_ENTITIES_TO_CLASSIFY>
{context['extracted_entities']}
</EXTRACTED_ENTITIES_TO_CLASSIFY>
<ENTITY_TYPES_FOR_CLASSIFICATION>
{context['entity_types']}
</ENTITY_TYPES_FOR_CLASSIFICATION>

TASK: For each entity in EXTRACTED_ENTITIES_TO_CLASSIFY, assign the most appropriate entity_type_id from ENTITY_TYPES_FOR_CLASSIFICATION.

KEY_POINTS:
- Each entity must have exactly one type.
- Only use the provided ENTITY_TYPES.
- If no type accurately classifies an entity, its type should be None (or handled by the Pydantic model if it allows null for type).
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt_content),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    sys_prompt = 'You are an AI assistant that extracts and updates entity properties (attributes and summary) based on provided text.'

    user_prompt_content = f"""
<CONTEXT_MESSAGES_AND_SUMMARIES>
Summaries of previous relevant episodes: {json.dumps([s for s in context.get('previous_episode_summaries', [])], indent=2)}
Content of current episode: {json.dumps(context['episode_content'], indent=2)}
</CONTEXT_MESSAGES_AND_SUMMARIES>

<ENTITY_TO_UPDATE>
Name: {context['node']['name']}
Existing Summary: {context['node']['summary']}
Existing Attributes: {context['node']['attributes']}
Entity Types: {context['node']['entity_types']}
</ENTITY_TO_UPDATE>

TASK: Based on CONTEXT_MESSAGES_AND_SUMMARIES, update the summary and extract other attributes for the ENTITY_TO_UPDATE.

KEY_POINTS:
1. Update summary with new information (max 250 words).
2. Only extract attribute values explicitly found in the context.
3. Refer to attribute descriptions (implicitly provided by the response_model schema) for what to extract.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt_content),
    ]


versions: Versions = {
    'extract_message': extract_message,
    'extract_json': extract_json,
    'extract_text': extract_text,
    'reflexion': reflexion,
    'classify_nodes': classify_nodes,
    'extract_attributes': extract_attributes,
}
