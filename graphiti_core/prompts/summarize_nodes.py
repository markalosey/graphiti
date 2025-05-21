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


class Summary(BaseModel):
    summary: str = Field(
        ...,
        description='Summary containing the important information about the entity. Under 250 words',
    )


class SummaryDescription(BaseModel):
    description: str = Field(..., description='One sentence description of the provided summary')


class Prompt(Protocol):
    summarize_pair: PromptVersion
    summarize_context: PromptVersion
    summary_description: PromptVersion


class Versions(TypedDict):
    summarize_pair: PromptFunction
    summarize_context: PromptFunction
    summary_description: PromptFunction


def summarize_pair(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that combines summaries.',
        ),
        Message(
            role='user',
            content=f"""
        Synthesize the information from the following two summaries into a single succinct summary.
        
        Summaries must be under 250 words.

        Summaries:
        {json.dumps(context['node_summaries'], indent=2)}
        """,
        ),
    ]


def summarize_context(context: dict[str, Any]) -> list[Message]:
    sys_prompt = (
        "You are an AI assistant that creates/updates an ENTITY's summary and extracts its attributes "
        'based on contextual messages and existing entity information.'
    )

    user_prompt_content = f"""
<CONTEXT_MESSAGES_AND_SUMMARIES>
Previous Episode Summaries: {json.dumps(context.get('previous_episode_summaries', []), indent=2)}
Current Episode Content: {json.dumps(context['episode_content'], indent=2)}
</CONTEXT_MESSAGES_AND_SUMMARIES>

<ENTITY_TO_UPDATE>
Name: {context['node_name']}
Existing Summary (for context): {context['node_summary']}
Existing Attributes (for context): {json.dumps(context.get('attributes', {}), indent=2)} 
</ENTITY_TO_UPDATE>

TASK: Based on CONTEXT_MESSAGES_AND_SUMMARIES and the ENTITY_TO_UPDATE's existing information:
1. Create an updated, concise summary for the ENTITY (max 250 words), incorporating new relevant information.
2. Extract values for any explicitly defined attributes (schema for attributes is implicitly part of the expected response_model).

KEY_POINTS:
- Focus on information directly relevant to the ENTITY_TO_UPDATE from the provided messages/summaries.
- Do not hallucinate values; if information for an attribute isn't present, it should not be included or should be null based on schema.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt_content),
    ]


def summary_description(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that describes provided contents in a single sentence.',
        ),
        Message(
            role='user',
            content=f"""
        Create a short one sentence description of the summary that explains what kind of information is summarized.
        Summaries must be under 250 words.

        Summary:
        {json.dumps(context['summary'], indent=2)}
        """,
        ),
    ]


versions: Versions = {
    'summarize_pair': summarize_pair,
    'summarize_context': summarize_context,
    'summary_description': summary_description,
}
