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

from typing import Any, Protocol, TypedDict
import json

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion


class EdgeDates(BaseModel):
    valid_at: str | None = Field(
        None,
        description='The date and time when the relationship described by the edge fact became true or was established. YYYY-MM-DDTHH:MM:SS.SSSSSSZ or null.',
    )
    invalid_at: str | None = Field(
        None,
        description='The date and time when the relationship described by the edge fact stopped being true or ended. YYYY-MM-DDTHH:MM:SS.SSSSSSZ or null.',
    )


class Prompt(Protocol):
    v1: PromptVersion


class Versions(TypedDict):
    v1: PromptFunction


def v1(context: dict[str, Any]) -> list[Message]:
    sys_prompt = (
        'You are an AI assistant that extracts specific datetime information (valid_at, invalid_at) for a given FACT, '
        'based on context from a CURRENT_EPISODE and PREVIOUS_EPISODE_SUMMARIES. Use REFERENCE_TIMESTAMP as current time.'
    )

    user_prompt_content = f"""
<PREVIOUS_EPISODE_SUMMARIES>
{json.dumps([s for s in context.get('previous_episode_summaries', [])], indent=2)}
</PREVIOUS_EPISODE_SUMMARIES>

<CURRENT_EPISODE>
{context['current_episode']}
</CURRENT_EPISODE>

<REFERENCE_TIMESTAMP>
{context['reference_timestamp']}
</REFERENCE_TIMESTAMP>
            
<FACT_TO_ANALYZE>
{context['edge_fact']}
</FACT_TO_ANALYZE>

TASK: Analyze FACT_TO_ANALYZE. Extract valid_at and invalid_at datetimes if explicitly stated or clearly implied by CURRENT_EPISODE or PREVIOUS_EPISODE_SUMMARIES as relating *directly* to when the FACT itself became true or ceased to be true.

KEY_POINTS:
- Output datetimes in ISO 8601 UTC format (YYYY-MM-DDTHH:MM:SSZ).
- Use REFERENCE_TIMESTAMP to resolve relative times (e.g., "yesterday", "2 hours ago") for the fact's validity.
- If fact is present tense / ongoing, valid_at is REFERENCE_TIMESTAMP.
- If no specific temporal information for the fact's validity/invalidity, leave fields null.
- Do NOT infer dates from unrelated events.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt_content),
    ]


versions: Versions = {'v1': v1}
