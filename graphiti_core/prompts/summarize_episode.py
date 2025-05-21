"""
Prompts for episode summarization.
"""

import json
from typing import Any

from pydantic import BaseModel, Field

from .models import Message  # Assuming .models has Message


class EpisodeSummary(BaseModel):
    summary: str = Field(
        ...,
        description='A concise 1-3 sentence summary of the episode content, including key topics, actions, entities, and significant temporal information.',
    )


def create_summary(context: dict[str, Any]) -> list[Message]:
    system_prompt = (
        'You are an AI assistant that creates a concise summary of the provided episode content. '
        'The summary should be 1-3 sentences and capture the main topics, actions, key entities, '
        'and any significant temporal references or event timings mentioned.'
    )

    user_prompt_content = f"""
<EPISODE_NAME>
{context.get('episode_name', 'N/A')}
</EPISODE_NAME>

<EPISODE_CONTENT>
{context['episode_content']} 
</EPISODE_CONTENT>

Please generate a concise 1-3 sentence summary of the EPISODE_CONTENT above. 
Focus on:
- Main topics and actions.
- Key entities involved.
- Any explicitly mentioned dates, times, durations, or sequences of events that are central to the episode's meaning.
"""
    # Ensure episode_content is a string, not a list or other type before including in f-string
    # This was a source of error in other prompts if context['episode_content'] was not a string
    # However, for summarization, episode_content should be the actual text content.

    return [
        Message(role='system', content=system_prompt),
        Message(role='user', content=user_prompt_content),
    ]


# This would typically be part of a larger prompt_library structure
# For now, this file defines the prompt directly.
