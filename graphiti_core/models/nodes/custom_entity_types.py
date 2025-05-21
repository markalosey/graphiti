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

This module defines Pydantic models for custom entity types within Graphiti.
These models are used to provide schema information and descriptions to the
entity extraction and classification processes.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class IdeaNodeSchema(BaseModel):
    """
    Represents a captured idea, concept, or undeveloped thought.
    Ideas are typically short pieces of text that can be tagged for future reference or development.
    """

    # Note: The actual 'content' of the idea is expected to be the 'name' field
    # of the base EntityNode, as Graphiti's extraction focuses on 'name' for the core text.
    # This model primarily defines additional, idea-specific attributes and provides a
    # description for the LLM to understand what an 'Idea' entity is.

    # If we wanted 'content' as a separate field specific to Idea, it would be defined here.
    # For now, we assume the main text of the idea will populate EntityNode.name.

    tags: Optional[List[str]] = Field(
        default_factory=list,
        description='A list of keywords or tags associated with the idea for categorization and searchability.',
        examples=['new_feature', 'refactor', 'marketing_campaign'],
    )

    # Example of another potential idea-specific field:
    # priority: Optional[int] = Field(None, description="A priority level for the idea (e.g., 1-5).")


# Example of how to define another custom entity type:
# class MeetingNoteNodeSchema(BaseModel):
#     """Represents notes taken during a meeting."""
#     attendees: Optional[List[str]] = Field(default_factory=list, description="List of attendees.")
#     action_items: Optional[List[str]] = Field(default_factory=list, description="Action items from the meeting.")
