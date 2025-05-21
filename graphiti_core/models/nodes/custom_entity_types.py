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
    The core title of the idea is expected to be stored in the base EntityNode's 'name' field.
    This schema defines additional, idea-specific attributes.
    Ideas are typically short pieces of text that can be tagged for future reference or development.
    The category might come from a top-right notation on a card, and details from the main body.
    """

    category: Optional[str] = Field(
        None,
        description="A general category for the idea (e.g., from top-right of an index card, like 'project nanoo').",
        examples=['project-nanoo', 'marketing', 'product-feature'],
    )

    details: Optional[str] = Field(
        None,
        description='The detailed content, notes, or bullet points describing the idea, supplementary to its main title/name.',
        examples=[
            'This involves using a prebuilt React component and integrating an LLM for prompt optimization.'
        ],
    )

    tags: Optional[List[str]] = Field(
        default_factory=list,
        description='A list of keywords or tags associated with the idea for categorization and searchability (e.g., from hashtags or keywords in the text).',
        examples=['llm-as-judge', 'react', 'prompt-optimizer'],
    )

    status: Optional[str] = Field(
        None,
        description='The current lifecycle status of the idea.',
        examples=['New', 'UnderConsideration', 'Accepted', 'Archived', 'Implemented'],
    )

    # Example of another potential idea-specific field:
    # priority: Optional[int] = Field(None, description="A priority level for the idea (e.g., 1-5).")


class Requirement(BaseModel):
    project_name: str = Field(
        ..., description='The name of the project to which the requirement belongs.'
    )
    description: str = Field(..., description='Description of the requirement.')


class Preference(BaseModel):
    category: str = Field(
        ..., description="The category of the preference. (e.g., 'Brands', 'Food', 'Music')"
    )
    description: str = Field(..., description='Brief description of the preference.')


class Procedure(BaseModel):
    description: str = Field(..., description='Brief description of the procedure.')


class CollectionNodeSchema(BaseModel):
    """
    Represents a collection or container for work items like Ideas and Tasks.
    The core title/name of the collection is expected to be stored in the base EntityNode's 'name' field.
    Collections can be nested to create hierarchies (e.g., Project -> Feature -> Sprint).
    Relationships (e.g., has_sub_collection, contains_idea, contains_task, member_of) are typically defined at the graph schema level.
    """

    description: Optional[str] = Field(
        None,
        description='A more detailed description of the collection and its purpose.',
        examples=[
            'A collection for all marketing efforts for the Q4 product launch.',
            'Container for all tasks related to the new API development.',
        ],
    )

    collection_type: Optional[str] = Field(
        None,
        description='The type of collection, e.g., Project, Feature, Epic, Sprint, IdeaPool, ActionPlan, Generic.',
        examples=['Project', 'Feature', 'IdeaPool', 'ActionPlan', 'Generic'],
    )

    status: Optional[str] = Field(
        None,
        description='The current status of the collection.',
        examples=['Active', 'Planning', 'OnHold', 'Completed', 'Archived'],
    )

    # Dates would typically be managed by base node properties or inferred from contents
    # start_date: Optional[str] = Field(None, description="The start date of the collection, if applicable.")
    # end_date: Optional[str] = Field(None, description="The end date or deadline for the collection, if applicable.")


class TaskNodeSchema(BaseModel):
    """
    Represents an actionable work item, potentially derived from an Idea or created directly.
    The core title of the task is expected to be stored in the base EntityNode's 'name' field.
    Relationships (e.g., derived_from_idea, member_of_collection, depends_on_task) are typically defined at the graph schema level.
    """

    full_description: Optional[str] = Field(
        None,
        description='A detailed description of the task, its goals, and any specific requirements.',
        examples=[
            'Develop the user authentication module including password reset functionality.',
            'Write documentation for the new billing API.',
        ],
    )

    status: str = Field(
        default='Backlog',
        description='The current workflow status of the task.',
        examples=['Backlog', 'InProgress', 'Blocked', 'InReview', 'Completed'],
    )

    priority: Optional[str] = Field(
        None,
        description='The priority level of the task.',
        examples=['High', 'Medium', 'Low'],
    )

    # Assignee would likely be a relationship to a User/Person entity, not a simple string field here.
    # assignee_id: Optional[str] = Field(None, description="Identifier for the user or agent assigned to this task.")

    # Dates would typically be managed by base node properties or specific date fields if ISO format is not an issue
    # due_date: Optional[str] = Field(None, description="The target completion date for the task.")
    # completion_date: Optional[str] = Field(None, description="The actual completion date of the task.")


# Example of how to define another custom entity type:
# class MeetingNoteNodeSchema(BaseModel):
#     """Represents notes taken during a meeting."""
#     attendees: Optional[List[str]] = Field(default_factory=list, description="List of attendees.")
#     action_items: Optional[List[str]] = Field(default_factory=list, description="Action items from the meeting.")
