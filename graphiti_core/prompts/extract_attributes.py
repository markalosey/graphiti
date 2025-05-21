"""
Prompts for extracting specific attributes for an already identified entity node,
based on its type and the provided episode context.
"""

from typing import Any, Dict, List

from .models import Message, PromptFunction, PromptVersion


def create_default_prompt_messages(context: Dict[str, Any]) -> List[Message]:
    """
    Creates prompt messages for extracting attributes for a given entity node.

    The context is expected to contain:
    - episode_content: The text of the current episode.
    - previous_episode_summaries: List of summaries of relevant prior episodes.
    - node: A dictionary with info about the current node being processed, including:
        - name: The name of the node.
        - summary: Any existing summary for the node.
        - entity_types: List of labels for the node (e.g., ["Entity", "Idea"])
        - attributes: Existing attributes of the node.
    - attributes_definitions: A dictionary where keys are attribute names to extract,
      and values are their descriptions (e.g., from Pydantic Field descriptions).
      Example: {
          "tags": "A list of keywords or tags...",
          "category": "A general category for the idea...",
          "details": "The detailed content, notes, or bullet points..."
      }
    """
    system_message_content = (
        'You are an AI assistant specialized in extracting specific attributes for an entity '
        "from provided text content. Focus on the entity described in the 'node' context. "
        "Use the 'episode_content' and 'previous_episode_summaries' for broader context. "
        "Your goal is to populate the attributes listed in 'attributes_definitions'."
    )

    # Constructing a clear user prompt
    attributes_to_extract_str = '\n'.join(
        f'- {name}: {description}'
        for name, description in context.get('attributes_definitions', {}).items()
    )

    user_message_content = f"""
Contextual Information:
<EPISODE_CONTENT>
{context.get('episode_content', '')}
</EPISODE_CONTENT>

<PREVIOUS_EPISODE_SUMMARIES>
{context.get('previous_episode_summaries', [])}
</PREVIOUS_EPISODE_SUMMARIES>

Entity to Process:
<NODE_NAME>
{context.get('node', {}).get('name', 'N/A')}
</NODE_NAME>
<NODE_SUMMARY>
{context.get('node', {}).get('summary', '')}
</NODE_SUMMARY>
<NODE_ENTITY_TYPES>
{context.get('node', {}).get('entity_types', [])}
</NODE_ENTITY_TYPES>
<NODE_EXISTING_ATTRIBUTES>
{context.get('node', {}).get('attributes', {})}
</NODE_EXISTING_ATTRIBUTES>

Attributes to Extract (Format: attribute_name: attribute_description):
{attributes_to_extract_str}

Task:
Carefully analyze the EPISODE_CONTENT to find and extract values for the attributes listed above. The NODE_NAME is the primary title or essence of the entity.
- For 'category' (if the entity is an Idea): Look for an overarching project, topic, or theme mentioned in EPISODE_CONTENT.
- For 'details' (if the entity is an Idea): Extract the main descriptive sentences or bullet points from EPISODE_CONTENT that elaborate on the NODE_NAME. This should be supplementary to the NODE_NAME.
- For 'tags' (if the entity is an Idea or potentially other types): Identify explicit hashtags (e.g., #example) or a few very specific keywords from EPISODE_CONTENT that are good for categorizing or searching for this entity.
- For 'status' (if the entity is a Task, Idea, or Collection): Look for explicit mentions of its current state (e.g., "Status: InProgress", "is New", "currently Active"). Choose from typical statuses like Backlog, InProgress, Blocked, InReview, Completed (for Tasks); New, UnderConsideration, Accepted, Archived, Implemented (for Ideas); Active, Planning, OnHold, Completed, Archived (for Collections).
- For 'priority' (if the entity is a Task): Look for terms indicating urgency like "High priority", "critical task", "Low priority".
- For 'collection_type' (if the entity is a Collection): Identify its functional type like "Project", "Feature", "Epic", "Sprint", "IdeaPool", "ActionPlan", or "Generic" if not specified.
- For 'full_description' (if the entity is a Task): Extract the detailed description of the task.
- For 'description' (if the entity is a Collection): Extract the detailed description of the collection.
- If an attribute's value is not clearly found in EPISODE_CONTENT, omit the attribute from your response or set its value to null, unless its definition implies a default (e.g., an empty list for tags if none are found).

Respond with a JSON object containing ONLY the extracted attributes and their values.
For list-based attributes like 'tags', ensure the value is a list of strings (e.g., ["tag1", "tag2"]).

Example for an 'Idea' entity:
{{
  "category": "project nanoo",
  "details": "This idea involves creating a new ML model to predict customer churn using a prebuilt component.",
  "tags": ["machine-learning", "new-product", "churn-prediction"],
  "status": "UnderConsideration"
}}

Example for a 'Task' entity:
{{
  "full_description": "Design the new user onboarding flow, including all screens and interaction points.",
  "status": "Backlog",
  "priority": "High"
}}

Example for a 'Collection' entity:
{{
  "description": "All design assets and specifications for the Alpha release.",
  "collection_type": "Project",
  "status": "Active"
}}

If only tags are found:
{{
  "tags": ["internal-tool"]
}}
If no specific attributes are found for this entity based on the definitions and content:
{{}}
"""

    return [
        Message(role='system', content=system_message_content),
        Message(role='user', content=user_message_content),
    ]


# The function itself conforms to the PromptVersion protocol (which is just a Callable)
# and also to PromptFunction type alias.
default_extract_attributes_prompt_v1: PromptFunction = create_default_prompt_messages

# Create a dictionary to hold different versions of attribute extraction prompts.
# The values must be functions that match the PromptFunction signature.
versions: Dict[str, PromptFunction] = {
    'default': default_extract_attributes_prompt_v1,
    'v1': default_extract_attributes_prompt_v1,
}
