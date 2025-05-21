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

Task: Based on all the provided context (especially EPISODE_CONTENT), extract values for the following attributes. 
If an attribute is not clearly present or applicable, omit it from your response or set its value to null, 
unless the attribute definition implies a default (like an empty list for tags if none are found).

Attributes to Extract (name: description):
{attributes_to_extract_str}

Respond with a JSON object containing only the extracted attributes and their values. 
For list-based attributes like 'tags', ensure the value is a list of strings. 
For optional fields, if no information is found, you can omit the field or return null.
Example for a node that is an 'Idea' and has tags and details:
{{
  "tags": ["machine-learning", "new-product"],
  "details": "This idea involves creating a new ML model to predict customer churn.",
  "category": "product-development"
}}
Example for an 'Idea' with only tags:
{{
  "tags": ["internal-tool"],
  "details": null,
  "category": null
}}
Example for an 'Idea' with no extractable attributes of these types (based on the definition):
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
