"""
Pattern library for regex-based entity extraction.

This module provides a collection of regex patterns that specifically match the
Graphiti Formatting Guidelines structure, allowing for efficient entity extraction
without requiring LLM calls for well-structured content.
"""

from typing import Dict, Any, List

# Core patterns matching the Graphiti Formatting Guidelines
GUIDELINE_PATTERNS: Dict[str, Dict[str, Any]] = {
    # Core entity definition patterns from guidelines
    'label_entity_type': {
        'regex': r'(?:^|\n)Label:\s*([A-Z][a-zA-Z]+)(?=\s*$|\s*\n)',
        'entity_type': 'EntityType',
        'confidence': 0.95,
        'group_name': 1,
    },
    'entity_name': {
        'regex': r'(?:^|\n)Name:\s*([^\n]+?)(?=\s*$|\s*\n)',
        'entity_type': 'EntityName',
        'confidence': 0.95,
        'group_name': 1,
    },
    'entity_content': {
        'regex': r'(?:^|\n)Content:\s*((?:.+?\n)+?)(?=\n[A-Z][\w]+:|\s*$)',
        'entity_type': 'Content',
        'confidence': 0.9,
        'group_name': 1,
    },
    # Common key-value pairs from guidelines
    'entity_uuid': {
        'regex': r'(?:^|\n)UUID:\s*([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})(?=\s*$|\s*\n)',
        'entity_type': 'UUID',
        'confidence': 0.98,
        'group_name': 1,
    },
    'entity_status': {
        'regex': r'(?:^|\n)Status:\s*([A-Za-z]+)(?=\s*$|\s*\n)',
        'entity_type': 'Status',
        'confidence': 0.9,
        'group_name': 1,
    },
    'entity_priority': {
        'regex': r'(?:^|\n)Priority:\s*([A-Za-z]+)(?=\s*$|\s*\n)',
        'entity_type': 'Priority',
        'confidence': 0.9,
        'group_name': 1,
    },
    'entity_category': {
        'regex': r'(?:^|\n)Category:\s*([^\n]+?)(?=\s*$|\s*\n)',
        'entity_type': 'Category',
        'confidence': 0.9,
        'group_name': 1,
    },
    'entity_filepath': {
        'regex': r'(?:^|\n)Filepath:\s*([^\n]+?)(?=\s*$|\s*\n)',
        'entity_type': 'Filepath',
        'confidence': 0.9,
        'group_name': 1,
    },
    'entity_tags': {
        'regex': r'(?:^|\n)Tags:\s*([^\n]+?)(?=\s*$|\s*\n)',
        'entity_type': 'Tags',
        'confidence': 0.9,
        'group_name': 1,
    },
    # Relationship patterns from guidelines
    'entity_relationship': {
        'regex': r'(?:^|\n)Relationships:\s*([A-Za-z]+):\s*([^\n(]+)(?:\s*\(([^)]+)\))?(?=\s*$|\s*\n)',
        'entity_type': 'Relationship',
        'confidence': 0.95,
        'multi_group': True,
        'groups': {'type': 1, 'target': 2, 'id': 3},
    },
    # Entity type-specific patterns (from Common EntityType examples in guidelines)
    'requirement_entity': {
        'regex': r'(?:^|\n)Label:\s*Requirement(?=\s*$|\s*\n)',
        'entity_type': 'Requirement',
        'confidence': 0.95,
        'pattern_match': True,
    },
    'preference_entity': {
        'regex': r'(?:^|\n)Label:\s*Preference(?=\s*$|\s*\n)',
        'entity_type': 'Preference',
        'confidence': 0.95,
        'pattern_match': True,
    },
    'procedure_entity': {
        'regex': r'(?:^|\n)Label:\s*Procedure(?=\s*$|\s*\n)',
        'entity_type': 'Procedure',
        'confidence': 0.95,
        'pattern_match': True,
    },
    'task_entity': {
        'regex': r'(?:^|\n)Label:\s*Task(?=\s*$|\s*\n)',
        'entity_type': 'Task',
        'confidence': 0.95,
        'pattern_match': True,
    },
    'idea_entity': {
        'regex': r'(?:^|\n)Label:\s*Idea(?=\s*$|\s*\n)',
        'entity_type': 'Idea',
        'confidence': 0.95,
        'pattern_match': True,
    },
}

# Additional patterns for unstructured or partially structured text
HEURISTIC_PATTERNS: Dict[str, Dict[str, Any]] = {
    # Infer entity type from colon syntax without explicit Label:
    'task_colon': {
        'regex': r'(?:^|\n)Task:\s*([^\n]+?)(?=\s*$|\s*\n)',
        'entity_type': 'Task',
        'confidence': 0.85,
        'group_name': 1,
    },
    'idea_colon': {
        'regex': r'(?:^|\n)Idea:\s*([^\n]+?)(?=\s*$|\s*\n)',
        'entity_type': 'Idea',
        'confidence': 0.85,
        'group_name': 1,
    },
    'requirement_colon': {
        'regex': r'(?:^|\n)Requirement:\s*([^\n]+?)(?=\s*$|\s*\n)',
        'entity_type': 'Requirement',
        'confidence': 0.85,
        'group_name': 1,
    },
    'preference_colon': {
        'regex': r'(?:^|\n)Preference:\s*([^\n]+?)(?=\s*$|\s*\n)',
        'entity_type': 'Preference',
        'confidence': 0.85,
        'group_name': 1,
    },
    'procedure_colon': {
        'regex': r'(?:^|\n)Procedure:\s*([^\n]+?)(?=\s*$|\s*\n)',
        'entity_type': 'Procedure',
        'confidence': 0.85,
        'group_name': 1,
    },
    # Indirect references with quoted names
    'task_called': {
        'regex': r"(?:a|the|new)\s+Task\s+(?:called|named|titled)\s+['\"]([^'\"\n]+?)['\"]",
        'entity_type': 'Task',
        'confidence': 0.75,
        'group_name': 1,
    },
    'idea_called': {
        'regex': r"(?:a|the|new)\s+Idea\s+(?:called|named|titled)\s+['\"]([^'\"\n]+?)['\"]",
        'entity_type': 'Idea',
        'confidence': 0.75,
        'group_name': 1,
    },
    # Common relationship phrases
    'depends_on': {
        'regex': r"['\"]([^'\"\n]+?)['\"](?:\s+\([^)]*\))?\s+depends\s+on\s+['\"]([^'\"\n]+?)['\"]",
        'entity_type': 'Relationship',
        'confidence': 0.7,
        'relationship_type': 'DependsOn',
        'multi_group': True,
        'groups': {'source': 1, 'target': 2},
    },
    'part_of': {
        'regex': r"['\"]([^'\"\n]+?)['\"](?:\s+\([^)]*\))?\s+is\s+part\s+of\s+['\"]([^'\"\n]+?)['\"]",
        'entity_type': 'Relationship',
        'confidence': 0.7,
        'relationship_type': 'PartOf',
        'multi_group': True,
        'groups': {'source': 1, 'target': 2},
    },
}


# Get a combined set of patterns
def get_all_patterns() -> Dict[str, Dict[str, Any]]:
    """
    Get a combined dictionary of all available patterns.

    Returns:
        Dictionary with all guideline and heuristic patterns.
    """
    combined = GUIDELINE_PATTERNS.copy()
    combined.update(HEURISTIC_PATTERNS)
    return combined


def get_guideline_patterns() -> Dict[str, Dict[str, Any]]:
    """
    Get only the patterns that strictly match the Graphiti Formatting Guidelines.

    Returns:
        Dictionary with only guideline-compliant patterns.
    """
    return GUIDELINE_PATTERNS.copy()


def get_patterns_by_entity_type(entity_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Get a filtered set of patterns for a specific entity type.

    Args:
        entity_type: The entity type to filter for

    Returns:
        Dictionary with patterns matching the specified entity type.
    """
    all_patterns = get_all_patterns()
    return {
        name: pattern
        for name, pattern in all_patterns.items()
        if pattern.get('entity_type') == entity_type
    }


def get_custom_patterns(pattern_dicts: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Create a pattern dictionary from a list of custom pattern dictionaries.

    Args:
        pattern_dicts: List of dictionaries containing pattern definitions

    Returns:
        Dictionary of named patterns
    """
    custom_patterns = {}

    for i, pattern_dict in enumerate(pattern_dicts):
        pattern_name = pattern_dict.get('name', f'custom_pattern_{i}')

        # Create the pattern entry
        custom_patterns[pattern_name] = {
            'regex': pattern_dict['regex'],
            'entity_type': pattern_dict.get('entity_type', 'Entity'),
            'confidence': pattern_dict.get('confidence', 0.8),
            'group_name': pattern_dict.get('group_name', 0),
        }

    return custom_patterns
