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

import logging
import pytest
import pytest_asyncio
from typing import Dict, List, Set, Tuple, Any

# Import the modules we're testing
from graphiti_core.utils.entity_extractor import RegexEntityExtractor, EntityMatch
from graphiti_core.utils.pattern_library import get_all_patterns, get_guideline_patterns

# Import the test fixtures
from tests.fixtures.entity_extraction.structured_examples import STRUCTURED_EXAMPLES

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def map_entity_type(regex_type: str) -> str:
    """
    Map entity types from RegexEntityExtractor to expected entity types.

    Args:
        regex_type: Entity type from RegexEntityExtractor

    Returns:
        Mapped entity type
    """
    # Handle special cases from pattern_library.py
    if regex_type == 'EntityType':
        return 'Entity'
    elif regex_type == 'EntityName':
        return 'Entity'

    # Direct mapping for common types
    type_mapping = {
        'Task': 'Task',
        'Idea': 'Idea',
        'Collection': 'Collection',
        'Requirement': 'Requirement',
        'Preference': 'Preference',
        'Procedure': 'Procedure',
    }

    return type_mapping.get(regex_type, regex_type)


@pytest_asyncio.fixture
async def extractor():
    """Fixture to provide an initialized RegexEntityExtractor."""
    return RegexEntityExtractor(get_all_patterns())


@pytest.mark.asyncio
async def test_entity_extraction_basic(extractor):
    """Test basic entity extraction functionality."""
    # Use the first example
    example = STRUCTURED_EXAMPLES[0]
    text = example['text']
    expected_entities = example['expected']

    # Extract entities
    entities = await extractor.extract_entities(text)
    entities = extractor.filter_overlapping_matches(entities)

    # Verify entity count
    assert len(entities) >= len(expected_entities), (
        f'Expected at least {len(expected_entities)} entities, got {len(entities)}'
    )

    # Check that each expected entity is found
    found_entities = set()
    for entity in entities:
        mapped_type = map_entity_type(entity.entity_type)
        for exp_type, exp_name in expected_entities:
            if mapped_type == exp_type and entity.name == exp_name:
                found_entities.add((exp_type, exp_name))

    # Make sure all expected entities are found
    missing = set(expected_entities) - found_entities
    assert not missing, f'Missing expected entities: {missing}'


@pytest.mark.asyncio
@pytest.mark.parametrize('example_index', range(len(STRUCTURED_EXAMPLES)))
async def test_entity_extraction_all_examples(extractor, example_index):
    """Parametrized test that runs through all structured examples."""
    example = STRUCTURED_EXAMPLES[example_index]
    text = example['text']
    expected_entities = example['expected']

    # Log which example we're testing
    logger.info(f'Testing example {example_index + 1}: {example["name"]}')

    # Extract entities
    entities = await extractor.extract_entities(text)
    entities = extractor.filter_overlapping_matches(entities)

    # Check that each expected entity is found
    found_entities = set()
    for entity in entities:
        mapped_type = map_entity_type(entity.entity_type)
        for exp_type, exp_name in expected_entities:
            if mapped_type == exp_type and entity.name == exp_name:
                found_entities.add((exp_type, exp_name))

    # Make sure all expected entities are found
    missing = set(expected_entities) - found_entities
    assert not missing, f"Missing expected entities in example '{example['name']}': {missing}"


@pytest.mark.asyncio
async def test_custom_pattern(extractor):
    """Test that custom patterns can be added and used for extraction."""
    # Create a custom pattern
    custom_pattern = {
        'pattern_name': 'CustomEntityType',
        'regex': r'Label:\s*CustomType\s*\n\s*Name:\s*([^\n]+)',
        'confidence': 0.95,
    }

    # Custom text with the custom entity
    custom_text = """
    This is a custom entity example.
    
    Label: CustomType
    Name: My Custom Entity
    Content: This is a custom entity with a custom type.
    """

    # Create an extractor with the custom pattern added
    custom_patterns = get_guideline_patterns() + [custom_pattern]
    custom_extractor = RegexEntityExtractor(custom_patterns)

    # Extract entities
    entities = await custom_extractor.extract_entities(custom_text)

    # Verify we found the custom entity
    assert len(entities) > 0, 'No entities found with custom pattern'
    assert any(entity.entity_type == 'CustomEntityType' for entity in entities), (
        'Custom entity type not found'
    )
    assert any(entity.name == 'My Custom Entity' for entity in entities), (
        'Custom entity name not found'
    )


@pytest.mark.asyncio
async def test_overlapping_matches_filtering(extractor):
    """Test that overlapping matches are properly filtered."""
    # Example with overlapping entity names
    overlapping_example = [
        ex for ex in STRUCTURED_EXAMPLES if ex['name'] == 'Overlapping entity names'
    ][0]

    # Extract entities without filtering
    entities = await extractor.extract_entities(overlapping_example['text'])

    # Count before filtering
    count_before = len(entities)

    # Apply filtering
    filtered_entities = extractor.filter_overlapping_matches(entities)

    # Verify filtering works
    assert len(filtered_entities) <= count_before, 'Filtering should not increase entity count'

    # Check for overlapping positions in filtered results
    positions = []
    for entity in filtered_entities:
        pos = (entity.start_pos, entity.end_pos)
        positions.append(pos)

    # Check that no positions overlap
    for i, pos1 in enumerate(positions):
        for j, pos2 in enumerate(positions):
            if i != j:
                # Check if positions overlap
                overlap = (pos1[0] <= pos2[0] < pos1[1]) or (pos2[0] <= pos1[0] < pos2[1])
                assert not overlap, f'Found overlapping positions: {pos1} and {pos2}'


if __name__ == '__main__':
    pytest.main(['-v', __file__])
