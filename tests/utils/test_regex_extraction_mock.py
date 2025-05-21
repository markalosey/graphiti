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

import asyncio
import logging
import re
import pytest
from typing import Dict, List, Optional, Set, Tuple, Any, AsyncGenerator
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Mock implementation of EntityMatch and RegexEntityExtractor for testing
@dataclass
class EntityMatch:
    """Simple representation of an entity extracted from text."""

    entity_type: str
    name: str
    confidence: float
    pattern_name: str
    start_pos: int
    end_pos: int
    source_text: str


class RegexEntityExtractor:
    """A simplified mock version of the RegexEntityExtractor for testing."""

    def __init__(self, patterns: List[Dict[str, Any]]):
        self.patterns = patterns

    async def extract_entities(self, text: str) -> List[EntityMatch]:
        """
        Extract entities from text using regex patterns.
        This is a simplified mock implementation for testing.
        """
        entities = []

        for pattern in self.patterns:
            pattern_name = pattern.get('pattern_name', 'unknown')
            regex = pattern.get('regex', '')
            confidence = pattern.get('confidence', 0.0)

            matches = re.finditer(regex, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                # Assume first capture group is the entity name
                if len(match.groups()) > 0:
                    name = match.group(1).strip()
                    start_pos = match.start()
                    end_pos = match.end()
                    source_text = match.group(0)

                    entities.append(
                        EntityMatch(
                            entity_type=pattern_name,
                            name=name,
                            confidence=confidence,
                            pattern_name=pattern_name,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            source_text=source_text,
                        )
                    )

        return entities

    def filter_overlapping_matches(self, entities: List[EntityMatch]) -> List[EntityMatch]:
        """
        Filter out overlapping matches based on confidence and match quality.
        This is a simplified implementation for testing.
        """
        if not entities:
            return []

        # Sort by confidence (descending)
        sorted_entities = sorted(entities, key=lambda e: e.confidence, reverse=True)

        # Keep track of positions that are already covered
        covered_positions = set()
        filtered_entities = []

        for entity in sorted_entities:
            # Check if this entity overlaps with any existing covered positions
            entity_positions = set(range(entity.start_pos, entity.end_pos))
            if not entity_positions.intersection(covered_positions):
                # No overlap, add this entity
                filtered_entities.append(entity)
                covered_positions.update(entity_positions)

        return filtered_entities


# Define some test patterns
GUIDELINE_PATTERNS = [
    {
        'pattern_name': 'Task',
        'regex': r'Label:\s*Task\s*\n\s*Name:\s*([^\n]+)',
        'confidence': 0.95,
    },
    {
        'pattern_name': 'Idea',
        'regex': r'Label:\s*Idea\s*\n\s*Name:\s*([^\n]+)',
        'confidence': 0.95,
    },
    {
        'pattern_name': 'Collection',
        'regex': r'Label:\s*Collection\s*\n\s*Name:\s*([^\n]+)',
        'confidence': 0.95,
    },
    {
        'pattern_name': 'Procedure',
        'regex': r'Label:\s*Procedure\s*\n\s*Name:\s*([^\n]+)',
        'confidence': 0.95,
    },
]

# Simple test fixtures
STRUCTURED_EXAMPLES = [
    # Example 1: Overlapping entity names
    {
        'name': 'Overlapping entity names',
        'text': """
        We need to organize our growth strategy for Q4.

        Label: Collection
        Name: Growth Plan
        Content: This collection contains our growth strategy initiatives for Q4.
        
        Label: Idea
        Name: Growth Plan
        Content: Expand our business operations to the European market in Q4.
        
        Label: Task
        Name: Draft Growth Plan document
        Content: Create a comprehensive document outlining our growth strategy.
        Status: Not Started
        """,
        'expected': [
            ('Collection', 'Growth Plan'),
            ('Idea', 'Growth Plan'),
            ('Task', 'Draft Growth Plan document'),
        ],
    },
    # Example 2: Multiple entity types
    {
        'name': 'Multiple entity types',
        'text': """
        Let's track our customer feedback initiative.
        
        Label: Procedure
        Name: Customer Feedback Collection
        Content: Process for gathering and analyzing customer feedback.
        
        Label: Task
        Name: Create feedback form
        Content: Develop and implement an online customer feedback form.
        """,
        'expected': [
            ('Procedure', 'Customer Feedback Collection'),
            ('Task', 'Create feedback form'),
        ],
    },
]


@pytest.fixture
def extractor():
    """Fixture to provide an initialized RegexEntityExtractor."""
    return RegexEntityExtractor(GUIDELINE_PATTERNS)


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
        for exp_type, exp_name in expected_entities:
            if entity.entity_type == exp_type and entity.name == exp_name:
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

    # Extract entities
    entities = await extractor.extract_entities(text)
    entities = extractor.filter_overlapping_matches(entities)

    # Check that each expected entity is found
    found_entities = set()
    for entity in entities:
        for exp_type, exp_name in expected_entities:
            if entity.entity_type == exp_type and entity.name == exp_name:
                found_entities.add((exp_type, exp_name))

    # Missing entities
    missing = set(expected_entities) - found_entities
    assert not missing, f'Missing expected entities: {missing}'


@pytest.mark.asyncio
async def test_custom_pattern():
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

    # Create an extractor with the custom pattern
    custom_extractor = RegexEntityExtractor([custom_pattern])

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
    overlapping_example = STRUCTURED_EXAMPLES[0]

    # Extract entities without filtering
    entities = await extractor.extract_entities(overlapping_example['text'])

    # Count before filtering
    count_before = len(entities)

    # Apply filtering
    filtered_entities = extractor.filter_overlapping_matches(entities)

    # Verify filtering works (shouldn't increase count)
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
