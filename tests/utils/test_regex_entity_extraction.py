"""
Test script for the RegexEntityExtractor functionality.

This script tests the regex-based entity extraction capability using
structured examples from our fixtures directory.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Make sure we're working from the Graphiti repo root
script_dir = Path(__file__).resolve().parent
graphiti_test_dir = script_dir.parent  # tests directory
graphiti_dir = graphiti_test_dir.parent  # graphiti root directory
sys.path.insert(0, str(graphiti_dir))

# Now we can import from graphiti_core
from graphiti_core.utils.entity_extractor import RegexEntityExtractor, EntityMatch
from graphiti_core.utils.pattern_library import get_all_patterns, get_guideline_patterns

# Import the structured examples from our fixtures
from tests.fixtures.entity_extraction.structured_examples import STRUCTURED_EXAMPLES


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


async def test_regex_extraction():
    """Test the regex-based entity extraction functionality with structured examples."""

    logger.info('TESTING REGEX-BASED ENTITY EXTRACTION')
    logger.info('=====================================')

    # Create entity extractor with all available patterns
    extractor = RegexEntityExtractor(get_all_patterns())

    total_found = 0
    total_expected = 0

    # Test each structured example
    for i, example in enumerate(STRUCTURED_EXAMPLES, 1):
        example_name = example['name']
        text = example['text']
        expected_entities = example['expected']

        logger.info(f'\nExample {i}: {example_name}')
        logger.info('=' * 40)

        # Extract entities
        entities = await extractor.extract_entities(text)
        entities = extractor.filter_overlapping_matches(entities)

        # Log what we found
        logger.info(f'Found {len(entities)} entities:')
        for entity in entities:
            logger.info(
                f'  - {entity.name} (Type: {entity.entity_type}, Confidence: {entity.confidence:.2f})'
            )

        # Check against expected entities
        found_entities = set()
        for entity in entities:
            mapped_type = map_entity_type(entity.entity_type)
            for exp_type, exp_name in expected_entities:
                if mapped_type == exp_type and entity.name == exp_name:
                    found_entities.add((exp_type, exp_name))

        # Look for missing entities
        expected_set = set(expected_entities)
        missing = expected_set - found_entities

        if missing:
            logger.warning(f'Missing {len(missing)} expected entities:')
            for etype, ename in missing:
                logger.warning(f'  - {ename} (Type: {etype})')
        else:
            logger.info('✅ All expected entities were found!')

        # Update counters
        example_found = len(found_entities)
        example_expected = len(expected_entities)
        total_found += example_found
        total_expected += example_expected

        # Report success rate
        success_rate = example_found / example_expected if example_expected else 1.0
        logger.info(f'Success rate: {success_rate:.2%} ({example_found}/{example_expected})')

    # Report overall success
    overall_success = total_found / total_expected if total_expected else 0
    logger.info('\n=====================================')
    logger.info(f'OVERALL SUCCESS RATE: {overall_success:.2%} ({total_found}/{total_expected})')

    if overall_success == 1.0:
        logger.info('✅ ALL TESTS PASSED! The regex-based entity extraction works perfectly!')
    else:
        logger.info(f'⚠️ SOME TESTS FAILED ({total_expected - total_found} entities not extracted)')


if __name__ == '__main__':
    asyncio.run(test_regex_extraction())
