"""
Test script for edge case scenarios using the RegexEntityExtractor.

This script loads the structured test cases from the edge_cases.feature file
and tests the regex-based entity extraction on these scenarios.
"""

import asyncio
import logging
import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the correct path to import graphiti_core
project_root = Path(__file__).resolve().parents[3]  # Up to project-nanoo
sys.path.append(str(project_root / 'libs' / 'graphiti'))

# Now import graphiti_core modules
from graphiti_core.utils.entity_extractor import RegexEntityExtractor, EntityMatch
from graphiti_core.utils.pattern_library import get_all_patterns, get_guideline_patterns

# Path to edge cases feature file
EDGE_CASES_PATH = (
    project_root
    / 'fixtures'
    / 'graphiti_custom_entity_extraction_tests'
    / 'features'
    / 'edge_cases.feature'
)


def extract_scenario_text(feature_content: str) -> List[Tuple[str, str, List[Tuple[str, str]]]]:
    """
    Extract scenario text and expected entities from the feature file.

    Args:
        feature_content: Content of the feature file

    Returns:
        List of tuples containing (scenario_name, input_text, expected_entities)
    """
    # Only extract structured scenarios
    scenario_pattern = r'Scenario: ([^\n]*?)\(Structured\).*?Given the episode text:\s*"""(.*?)""".*?Then the following entities should be extracted:(.*?)(?=\n\s*(?:Scenario:|$))'

    matches = re.finditer(scenario_pattern, feature_content, re.DOTALL)

    scenarios = []
    for match in matches:
        scenario_name = match.group(1).strip()
        text = match.group(2).strip()

        # Extract expected entities from the table
        entity_table = match.group(3)
        entity_lines = entity_table.strip().split('\n')

        # Skip the header and separator rows
        entity_rows = [line.strip() for line in entity_lines if '|' in line]

        expected_entities = []
        for row in entity_rows:
            columns = [col.strip() for col in row.split('|')]
            if len(columns) >= 3:  # There will be empty strings at the beginning and end
                entity_type = columns[1].strip()
                entity_name = columns[2].strip()
                if entity_type and entity_name:  # Ensure we have both type and name
                    expected_entities.append((entity_type, entity_name))

        scenarios.append((scenario_name, text, expected_entities))

    return scenarios


async def test_edge_case(scenario_name: str, text: str, expected_entities: List[Tuple[str, str]]):
    """
    Test a single edge case scenario.

    Args:
        scenario_name: Name of the scenario
        text: Input text
        expected_entities: List of (type, name) pairs for expected entities
    """
    logger.info(f'Testing scenario: {scenario_name}')

    # Create entity extractor with all patterns
    extractor = RegexEntityExtractor(get_all_patterns())

    # Extract entities using regex
    entities = await extractor.extract_entities(text)
    entities = extractor.filter_overlapping_matches(entities)

    # Track which expected entities were found
    found_entities: Set[Tuple[str, str]] = set()
    for entity in entities:
        logger.info(
            f'  Found: {entity.name} (Type: {entity.entity_type}, Confidence: {entity.confidence:.2f})'
        )

        # Map RegexEntityExtractor's entity_type to the expected type
        mapped_type = map_entity_type(entity.entity_type)

        # Check if this entity matches any expected entity
        for exp_type, exp_name in expected_entities:
            if mapped_type == exp_type and entity.name == exp_name:
                found_entities.add((exp_type, exp_name))

    # Check for missing entities
    expected_set = set(expected_entities)
    missing = expected_set - found_entities
    if missing:
        logger.warning(f"  Missing entities in scenario '{scenario_name}':")
        for etype, ename in missing:
            logger.warning(f'    - {ename} (Type: {etype})')

    # Calculate success rate
    success_rate = len(found_entities) / len(expected_entities) if expected_entities else 0
    logger.info(
        f'  Success rate: {success_rate:.2%} ({len(found_entities)}/{len(expected_entities)} entities found)'
    )

    return success_rate, len(found_entities), len(expected_entities)


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


async def main():
    """Main test function."""
    logger.info('TESTING REGEX ENTITY EXTRACTION ON EDGE CASES')
    logger.info('============================================')

    # Load the edge cases feature file
    try:
        with open(EDGE_CASES_PATH, 'r') as f:
            feature_content = f.read()
    except FileNotFoundError:
        logger.error(f'Edge cases feature file not found at: {EDGE_CASES_PATH}')
        return

    # Extract scenarios from the feature file
    scenarios = extract_scenario_text(feature_content)
    logger.info(f'Extracted {len(scenarios)} structured scenarios from the feature file')

    # Test each scenario
    total_found = 0
    total_expected = 0

    for scenario_name, text, expected_entities in scenarios:
        success_rate, found, expected = await test_edge_case(scenario_name, text, expected_entities)
        total_found += found
        total_expected += expected
        logger.info('')  # Add a blank line between scenarios

    # Calculate overall success rate
    overall_success_rate = total_found / total_expected if total_expected else 0
    logger.info('============================================')
    logger.info(
        f'OVERALL SUCCESS RATE: {overall_success_rate:.2%} ({total_found}/{total_expected} entities found)'
    )

    if overall_success_rate == 1.0:
        logger.info('✅ ALL ENTITIES SUCCESSFULLY EXTRACTED!')
    else:
        logger.info(f'⚠️ SOME ENTITIES WERE NOT EXTRACTED ({total_expected - total_found} missing)')


if __name__ == '__main__':
    asyncio.run(main())
