#!/usr/bin/env python
"""
Demo script for the Regex-Based Entity Extraction feature.

This script demonstrates how the RegexEntityExtractor can be used to
extract entities from text following the Graphiti Formatting Guidelines
without requiring LLM calls.
"""

import asyncio
import logging
import sys
from pathlib import Path
from pprint import pprint
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the Graphiti directory to the path
graphiti_dir = Path(__file__).resolve().parents[2]  # Up to libs/graphiti
sys.path.insert(0, str(graphiti_dir))

# Import Graphiti components
from graphiti_core.utils.entity_extractor import RegexEntityExtractor, EntityMatch
from graphiti_core.utils.pattern_library import get_all_patterns, get_guideline_patterns
from tests.fixtures.entity_extraction.structured_examples import STRUCTURED_EXAMPLES


async def demo():
    """Demonstrate the regex-based entity extraction feature."""

    logger.info('=== Regex-Based Entity Extraction Demo ===')
    logger.info('This demo shows how the RegexEntityExtractor extracts entities from')
    logger.info('text that follows the Graphiti Formatting Guidelines without using LLMs.')

    # Create entity extractor with guidelines patterns
    extractor = RegexEntityExtractor(get_guideline_patterns())
    logger.info('\nCreated entity extractor with guideline patterns')

    # Use a sample text from our fixtures
    example = STRUCTURED_EXAMPLES[0]  # Overlapping entity names example
    logger.info(f'\nExample: {example["name"]}')
    logger.info('----------------------------')
    logger.info(f'Text:\n{example["text"]}')

    # Time the extraction
    start_time = time.time()

    # Extract entities
    entities = await extractor.extract_entities(example['text'])
    entities = extractor.filter_overlapping_matches(entities)

    elapsed_time = (time.time() - start_time) * 1000  # Convert to ms

    # Display results
    logger.info('\nExtracted Entities:')
    logger.info('------------------')
    for entity in entities:
        logger.info(f'  • Type: {entity.entity_type}')
        logger.info(f'    Name: {entity.name}')
        logger.info(f'    Confidence: {entity.confidence:.2f}')
        logger.info(f'    Match: {entity.source_text[:50]}...')
        logger.info('')

    logger.info(f'Extraction completed in {elapsed_time:.2f}ms')

    # Show expected entities for comparison
    logger.info('\nExpected Entities:')
    logger.info('-----------------')
    for entity_type, entity_name in example['expected']:
        logger.info(f'  • Type: {entity_type}')
        logger.info(f'    Name: {entity_name}')
        logger.info('')

    # Performance comparison
    logger.info('\nPerformance Comparison:')
    logger.info('---------------------')
    logger.info(f'Regex extraction: {elapsed_time:.2f}ms')
    logger.info('LLM-based extraction: ~1000-5000ms (estimated)')
    logger.info('Regex is ~10-50x faster for well-structured text!')

    # Now let's run a more comprehensive benchmark
    logger.info('\n=== Comprehensive Benchmark ===')

    # Time extraction for all examples
    total_start_time = time.time()
    total_entities = 0

    for i, example in enumerate(STRUCTURED_EXAMPLES, 1):
        logger.info(f'\nProcessing example {i}: {example["name"]}')

        # Extract entities
        start_time = time.time()
        entities = await extractor.extract_entities(example['text'])
        entities = extractor.filter_overlapping_matches(entities)
        extraction_time = (time.time() - start_time) * 1000  # Convert to ms

        # Count matches
        total_entities += len(entities)
        expected_count = len(example['expected'])

        logger.info(f'Found {len(entities)}/{expected_count} entities in {extraction_time:.2f}ms')

    total_time = (time.time() - total_start_time) * 1000  # Convert to ms
    logger.info('\n=== Benchmark Results ===')
    logger.info(f'Processed {len(STRUCTURED_EXAMPLES)} examples')
    logger.info(f'Extracted {total_entities} entities')
    logger.info(f'Total time: {total_time:.2f}ms')
    logger.info(f'Average time per example: {total_time / len(STRUCTURED_EXAMPLES):.2f}ms')

    # Show how to create a custom pattern
    logger.info('\n=== Creating Custom Patterns ===')
    logger.info('You can extend the pattern library with your own patterns:')

    custom_pattern = {
        'pattern_name': 'CustomEntityType',
        'regex': r'Label:\s*CustomType\s*\n\s*Name:\s*([^\n]+)',
        'confidence': 0.95,
    }

    logger.info(f'\nCustom pattern definition:')
    pprint(custom_pattern)

    custom_text = """
    This is a custom entity example.
    
    Label: CustomType
    Name: My Custom Entity
    Content: This is a custom entity with a custom type.
    """

    logger.info(f'\nText with custom entity:\n{custom_text}')

    # Create extractor with custom pattern
    custom_patterns = get_guideline_patterns() + [custom_pattern]
    custom_extractor = RegexEntityExtractor(custom_patterns)

    # Extract entities with custom pattern
    custom_entities = await custom_extractor.extract_entities(custom_text)

    logger.info('\nEntities extracted with custom pattern:')
    for entity in custom_entities:
        logger.info(f'  • Type: {entity.entity_type}')
        logger.info(f'    Name: {entity.name}')
        logger.info(f'    Confidence: {entity.confidence:.2f}')
        logger.info('')


if __name__ == '__main__':
    asyncio.run(demo())
