"""
Demo script for using RegexEntityExtractor with structured fixture examples.

This script demonstrates how the regex-based entity extraction works with
structured examples from the edge cases feature file.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the path to the graphiti directory
script_dir = Path(__file__).resolve().parent
graphiti_dir = script_dir.parents[1]  # libs/graphiti
project_dir = graphiti_dir.parent.parent  # project-nanoo

# Make sure we're running from the correct directory
os.chdir(graphiti_dir)
sys.path.insert(0, str(graphiti_dir))

# Import graphiti_core modules
from graphiti_core.utils.entity_extractor import RegexEntityExtractor, EntityMatch
from graphiti_core.utils.pattern_library import get_all_patterns

# Structured examples from the edge cases feature file
STRUCTURED_EXAMPLES = [
    # Overlapping entity names
    """
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
    # Special characters and non-English
    """
    We need to organize our research and development initiatives.
    
    Label: Collection
    Name: R&D – 🚀 Projects
    Content: A collection of our cutting-edge research and development projects.
    
    Label: Idea
    Name: Über-automation for QA
    Content: Implement advanced automation for quality assurance processes.
    
    Label: Task
    Name: Fix bug #42 in módulo de pagos
    Content: Resolve the identified issue in the payment module.
    Priority: Critical
    Status: In Progress
    """,
    # Nested/referenced entities
    """
    We're planning our next development sprint.
    
    Label: Collection
    Name: Sprint 7
    Content: All tasks and ideas for Sprint 7 of our development cycle.
    
    Label: Task
    Name: Implement login
    Content: Develop and implement the login functionality for the application.
    Status: Not Started
    Relationships: BasedOn: Passwordless authentication
    
    Label: Idea
    Name: Passwordless authentication
    Content: Authentication system that doesn't require traditional passwords.
    """,
]


async def demonstrate_regex_extraction():
    """Demonstrate the regex-based entity extraction on structured examples."""
    logger.info('REGEX-BASED ENTITY EXTRACTION DEMO - STRUCTURED EXAMPLES')
    logger.info('=======================================================')

    # Create an entity extractor with all available patterns
    extractor = RegexEntityExtractor(get_all_patterns())

    # Process each example
    for i, example in enumerate(STRUCTURED_EXAMPLES, 1):
        logger.info(f'\nExample {i}:')
        logger.info('-' * 40)

        # Show a preview of the example
        preview = example.strip().split('\n')[:3]
        logger.info(f'Preview: {" ".join(preview)}...')

        # Extract entities
        entities = await extractor.extract_entities(example)
        entities = extractor.filter_overlapping_matches(entities)

        # Group entities by type for better display
        entities_by_type = {}
        for entity in entities:
            if entity.entity_type not in entities_by_type:
                entities_by_type[entity.entity_type] = []
            entities_by_type[entity.entity_type].append(entity)

        # Display extracted entities by type
        logger.info(f'\nExtracted {len(entities)} entities:')
        for entity_type, type_entities in entities_by_type.items():
            logger.info(f'\n{entity_type} entities:')
            for entity in type_entities:
                logger.info(
                    f'  - {entity.name} '
                    f'(Confidence: {entity.confidence:.2f}, Pattern: {entity.pattern_name})'
                )

    logger.info('\n=======================================================')
    logger.info('DEMO COMPLETE!')


if __name__ == '__main__':
    asyncio.run(demonstrate_regex_extraction())
