"""
Regex-based entity extraction utility for Graphiti.

This module provides functionality to extract entities from text using regex patterns,
without requiring LLM calls for simple, structured entity formats.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EntityMatch(BaseModel):
    """Represents an entity match found by regex patterns."""

    name: str = Field(..., description='Name of the extracted entity')
    entity_type: str = Field(..., description='Type of the extracted entity')
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description='Confidence score of the match (0.0-1.0)'
    )
    start_pos: int = Field(..., description='Start position in the original text')
    end_pos: int = Field(..., description='End position in the original text')
    pattern_name: str = Field(..., description='Name of the pattern that matched this entity')


class RegexEntityExtractor:
    """
    A utility class that extracts entities from text using regex patterns.

    This class provides an alternative to LLM-based entity extraction for cases
    where the entities follow known, structured formats.
    """

    def __init__(self, patterns: Dict[str, Dict[str, Any]]):
        """
        Initialize the RegexEntityExtractor with a dictionary of patterns.

        Args:
            patterns: A dictionary where keys are pattern names and values are
                      dictionaries containing:
                      - 'regex': The regex pattern string
                      - 'entity_type': The type to assign to matching entities
                      - 'confidence': Base confidence for this pattern type
                      - 'group_name': Optional name of the regex group to extract (default: 0)
        """
        self.patterns = {}

        # Compile all regex patterns
        for pattern_name, pattern_info in patterns.items():
            try:
                self.patterns[pattern_name] = {
                    'regex': re.compile(pattern_info['regex'], re.MULTILINE | re.DOTALL),
                    'entity_type': pattern_info['entity_type'],
                    'confidence': pattern_info.get('confidence', 0.8),
                    'group_name': pattern_info.get('group_name', 0),
                }
            except re.error as e:
                logger.error(f"Invalid regex pattern '{pattern_name}': {e}")

    async def extract_entities(self, text: str) -> List[EntityMatch]:
        """
        Extract entities from the provided text using the configured regex patterns.

        Args:
            text: The text to extract entities from

        Returns:
            A list of EntityMatch objects representing the extracted entities
        """
        matches = []

        for pattern_name, pattern_info in self.patterns.items():
            regex = pattern_info['regex']
            entity_type = pattern_info['entity_type']
            base_confidence = pattern_info['confidence']
            group_name = pattern_info['group_name']

            # Find all matches
            for match in regex.finditer(text):
                try:
                    # Extract the entity name from the specified group
                    entity_name = match.group(group_name).strip()

                    # Skip empty matches
                    if not entity_name:
                        continue

                    # Create the match object
                    entity_match = EntityMatch(
                        name=entity_name,
                        entity_type=entity_type,
                        confidence=base_confidence,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        pattern_name=pattern_name,
                    )

                    matches.append(entity_match)
                    logger.debug(f'Found entity: {entity_match}')
                except IndexError:
                    logger.warning(
                        f"Pattern '{pattern_name}' matched but group {group_name} doesn't exist"
                    )

        # Sort matches by confidence (higher first)
        return sorted(matches, key=lambda m: m.confidence, reverse=True)

    def filter_overlapping_matches(self, matches: List[EntityMatch]) -> List[EntityMatch]:
        """
        Filter out overlapping matches, keeping the ones with higher confidence.

        Args:
            matches: List of entity matches to filter

        Returns:
            Filtered list of matches with overlaps removed
        """
        if not matches:
            return []

        # Sort by confidence (higher first), then by match length (shorter first)
        sorted_matches = sorted(
            matches, key=lambda m: (m.confidence, m.end_pos - m.start_pos), reverse=True
        )

        filtered_matches = []
        used_ranges = []

        for match in sorted_matches:
            # Check if this match overlaps with any previously accepted match
            overlap = False
            for start, end in used_ranges:
                # Check for any overlap
                if not (match.end_pos <= start or match.start_pos >= end):
                    overlap = True
                    break

            if not overlap:
                filtered_matches.append(match)
                used_ranges.append((match.start_pos, match.end_pos))

        return filtered_matches
