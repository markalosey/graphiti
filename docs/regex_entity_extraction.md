# Regex-Based Entity Extraction for Graphiti

## Overview

The Regex-Based Entity Extraction feature provides an efficient way to extract entities from text that follows the Graphiti Formatting Guidelines without requiring LLM calls. This approach significantly improves performance and reduces costs for well-structured content.

## Key Components

The implementation consists of two main components:

1. **RegexEntityExtractor Class (`entity_extractor.py`)**:
   - Core extraction functionality
   - Pattern matching and entity identification
   - Confidence scoring
   - Overlapping match resolution

2. **Pattern Library (`pattern_library.py`)**:
   - Collection of regex patterns specifically designed for Graphiti Formatting Guidelines
   - Core entity structure patterns (Label, Name, Content)
   - Entity-specific type patterns
   - Attribute patterns (Status, Priority, Category, etc.)
   - Relationship patterns

## How It Works

The regex-based extraction works as follows:

1. Text is processed using the pattern library
2. Matches are converted to `EntityMatch` objects with confidence scores
3. Overlapping matches are filtered based on confidence and match quality
4. Extracted entities are returned in the same format as the LLM-based extraction

### Performance Benefits

Regex-based entity extraction provides significant performance improvements:

- **Speed**: 10-50x faster than LLM-based extraction for well-structured text
- **Cost**: No API calls required, reducing operational costs
- **Reliability**: Consistent extraction patterns regardless of LLM variability

## Integration with Existing Codebase

The feature is integrated with the existing entity extraction pipeline as a preprocessing step:

1. Text is first processed with the regex extractor
2. If entities are found with high confidence, they are used directly
3. If no entities or only low-confidence entities are found, the system falls back to LLM extraction
4. The results can be combined to get the benefits of both approaches

## Usage Examples

### Basic Usage

```python
from graphiti_core.utils.entity_extractor import RegexEntityExtractor
from graphiti_core.utils.pattern_library import get_guideline_patterns

# Create an extractor with guideline patterns
extractor = RegexEntityExtractor(get_guideline_patterns())

# Extract entities from text
entities = await extractor.extract_entities(text)

# Filter overlapping matches
filtered_entities = extractor.filter_overlapping_matches(entities)

# Process the extracted entities
for entity in filtered_entities:
    print(f"Type: {entity.entity_type}")
    print(f"Name: {entity.name}")
    print(f"Confidence: {entity.confidence}")
```

### Custom Patterns

You can extend the pattern library with your own custom patterns:

```python
# Define a custom pattern
custom_pattern = {
    "pattern_name": "CustomEntityType",
    "regex": r"Label:\s*CustomType\s*\n\s*Name:\s*([^\n]+)",
    "confidence": 0.95,
}

# Combine with existing patterns
all_patterns = get_guideline_patterns() + [custom_pattern]

# Create extractor with combined patterns
custom_extractor = RegexEntityExtractor(all_patterns)
```

## Testing

The feature includes comprehensive automated tests to ensure reliability:

1. **Unit Tests**: Validate the core functionality of the RegexEntityExtractor class
2. **Integration Tests**: Test the regex extractor with real-world examples
3. **Parameterized Tests**: Run tests against a variety of edge cases
4. **Performance Benchmarks**: Compare performance with LLM-based extraction

### Running Tests

Tests can be run using pytest:

```bash
# Run all tests
pytest tests/utils/test_regex_extraction.py

# Run with verbose output
pytest tests/utils/test_regex_extraction.py -v

# Run a specific test
pytest tests/utils/test_regex_extraction.py::test_entity_extraction_basic
```

### GitHub Actions Integration

The tests are automatically run via GitHub Actions whenever changes are made to:
- The entity extractor or pattern library code
- The test files or fixtures

This ensures that any changes to the extraction logic are thoroughly tested before being merged.

## Fixtures and Test Data

The test suite includes a variety of structured examples designed to test different aspects of entity extraction:

- Overlapping entity names
- Special characters and non-English text
- Nested/referenced entities
- Explicit relationships
- Very long and very short names
- Similar names with different types
- Entities with numbers and special formatting
- Entity attributes and multiple fields

These test cases help ensure the regex extractor handles real-world scenarios correctly.

## Future Enhancements

Planned improvements to the regex-based entity extraction feature:

1. **Performance Optimizations**: Further optimize patterns for even faster extraction
2. **Additional Pattern Types**: Extend the pattern library to support more entity types
3. **Confidence Calibration**: Refine confidence scoring for better decision-making
4. **Adaptive Extraction**: Dynamically select patterns based on text characteristics
5. **Multilingual Support**: Add patterns for entity extraction in multiple languages 