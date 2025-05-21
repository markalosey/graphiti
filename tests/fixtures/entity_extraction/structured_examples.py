"""
Structured examples for testing regex-based entity extraction.

This module contains example texts that follow the Graphiti Formatting Guidelines
to test the RegexEntityExtractor functionality.
"""

from typing import Dict, List, Tuple, Any

# Collection of structured examples for testing
# Each example is a dictionary with:
# - name: descriptive name of the example
# - text: the input text following Graphiti Formatting Guidelines
# - expected: list of (entity_type, entity_name) tuples
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
    # Example 2: Special characters and non-English
    {
        'name': 'Special characters and non-English',
        'text': """
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
        'expected': [
            ('Collection', 'R&D – 🚀 Projects'),
            ('Idea', 'Über-automation for QA'),
            ('Task', 'Fix bug #42 in módulo de pagos'),
        ],
    },
    # Example 3: Nested/referenced entities
    {
        'name': 'Nested/referenced entities',
        'text': """
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
        'expected': [
            ('Collection', 'Sprint 7'),
            ('Task', 'Implement login'),
            ('Idea', 'Passwordless authentication'),
        ],
    },
    # Example 4: Explicit relationships
    {
        'name': 'Explicit relationships between entities',
        'text': """
        Planning our release process.
        
        Label: Collection
        Name: Release 1.0
        Content: All tasks related to our 1.0 product release.
        
        Label: Task
        Name: Deploy to production
        Content: Push the finalized code to the production environment.
        Status: Blocked
        Relationships: DependsOn: Pass all tests
        
        Label: Task
        Name: Pass all tests
        Content: Ensure that all automated tests pass successfully.
        Status: In Progress
        Relationships: PartOf: Release 1.0
        """,
        'expected': [
            ('Collection', 'Release 1.0'),
            ('Task', 'Deploy to production'),
            ('Task', 'Pass all tests'),
        ],
    },
    # Example 5: Very long and very short names
    {
        'name': 'Very long and very short names',
        'text': """
        Testing name length variations.
        
        Label: Collection
        Name: A
        Content: A simple test collection with a very short name.
        
        Label: Idea
        Name: This is an extremely long idea name that is meant to test the upper limits of the entity extraction system and see how it handles verbose input without truncation or errors.
        Content: Testing how the system handles extremely long entity names.
        
        Label: Task
        Name: B
        Content: A simple test task with a very short name.
        """,
        'expected': [
            ('Collection', 'A'),
            (
                'Idea',
                'This is an extremely long idea name that is meant to test the upper limits of the entity extraction system and see how it handles verbose input without truncation or errors.',
            ),
            ('Task', 'B'),
        ],
    },
    # Example 6: Similar names with different types
    {
        'name': 'Similar names with different types',
        'text': """
        We're planning our refactoring efforts.
        
        Label: Idea
        Name: Refactor
        Content: Conceptual approach to refactoring our codebase.
        
        Label: Task
        Name: Refactor
        Content: Implement the refactoring of specific code components.
        Status: Not Started
        
        Label: Collection
        Name: Refactor
        Content: A collection of all our refactoring initiatives.
        """,
        'expected': [
            ('Idea', 'Refactor'),
            ('Task', 'Refactor'),
            ('Collection', 'Refactor'),
        ],
    },
    # Example 7: Entities with numbers and special formatting
    {
        'name': 'Entities with numbers and special formatting',
        'text': """
        Planning our quarterly initiatives.
        
        Label: Collection
        Name: 2025-Q2
        Content: All initiatives planned for the second quarter of 2025.
        
        Label: Idea
        Name: 5x performance improvement
        Content: Conceptual approach to achieving a five-fold increase in system performance.
        
        Label: Task
        Name: Update API (v2.1)
        Content: Implement updates to version 2.1 of our application programming interface.
        Status: Not Started
        Priority: Medium
        """,
        'expected': [
            ('Collection', '2025-Q2'),
            ('Idea', '5x performance improvement'),
            ('Task', 'Update API (v2.1)'),
        ],
    },
    # Example 8: Entity attributes and multiple fields
    {
        'name': 'Entity attributes and multiple fields',
        'text': """
        Let's track our customer feedback initiative.
        
        Label: Procedure
        Name: Customer Feedback Collection
        Content: Process for gathering and analyzing customer feedback.
        Category: Customer Success
        Priority: High
        Status: Active
        Tags: feedback, customer experience, analysis
        UUID: f47ac10b-58cc-4372-a567-0e02b2c3d479
        """,
        'expected': [
            ('Procedure', 'Customer Feedback Collection'),
        ],
    },
]
