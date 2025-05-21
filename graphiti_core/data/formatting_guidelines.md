## Graphiti Episode Formatting Guidelines:

**Core Principle:** Episodes should be atomic and focused. Aim for 1-4 facts per episode where possible. If an interaction covers multiple distinct topics or results in many new facts, break it down into multiple, more targeted episodes.

**1. Episode Structure (`body_content`):**
    - Use `Label: <EntityType>` as the first line to define the primary entity type of the episode's main subject.
        - Common `EntityType` examples: `Task`, `Idea`, `Requirement`, `Preference`, `BugReport`, `Decision`, `SystemPrinciple`, `UserFeedback`, `MeetingNotes`, `DocumentationSnippet`, `CodeReviewComment`, `File`, `Function`, `Class`, `Module`, `Commit`, `PullRequest`, `Issue`, `Hypothesis`, `Evidence`, `Experiment`, `Result`, `Persona`, `Goal`, `Constraint`, `Metric`, `Feature`, `UserStory`, `Summary`, `Question`, `Answer`, `Clarification`, `Correction`, `Confirmation`, `TutorialStep`.
        - Custom entity types are allowed and encouraged for domain specificity.
    - Follow with `Name: <Name of the entity>` on the second line. This should be a concise, descriptive name.
    - Use `Content: <Detailed textual content or description>` for the main body. This can be multi-line.
    - For structured data or specific attributes, use `Key: Value` pairs on subsequent lines.
        - **Common Keys:**
            - `UUID: <UUID of the entity if updating or referencing>` (Essential for updates)
            - `Status: <e.g., ToDo, InProgress, Done, Approved, Rejected, Deprecated, Active, Inactive, Blocked, Open, Closed, Merged>`
            - `Priority: <e.g., High, Medium, Low, Critical>`
            - `Category: <e.g., FeatureDevelopment, BugFix, Documentation, Testing, Research, Planning, Refactoring, Architecture, UX, UI, Security, Performance, DataMigration, SystemImprovement, UserRequest, AgentTask>` (Use comma-separated for multiple, e.g., `Category: FeatureDevelopment, UX`)
            - `Source: <Origin of the information, e.g., UserInput, SystemLog, Document:[path/to/doc], Meeting:[YYYY-MM-DD Subject], Commit:[SHA], PR:[URL or #ID], Issue:[URL or #ID]>`
            - `Assignee: <username or ID>`
            - `DueDate: <YYYY-MM-DD>`
            - `Version: <e.g., 1.0.2, "main" branch>`
            - `Context: <Brief description of the surrounding context if not obvious>`
            - `Filepath: <path/to/relevant/file>`
            - `FunctionSignature: <full function signature>`
            - `CodeSnippet: <A few lines of relevant code. Use triple backticks for multi-line.>`
            - `User: <Username of the interacting user, if applicable>`
            - `Agent: <Name/ID of the AI agent, if applicable>`
            - `Timestamp: <ISO 8601 Timestamp, e.g., 2023-10-27T10:30:00Z>`
            - `ErrorLog: <Relevant error message or stack trace>`
            - `UserStoryPoints: <Number>`
            - `EffortEstimate: <e.g., "2 hours", "3 days">`
            - `Impact: <e.g., High, Medium, Low>`
            - `Confidence: <e.g., High, Medium, Low, Percentage>`
            - `Dependencies: <Comma-separated list of Task names or UUIDs this item depends on>` (Superseded by `Relationships`)
            - `Blocks: <Comma-separated list of Task names or UUIDs this item blocks>` (Superseded by `Relationships`)
            - `Project: <Project Name/ID>`
            - `Tags: <Comma-separated list of relevant tags/keywords>`
    - **Relationships:** Use `Relationships: <RelationshipType>: <TargetEntityName> (<TargetEntityUUID_or_NameIfUUIDUnknown>)` for defining connections.
        - One relationship per line if multiple.
        - Example: `Relationships: RelatedTo: Some Other Idea (f123...)`
        - Example: `Relationships: DependsOn: Task: Setup Database`
        - Example: `Relationships: PartOf: Collection: MCP Helpers`
        - **Common RelationshipTypes:** `RelatedTo`, `DependsOn` (source depends on target), `Blocks` (source blocks target), `Implies`, `PartOf` (source is part of target collection/epic), `HasPart` (source has target as a sub-component), `CreatedBy`, `ModifiedBy`, `Mentions`, `Affects`, `Resolves`, `DuplicateOf`, `BasedOn`, `DerivedFrom`, `References`, `AlternativeTo`, `TestedBy`, `Exemplifies` (source is an example of target concept), `Contradicts`.

**2. `episode_name`:**
    - A concise summary of the episode's purpose, often mirroring the `Name:` field of the primary entity.
    - Example: `"Task: Implement User Login Endpoint"`
    - Example: `"Idea: Gamify Onboarding Flow"`
    - Example: `"Update Task Status: Implement User Login Endpoint"`

**3. `target_group_id`:**
    - Specifies the project or context for the episode. E.g., `"project-nanoo-project-nanoo-main"`.
    - This is crucial for data partitioning and relevance.

**4. General Best Practices:**
    - **Be Explicit:** Clearly state the nature of the information.
    - **Be Atomic:** Prefer smaller, focused episodes. One primary entity or update per episode.
    - **Reference by UUID when possible:** When referring to existing entities in `Relationships` or other fields, use their UUID if known. If not known, use the exact Name and the system will attempt to resolve it.
    - **Timestamping:** While the system timestamps episodes, including a `Timestamp` field in `body_content` can be useful if the event occurred at a different time than its recording.
    - **Updates vs. New:** If an episode is intended to update an existing node, ensure the `UUID:` field in `body_content` matches the UUID of the node to be updated. The system will treat it as an update. Otherwise, it will be a new node.
    - **Clarity over brevity if necessary, but aim for conciseness.**
    - **Use Markdown for `Content` and other free-text fields if it enhances readability (e.g., lists, bolding).**

These guidelines will help ensure that data added to Graphiti is well-structured, easily searchable, and effectively contributes to the agent's knowledge and memory. 