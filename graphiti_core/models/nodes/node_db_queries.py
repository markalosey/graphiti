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

EPISODIC_NODE_SAVE = """
        MERGE (n:Episodic {uuid: $uuid})
        SET n = {uuid: $uuid, name: $name, group_id: $group_id, source_description: $source_description, source: $source, content: $content, 
        entity_edges: $entity_edges, created_at: $created_at, valid_at: $valid_at}
        RETURN n.uuid AS uuid"""

EPISODIC_NODE_SAVE_BULK = """
    UNWIND $episodes AS episode
    MERGE (n:Episodic {uuid: episode.uuid})
    SET n = {uuid: episode.uuid, name: episode.name, group_id: episode.group_id, source_description: episode.source_description, 
        source: episode.source, content: episode.content, 
        entity_edges: episode.entity_edges, created_at: episode.created_at, valid_at: episode.valid_at,
        summary_text: episode.summary_text}
    RETURN n.uuid AS uuid
"""

ENTITY_NODE_SAVE = """
    MERGE (n:Entity {uuid: $uuid_param})
    ON CREATE SET n = $entity_props, n.created_at = $entity_props.created_at
    ON MATCH SET n += $entity_props
    // Set labels using APOC if available, ensuring :Entity is always present
    WITH n, $labels AS labelList, $entity_props AS props, $name_embedding_param AS embedding
    // Ensure 'Entity' is part of the label list for APOC
    WITH n, [label IN labelList WHERE label <> 'Entity'] + ['Entity'] AS finalLabels, props, embedding
    CALL apoc.create.addLabels(n, finalLabels) YIELD node AS n_labeled
    // Explicitly set/update the vector property
    WITH n_labeled, embedding
    CALL db.create.setNodeVectorProperty(n_labeled, "name_embedding", embedding)
    RETURN n_labeled.uuid AS uuid
"""

ENTITY_NODE_SAVE_BULK = """
    UNWIND $nodes AS node_map // Each item in $nodes is a map representing an EntityNode
    MERGE (n:Entity {uuid: node_map.uuid})
    // Set core properties directly from the map
    SET n.name = node_map.name,
        n.group_id = node_map.group_id,
        n.summary = node_map.summary,
        n.created_at = node_map.created_at
    // Add properties from the 'attributes' dictionary within the map
    WITH n, node_map // Ensure node_map is carried forward
    FOREACH (key IN keys(node_map.attributes) |
        SET n[key] = node_map.attributes[key]
    )
    // Set labels using APOC, ensuring :Entity is always present
    WITH n, node_map.labels AS labelList, node_map.name_embedding AS embedding
    WITH n, [label IN labelList WHERE label <> 'Entity'] + ['Entity'] AS finalLabels, embedding
    CALL apoc.create.addLabels(n, finalLabels) YIELD node AS n_labeled
    // Explicitly set/update the vector property
    WITH n_labeled, embedding
    CALL db.create.setNodeVectorProperty(n_labeled, "name_embedding", embedding)
    RETURN n_labeled.uuid AS uuid
"""

COMMUNITY_NODE_SAVE = """
        MERGE (n:Community {uuid: $uuid})
        SET n = {uuid: $uuid, name: $name, group_id: $group_id, summary: $summary, created_at: $created_at}
        WITH n CALL db.create.setNodeVectorProperty(n, "name_embedding", $name_embedding)
        RETURN n.uuid AS uuid"""
