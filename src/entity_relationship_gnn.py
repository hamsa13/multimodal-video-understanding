"""
Entity Relationship GNN Module
Graph Neural Network for modeling entity and object relationships in video
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import base64
from io import BytesIO

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

# Try to import visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available - graph visualization disabled")


@dataclass
class Entity:
    """Represents an entity (person, object, location) in the video"""
    id: str
    name: str
    entity_type: str  # 'person', 'object', 'location', 'speaker'
    attributes: Dict[str, Any] = field(default_factory=dict)
    appearances: List[Tuple[float, float]] = field(default_factory=list)  # (start, end) times
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return self.id == other.id


@dataclass  
class Relationship:
    """Represents a relationship between two entities"""
    source: str  # entity id
    target: str  # entity id
    relation_type: str  # 'interacts_with', 'speaks_to', 'uses', 'near', 'in_scene_with'
    weight: float = 1.0
    scenes: List[int] = field(default_factory=list)
    timestamps: List[Tuple[float, float]] = field(default_factory=list)
    
    def __hash__(self):
        return hash((self.source, self.target, self.relation_type))


class EntityRelationshipGNN:
    """
    Graph Neural Network-based entity relationship modeling
    
    Creates a knowledge graph of entities (people, objects, locations) and their
    relationships based on video analysis data.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        
        # Node type colors for visualization
        self.node_colors = {
            'person': '#FF6B6B',      # Red
            'speaker': '#4ECDC4',     # Teal
            'object': '#45B7D1',      # Blue
            'location': '#96CEB4',    # Green
            'action': '#FFEAA7',      # Yellow
        }
        
        # Edge type styles
        self.edge_styles = {
            'speaks_to': {'color': '#E74C3C', 'style': 'solid', 'width': 2.5},
            'interacts_with': {'color': '#3498DB', 'style': 'solid', 'width': 2.0},
            'uses': {'color': '#9B59B6', 'style': 'dashed', 'width': 1.5},
            'near': {'color': '#95A5A6', 'style': 'dotted', 'width': 1.0},
            'in_scene_with': {'color': '#BDC3C7', 'style': 'dotted', 'width': 0.8},
            'performs': {'color': '#F39C12', 'style': 'solid', 'width': 2.0},
            'located_in': {'color': '#27AE60', 'style': 'dashed', 'width': 1.5},
        }
    
    def build_graph_from_analysis(self, analysis_data: Dict[str, Any]) -> nx.DiGraph:
        """
        Build entity relationship graph from video analysis data
        
        Args:
            analysis_data: The analysis.json data structure
            
        Returns:
            NetworkX directed graph with entities and relationships
        """
        self.graph.clear()
        self.entities.clear()
        self.relationships.clear()
        
        scenes = analysis_data.get('scenes', [])
        summary = analysis_data.get('summary', {})
        
        # Extract all entities
        self._extract_speakers(scenes, summary)
        self._extract_people(scenes)
        self._extract_objects(scenes, summary)
        self._extract_locations(scenes, summary)
        self._extract_actions(scenes, summary)
        
        # Build relationships
        self._build_speaker_relationships(scenes)
        self._build_person_object_relationships(scenes)
        self._build_spatial_relationships(scenes)
        self._build_temporal_cooccurrence(scenes)
        self._build_action_relationships(scenes)
        
        # Add nodes to graph
        for entity_id, entity in self.entities.items():
            self.graph.add_node(
                entity_id,
                name=entity.name,
                type=entity.entity_type,
                attributes=entity.attributes,
                appearances=len(entity.appearances)
            )
        
        # Add edges to graph
        for rel in self.relationships:
            if rel.source in self.graph and rel.target in self.graph:
                self.graph.add_edge(
                    rel.source,
                    rel.target,
                    relation=rel.relation_type,
                    weight=rel.weight,
                    scenes=rel.scenes
                )
        
        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def _extract_speakers(self, scenes: List[Dict], summary: Dict):
        """Extract speaker entities from scenes"""
        speakers = set(summary.get('speakers', []))
        
        for scene in scenes:
            for dlg in scene.get('dialogue', []):
                speaker = dlg.get('speaker', '')
                if speaker and speaker != 'Unknown':
                    speakers.add(speaker)
        
        for speaker in speakers:
            entity_id = f"speaker_{speaker.lower().replace(' ', '_')}"
            self.entities[entity_id] = Entity(
                id=entity_id,
                name=speaker,
                entity_type='speaker',
                attributes={'role': 'speaker'}
            )
    
    def _extract_people(self, scenes: List[Dict]):
        """Extract person entities from scenes"""
        for scene in scenes:
            for person in scene.get('people', []):
                if isinstance(person, dict):
                    name = person.get('name', 'Unknown')
                else:
                    name = str(person)
                
                if name and name != 'Unknown':
                    entity_id = f"person_{name.lower().replace(' ', '_')}"
                    if entity_id not in self.entities:
                        self.entities[entity_id] = Entity(
                            id=entity_id,
                            name=name,
                            entity_type='person',
                            attributes={}
                        )
                    
                    # Record appearance
                    scene_time = scene.get('time', {})
                    start = scene_time.get('start', 0)
                    end = scene_time.get('end', 0)
                    self.entities[entity_id].appearances.append((start, end))
    
    def _extract_objects(self, scenes: List[Dict], summary: Dict):
        """Extract object entities"""
        # Get key objects from summary
        key_objects = set(summary.get('key_objects', []))
        
        # Also extract from scenes
        for scene in scenes:
            for obj in scene.get('objects', []):
                key_objects.add(obj)
        
        for obj in key_objects:
            entity_id = f"object_{obj.lower().replace(' ', '_')}"
            if entity_id not in self.entities:
                self.entities[entity_id] = Entity(
                    id=entity_id,
                    name=obj,
                    entity_type='object',
                    attributes={}
                )
    
    def _extract_locations(self, scenes: List[Dict], summary: Dict):
        """Extract location entities"""
        locations = set(summary.get('locations', []))
        
        for scene in scenes:
            loc = scene.get('location', '')
            if loc:
                locations.add(loc)
        
        for loc in locations:
            entity_id = f"location_{loc.lower().replace(' ', '_').replace(',', '')}"
            if entity_id not in self.entities:
                self.entities[entity_id] = Entity(
                    id=entity_id,
                    name=loc,
                    entity_type='location',
                    attributes={}
                )
    
    def _extract_actions(self, scenes: List[Dict], summary: Dict):
        """Extract action entities"""
        actions = set(summary.get('key_actions', []))
        
        for scene in scenes:
            for action in scene.get('actions', []):
                actions.add(action)
            if scene.get('dominant_action'):
                actions.add(scene['dominant_action'])
        
        for action in list(actions)[:10]:  # Limit to top 10 actions
            entity_id = f"action_{action.lower().replace(' ', '_')}"
            if entity_id not in self.entities:
                self.entities[entity_id] = Entity(
                    id=entity_id,
                    name=action,
                    entity_type='action',
                    attributes={}
                )
    
    def _build_speaker_relationships(self, scenes: List[Dict]):
        """Build relationships between speakers based on dialogue"""
        for scene in scenes:
            scene_num = scene.get('scene', 0)
            dialogues = scene.get('dialogue', [])
            
            # Find consecutive speakers (conversation)
            speakers_in_scene = []
            for dlg in dialogues:
                speaker = dlg.get('speaker', '')
                if speaker and speaker != 'Unknown':
                    speakers_in_scene.append(speaker)
            
            # Create speaks_to relationships for consecutive speakers
            for i in range(len(speakers_in_scene) - 1):
                s1 = speakers_in_scene[i]
                s2 = speakers_in_scene[i + 1]
                
                if s1 != s2:
                    src_id = f"speaker_{s1.lower().replace(' ', '_')}"
                    tgt_id = f"speaker_{s2.lower().replace(' ', '_')}"
                    
                    self._add_relationship(src_id, tgt_id, 'speaks_to', scene_num)
    
    def _build_person_object_relationships(self, scenes: List[Dict]):
        """Build relationships between people and objects in scenes"""
        for scene in scenes:
            scene_num = scene.get('scene', 0)
            objects = scene.get('objects', [])
            people = scene.get('people', [])
            dialogues = scene.get('dialogue', [])
            
            # Get speakers and people in this scene
            speaker_ids = set()
            for dlg in dialogues:
                speaker = dlg.get('speaker', '')
                if speaker and speaker != 'Unknown':
                    speaker_ids.add(f"speaker_{speaker.lower().replace(' ', '_')}")
            
            person_ids = set()
            for person in people:
                name = person.get('name', str(person)) if isinstance(person, dict) else str(person)
                if name and name != 'Unknown':
                    person_ids.add(f"person_{name.lower().replace(' ', '_')}")
            
            # Connect speakers/people to objects in scene (uses relationship)
            all_people = speaker_ids | person_ids
            for person_id in all_people:
                for obj in objects[:5]:  # Top 5 objects
                    obj_id = f"object_{obj.lower().replace(' ', '_')}"
                    if obj_id in self.entities:
                        self._add_relationship(person_id, obj_id, 'uses', scene_num, weight=0.5)
    
    def _build_spatial_relationships(self, scenes: List[Dict]):
        """Build location relationships"""
        for scene in scenes:
            scene_num = scene.get('scene', 0)
            location = scene.get('location', '')
            
            if not location:
                continue
            
            loc_id = f"location_{location.lower().replace(' ', '_').replace(',', '')}"
            
            # Connect speakers to location
            for dlg in scene.get('dialogue', []):
                speaker = dlg.get('speaker', '')
                if speaker and speaker != 'Unknown':
                    speaker_id = f"speaker_{speaker.lower().replace(' ', '_')}"
                    self._add_relationship(speaker_id, loc_id, 'located_in', scene_num)
            
            # Connect objects to location
            for obj in scene.get('objects', [])[:3]:
                obj_id = f"object_{obj.lower().replace(' ', '_')}"
                if obj_id in self.entities:
                    self._add_relationship(obj_id, loc_id, 'located_in', scene_num, weight=0.3)
    
    def _build_temporal_cooccurrence(self, scenes: List[Dict]):
        """Build co-occurrence relationships (entities appearing together)"""
        for scene in scenes:
            scene_num = scene.get('scene', 0)
            
            # Get all entities in this scene
            entities_in_scene = set()
            
            for dlg in scene.get('dialogue', []):
                speaker = dlg.get('speaker', '')
                if speaker and speaker != 'Unknown':
                    entities_in_scene.add(f"speaker_{speaker.lower().replace(' ', '_')}")
            
            for person in scene.get('people', []):
                name = person.get('name', str(person)) if isinstance(person, dict) else str(person)
                if name and name != 'Unknown':
                    entities_in_scene.add(f"person_{name.lower().replace(' ', '_')}")
            
            # Create in_scene_with relationships
            entities_list = list(entities_in_scene)
            for i, e1 in enumerate(entities_list):
                for e2 in entities_list[i+1:]:
                    if e1 in self.entities and e2 in self.entities:
                        self._add_relationship(e1, e2, 'in_scene_with', scene_num, weight=0.3)
    
    def _build_action_relationships(self, scenes: List[Dict]):
        """Build relationships between entities and actions"""
        for scene in scenes:
            scene_num = scene.get('scene', 0)
            actions = scene.get('actions', [])
            dominant_action = scene.get('dominant_action', '')
            
            if dominant_action:
                actions = [dominant_action] + [a for a in actions if a != dominant_action]
            
            # Get speakers in scene
            speakers_in_scene = set()
            for dlg in scene.get('dialogue', []):
                speaker = dlg.get('speaker', '')
                if speaker and speaker != 'Unknown':
                    speakers_in_scene.add(f"speaker_{speaker.lower().replace(' ', '_')}")
            
            # Connect speakers to actions
            for speaker_id in speakers_in_scene:
                for action in actions[:2]:  # Top 2 actions
                    action_id = f"action_{action.lower().replace(' ', '_')}"
                    if action_id in self.entities:
                        self._add_relationship(speaker_id, action_id, 'performs', scene_num)
    
    def _add_relationship(self, source: str, target: str, rel_type: str, scene_num: int, weight: float = 1.0):
        """Add or update a relationship"""
        # Check if relationship exists
        for rel in self.relationships:
            if rel.source == source and rel.target == target and rel.relation_type == rel_type:
                rel.weight += weight
                if scene_num not in rel.scenes:
                    rel.scenes.append(scene_num)
                return
        
        # Create new relationship
        self.relationships.append(Relationship(
            source=source,
            target=target,
            relation_type=rel_type,
            weight=weight,
            scenes=[scene_num]
        ))
    
    def compute_node_features(self) -> Dict[str, np.ndarray]:
        """
        Compute GNN-style node features using graph properties
        
        Returns:
            Dictionary mapping node IDs to feature vectors
        """
        features = {}
        
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            
            # Basic features
            in_degree = self.graph.in_degree(node)
            out_degree = self.graph.out_degree(node)
            
            # Centrality measures
            try:
                pagerank = nx.pagerank(self.graph).get(node, 0)
            except:
                pagerank = 0
            
            try:
                betweenness = nx.betweenness_centrality(self.graph).get(node, 0)
            except:
                betweenness = 0
            
            # Type encoding
            type_encoding = {
                'person': [1, 0, 0, 0, 0],
                'speaker': [0, 1, 0, 0, 0],
                'object': [0, 0, 1, 0, 0],
                'location': [0, 0, 0, 1, 0],
                'action': [0, 0, 0, 0, 1],
            }
            type_vec = type_encoding.get(node_data.get('type', ''), [0, 0, 0, 0, 0])
            
            # Combine features
            feature_vec = np.array([
                in_degree,
                out_degree,
                pagerank,
                betweenness,
                node_data.get('appearances', 0),
            ] + type_vec)
            
            features[node] = feature_vec
        
        return features
    
    def get_important_entities(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get most important entities based on graph centrality"""
        if not self.graph.nodes():
            return []
        
        try:
            pagerank = nx.pagerank(self.graph)
        except:
            pagerank = {n: 1.0 / len(self.graph) for n in self.graph.nodes()}
        
        sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        
        result = []
        for node_id, score in sorted_nodes[:top_k]:
            node_data = self.graph.nodes[node_id]
            result.append({
                'id': node_id,
                'name': node_data.get('name', node_id),
                'type': node_data.get('type', 'unknown'),
                'importance': round(score, 4),
                'connections': self.graph.degree(node_id)
            })
        
        return result
    
    def get_relationship_summary(self) -> Dict[str, Any]:
        """Get summary of all relationships"""
        rel_counts = defaultdict(int)
        for rel in self.relationships:
            rel_counts[rel.relation_type] += 1
        
        return {
            'total_entities': len(self.entities),
            'total_relationships': len(self.relationships),
            'entity_types': {
                'speakers': len([e for e in self.entities.values() if e.entity_type == 'speaker']),
                'people': len([e for e in self.entities.values() if e.entity_type == 'person']),
                'objects': len([e for e in self.entities.values() if e.entity_type == 'object']),
                'locations': len([e for e in self.entities.values() if e.entity_type == 'location']),
                'actions': len([e for e in self.entities.values() if e.entity_type == 'action']),
            },
            'relationship_types': dict(rel_counts),
            'graph_density': nx.density(self.graph) if self.graph.nodes() else 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Export graph data as dictionary"""
        return {
            'nodes': [
                {
                    'id': node,
                    'name': self.graph.nodes[node].get('name', node),
                    'type': self.graph.nodes[node].get('type', 'unknown'),
                    'appearances': self.graph.nodes[node].get('appearances', 0)
                }
                for node in self.graph.nodes()
            ],
            'edges': [
                {
                    'source': u,
                    'target': v,
                    'relation': self.graph.edges[u, v].get('relation', 'unknown'),
                    'weight': self.graph.edges[u, v].get('weight', 1.0),
                    'scenes': self.graph.edges[u, v].get('scenes', [])
                }
                for u, v in self.graph.edges()
            ],
            'summary': self.get_relationship_summary(),
            'important_entities': self.get_important_entities()
        }
    
    def visualize(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 12),
        title: str = "Entity Relationship Graph"
    ) -> Optional[str]:
        """
        Visualize the entity relationship graph
        
        Args:
            output_path: Path to save the image (optional)
            figsize: Figure size
            title: Graph title
            
        Returns:
            Base64 encoded image string if no output_path, else None
        """
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available for visualization")
            return None
        
        if not self.graph.nodes():
            logger.warning("Empty graph - nothing to visualize")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Layout
        try:
            pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        except:
            pos = nx.circular_layout(self.graph)
        
        # Draw nodes by type
        for node_type, color in self.node_colors.items():
            nodes = [n for n in self.graph.nodes() if self.graph.nodes[n].get('type') == node_type]
            if nodes:
                # Size based on importance
                try:
                    pr = nx.pagerank(self.graph)
                    sizes = [max(500, pr.get(n, 0.01) * 10000) for n in nodes]
                except:
                    sizes = [700] * len(nodes)
                
                nx.draw_networkx_nodes(
                    self.graph, pos,
                    nodelist=nodes,
                    node_color=color,
                    node_size=sizes,
                    alpha=0.9,
                    ax=ax
                )
        
        # Draw edges by type
        for rel_type, style in self.edge_styles.items():
            edges = [(u, v) for u, v in self.graph.edges() 
                     if self.graph.edges[u, v].get('relation') == rel_type]
            if edges:
                weights = [self.graph.edges[u, v].get('weight', 1.0) for u, v in edges]
                max_weight = max(weights) if weights else 1
                widths = [style['width'] * (w / max_weight) for w in weights]
                
                nx.draw_networkx_edges(
                    self.graph, pos,
                    edgelist=edges,
                    edge_color=style['color'],
                    style=style['style'],
                    width=widths,
                    alpha=0.6,
                    arrows=True,
                    arrowsize=15,
                    ax=ax
                )
        
        # Draw labels
        labels = {n: self.graph.nodes[n].get('name', n)[:15] for n in self.graph.nodes()}
        nx.draw_networkx_labels(
            self.graph, pos,
            labels=labels,
            font_size=8,
            font_weight='bold',
            ax=ax
        )
        
        # Legend
        legend_elements = []
        for node_type, color in self.node_colors.items():
            nodes_of_type = [n for n in self.graph.nodes() if self.graph.nodes[n].get('type') == node_type]
            if nodes_of_type:
                legend_elements.append(mpatches.Patch(color=color, label=f'{node_type.title()} ({len(nodes_of_type)})'))
        
        # Add edge type legend
        for rel_type, style in self.edge_styles.items():
            edges_of_type = [(u, v) for u, v in self.graph.edges() 
                           if self.graph.edges[u, v].get('relation') == rel_type]
            if edges_of_type:
                legend_elements.append(mpatches.Patch(
                    color=style['color'], 
                    label=f'{rel_type.replace("_", " ").title()} ({len(edges_of_type)})'
                ))
        
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        # Title and styling
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            logger.info(f"Saved graph visualization to {output_path}")
            return None
        else:
            # Return as base64
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            return img_base64
    
    def get_visualization_html(self) -> str:
        """Get interactive HTML visualization using vis.js"""
        nodes_data = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            nodes_data.append({
                'id': node,
                'label': node_data.get('name', node)[:20],
                'group': node_data.get('type', 'unknown'),
                'title': f"{node_data.get('name', node)}\nType: {node_data.get('type', 'unknown')}\nAppearances: {node_data.get('appearances', 0)}"
            })
        
        edges_data = []
        for u, v in self.graph.edges():
            edge_data = self.graph.edges[u, v]
            edges_data.append({
                'from': u,
                'to': v,
                'label': edge_data.get('relation', '').replace('_', ' '),
                'arrows': 'to',
                'color': self.edge_styles.get(edge_data.get('relation', ''), {}).get('color', '#999')
            })
        
        html = f"""
        <div id="entity-graph" style="width: 100%; height: 600px; border: 1px solid #ddd; border-radius: 8px;"></div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.6/dist/vis-network.min.js"></script>
        <script>
            var nodes = new vis.DataSet({json.dumps(nodes_data)});
            var edges = new vis.DataSet({json.dumps(edges_data)});
            
            var container = document.getElementById('entity-graph');
            var data = {{ nodes: nodes, edges: edges }};
            var options = {{
                groups: {{
                    speaker: {{ color: {{ background: '#4ECDC4' }}, shape: 'dot', size: 25 }},
                    person: {{ color: {{ background: '#FF6B6B' }}, shape: 'dot', size: 20 }},
                    object: {{ color: {{ background: '#45B7D1' }}, shape: 'square', size: 15 }},
                    location: {{ color: {{ background: '#96CEB4' }}, shape: 'diamond', size: 20 }},
                    action: {{ color: {{ background: '#FFEAA7' }}, shape: 'triangle', size: 15 }}
                }},
                physics: {{
                    stabilization: {{ iterations: 100 }},
                    barnesHut: {{ gravitationalConstant: -2000, springLength: 150 }}
                }},
                interaction: {{ hover: true, tooltipDelay: 100 }}
            }};
            
            var network = new vis.Network(container, data, options);
        </script>
        """
        return html


def build_entity_graph(analysis_data: Dict[str, Any], output_dir: Optional[str] = None) -> Tuple[Dict, Optional[str]]:
    """
    Convenience function to build entity graph and generate visualization
    
    Args:
        analysis_data: Video analysis data
        output_dir: Optional directory to save visualization
        
    Returns:
        Tuple of (graph_data_dict, visualization_base64_or_path)
    """
    gnn = EntityRelationshipGNN()
    gnn.build_graph_from_analysis(analysis_data)
    
    graph_data = gnn.to_dict()
    
    # Generate visualization
    viz_result = None
    if output_dir:
        viz_path = str(Path(output_dir) / "entity_graph.png")
        gnn.visualize(output_path=viz_path)
        viz_result = viz_path
    else:
        viz_result = gnn.visualize()
    
    return graph_data, viz_result
