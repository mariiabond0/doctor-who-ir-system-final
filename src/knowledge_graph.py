"""Knowledge graph construction and entity linking helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sentence_transformers import SentenceTransformer, util

try:
    from gliner2 import GLiNER2
    GLINER2_AVAILABLE = True
except ImportError:  # pragma: no cover
    GLINER2_AVAILABLE = False


@dataclass
class EntityMention:
    text: str
    label: str
    start: Optional[int] = None
    end: Optional[int] = None
    confidence: Optional[float] = None
    source_id: Optional[str] = None


@dataclass
class RelationMention:
    subject: str
    object: str
    relation: str
    confidence: Optional[float] = None
    subject_span: Optional[Tuple[int, int]] = None
    object_span: Optional[Tuple[int, int]] = None


@dataclass
class KnowledgeGraph:
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: List[Dict[str, Any]] = field(default_factory=list)

    def add_entity(self, entity_id: str, label: str, aliases: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None):
        if entity_id not in self.nodes:
            self.nodes[entity_id] = {
                "id": entity_id,
                "label": label,
                "aliases": aliases or [],
                "metadata": metadata or {},
            }
        else:
            node = self.nodes[entity_id]
            node["aliases"] = list(set(node["aliases"] + (aliases or [])))
            if metadata:
                node["metadata"].update(metadata)
        return self.nodes[entity_id]

    def add_edge(self, subject_id: str, relation: str, object_id: str, metadata: Optional[Dict[str, Any]] = None):
        self.edges.append({
            "subject": subject_id,
            "relation": relation,
            "object": object_id,
            "metadata": metadata or {},
        })

    def query(self, subject: Optional[str] = None, relation: Optional[str] = None, object: Optional[str] = None):
        return [edge for edge in self.edges
                if (subject is None or edge["subject"] == subject)
                and (relation is None or edge["relation"] == relation)
                and (object is None or edge["object"] == object)]

    def as_triplets(self) -> List[Tuple[str, str, str]]:
        return [(edge["subject"], edge["relation"], edge["object"]) for edge in self.edges]

    def to_dict(self) -> Dict[str, Any]:
        return {"nodes": self.nodes, "edges": self.edges}

    def save_json(self, path: str) -> None:
        import json

        with open(path, "w", encoding="utf-8") as output:
            json.dump(self.to_dict(), output, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        graph = cls()
        graph.nodes = data.get("nodes", {})
        graph.edges = data.get("edges", [])
        return graph

    @classmethod
    def load_json(cls, path: str) -> "KnowledgeGraph":
        import json

        with open(path, "r", encoding="utf-8") as input_file:
            data = json.load(input_file)
        return cls.from_dict(data)


def load_gliner2_model(model_name: str, device: str = "cpu", quantize: bool = False) -> GLiNER2:
    if not GLINER2_AVAILABLE:
        raise ImportError("GliNER2 is not installed. Install it with `pip install gliner2`.")
    return GLiNER2.from_pretrained(model_name, map_location=device, quantize=quantize)


def extract_entities(model: Any, text: str, entity_types: Optional[Sequence[str]] = None, threshold: float = 0.5) -> List[EntityMention]:
    if entity_types is None:
        entity_types = ["PERSON", "ALIEN", "LOCATION", "ORGANIZATION", "DATE", "EVENT", "TIME"]

    response = model.extract_entities(text, entity_types, threshold=threshold, format_results=True, include_confidence=True, include_spans=True)
    entities = response.get("entities", []) if isinstance(response, dict) else []
    return [EntityMention(
        text=ent.get("text", ""),
        label=ent.get("type", ent.get("label", "UNKNOWN")),
        start=ent.get("start"),
        end=ent.get("end"),
        confidence=ent.get("confidence"),
    ) for ent in entities]


def extract_relations(model: Any, text: str, relation_types: Optional[Sequence[str]] = None, threshold: float = 0.5) -> List[RelationMention]:
    if relation_types is None:
        relation_types = ["related_to", "works_for", "located_in", "created_by", "mentions"]

    response = model.extract_relations(text, relation_types, threshold=threshold, format_results=True, include_confidence=True, include_spans=True)
    relations = response.get("relations", []) if isinstance(response, dict) else []
    return [RelationMention(
        subject=rel.get("subject", {}).get("text", ""),
        object=rel.get("object", {}).get("text", ""),
        relation=rel.get("type", rel.get("label", "RELATED_TO")),
        confidence=rel.get("confidence"),
        subject_span=tuple(rel.get("subject", {}).get("span", [])) if rel.get("subject", {}).get("span") else None,
        object_span=tuple(rel.get("object", {}).get("span", [])) if rel.get("object", {}).get("span") else None,
    ) for rel in relations]


def build_alias_index(entity_catalog: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    alias_index: Dict[str, List[str]] = {}
    for entity in entity_catalog:
        entity_id = entity["id"]
        names = [entity.get("name", "")] + entity.get("aliases", [])
        for name in filter(None, names):
            alias_index.setdefault(name.lower(), []).append(entity_id)
    return alias_index


def link_entities(
    mentions: Sequence[EntityMention],
    entity_catalog: Sequence[Dict[str, Any]],
    alias_index: Optional[Dict[str, List[str]]] = None,
    embedding_model: Optional[SentenceTransformer] = None,
    similarity_threshold: float = 0.70,
) -> List[Dict[str, Any]]:
    if alias_index is None:
        alias_index = build_alias_index(entity_catalog)

    if embedding_model is None:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    candidate_texts = [entity["name"] for entity in entity_catalog]
    candidate_ids = [entity["id"] for entity in entity_catalog]
    candidate_embeddings = embedding_model.encode(candidate_texts, convert_to_tensor=True, normalize_embeddings=True)

    resolved = []
    for mention in mentions:
        mention_text = mention.text.strip()
        lower_text = mention_text.lower()
        candidates = alias_index.get(lower_text, [])
        best_id = None
        score = 1.0

        if candidates:
            best_id = candidates[0]
        else:
            mention_embedding = embedding_model.encode(mention_text, convert_to_tensor=True, normalize_embeddings=True)
            similarity_scores = util.cos_sim(mention_embedding, candidate_embeddings)[0]
            best_idx = int(similarity_scores.argmax())
            score = float(similarity_scores[best_idx])
            if score >= similarity_threshold:
                best_id = candidate_ids[best_idx]

        resolved.append({
            "mention": mention_text,
            "label": mention.label,
            "entity_id": best_id,
            "confidence": mention.confidence,
            "similarity": score,
        })

    return resolved


def document_to_graph(
    text: str,
    model: Any,
    entity_types: Optional[Sequence[str]] = None,
    relation_types: Optional[Sequence[str]] = None,
    threshold: float = 0.5,
    source_id: Optional[str] = None,
) -> KnowledgeGraph:
    entities = extract_entities(model, text, entity_types=entity_types, threshold=threshold)
    relations = extract_relations(model, text, relation_types=relation_types, threshold=threshold)

    graph = KnowledgeGraph()
    for entity in entities:
        entity_id = f"{entity.label}:{entity.text}"
        graph.add_entity(entity_id, entity.label, aliases=[entity.text], metadata={"source_id": source_id})

    for relation in relations:
        subject_id = f"{relation.subject}" if relation.subject in graph.nodes else f"TEXT:{relation.subject}"
        object_id = f"{relation.object}" if relation.object in graph.nodes else f"TEXT:{relation.object}"
        graph.add_entity(subject_id, "ENTITY", aliases=[relation.subject], metadata={"source_id": source_id})
        graph.add_entity(object_id, "ENTITY", aliases=[relation.object], metadata={"source_id": source_id})
        graph.add_edge(
            subject_id,
            relation.relation,
            object_id,
            metadata={
                "confidence": relation.confidence,
                "source_id": source_id,
                "subject_text": relation.subject,
                "object_text": relation.object,
            },
        )

    return graph


def get_graph_match_texts(graph: KnowledgeGraph, edge: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    node = graph.nodes.get(edge["subject"], {})
    texts.extend([edge["subject"]] + [alias for alias in node.get("aliases", []) if alias])
    node = graph.nodes.get(edge["object"], {})
    texts.extend([edge["object"]] + [alias for alias in node.get("aliases", []) if alias])
    texts.append(edge["relation"])
    if edge["metadata"].get("subject_text"):
        texts.append(edge["metadata"]["subject_text"])
    if edge["metadata"].get("object_text"):
        texts.append(edge["metadata"]["object_text"])
    return [text.lower() for text in texts if text]


def kg_search(
    query: str,
    graph: KnowledgeGraph,
    model: Any,
    entity_types: Optional[Sequence[str]] = None,
    relation_types: Optional[Sequence[str]] = None,
    threshold: float = 0.5,
    top_k: int = 5,
):
    query_lower = query.lower().strip()
    mentions = extract_entities(model, query, entity_types=entity_types, threshold=threshold)
    mention_texts = [mention.text.lower() for mention in mentions if mention.text]
    matched_edges: List[Dict[str, Any]] = []

    for edge in graph.edges:
        edge_texts = get_graph_match_texts(graph, edge)
        if any(query_lower in text for text in edge_texts):
            matched_edges.append(edge)
            continue

        if any(any(mention in text for text in edge_texts) for mention in mention_texts):
            matched_edges.append(edge)

    if not matched_edges and mention_texts:
        for edge in graph.edges:
            edge_texts = get_graph_match_texts(graph, edge)
            if any(mention == text for text in edge_texts for mention in mention_texts):
                matched_edges.append(edge)

    doc_scores: Dict[str, int] = {}
    for edge in matched_edges:
        source_id = edge.get("metadata", {}).get("source_id")
        if source_id:
            doc_scores[source_id] = doc_scores.get(source_id, 0) + 1

    sorted_docs = [doc for doc, _ in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)]
    return sorted_docs[:top_k], matched_edges


def summarize_kg_results(query: str, top_docs: List[str], matched_edges: List[Dict[str, Any]], max_edges: int = 5) -> Dict[str, Any]:
    summary = {
        "query": query,
        "found_docs": top_docs,
        "matched_edge_count": len(matched_edges),
        "top_edges": [],
    }

    for edge in matched_edges[:max_edges]:
        summary["top_edges"].append({
            "subject": edge["subject"],
            "relation": edge["relation"],
            "object": edge["object"],
            "confidence": edge.get("metadata", {}).get("confidence"),
            "source_id": edge.get("metadata", {}).get("source_id"),
        })

    return summary


def merge_graphs(graphs: Sequence[KnowledgeGraph]) -> KnowledgeGraph:
    merged = KnowledgeGraph()
    for graph in graphs:
        for node_id, node_data in graph.nodes.items():
            merged.add_entity(node_id, node_data["label"], aliases=node_data.get("aliases", []), metadata=node_data.get("metadata", {}))
        for edge in graph.edges:
            merged.add_edge(edge["subject"], edge["relation"], edge["object"], metadata=edge.get("metadata", {}))
    return merged