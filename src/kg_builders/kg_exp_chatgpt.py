import json
import re
import spacy
import networkx as nx

from gliner import GLiNER
from pyvis.network import Network
from config import CORPUS_PATH

import os

os.makedirs("graph_depo", exist_ok=True)

# -------------------------
# Load corpus
# -------------------------
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)

# -------------------------
# NLP
# -------------------------
nlp = spacy.load("en_core_web_trf")

if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

# -------------------------
# GLiNER
# -------------------------
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

GLINER_LABELS = [
    "person",
    "location",
    "organization",
    "date",
    "time",
    "alien species",
    "spaceship",
    "planet",
    "artifact",
]

# -------------------------
# Labels / Aliases
# -------------------------
STANDARD_LABELS = {
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "LOCATION": "LOCATION",
    "DATE": "DATE",
    "TIME": "TIME",
    "ALIEN SPECIES": "ALIEN",
    "ALIEN": "ALIEN",
    "SPACESHIP": "VEHICLE",
    "PLANET": "LOCATION",
    "ARTIFACT": "ARTIFACT",
}

ALIASES = {
    "the doctor": "doctor",
    "doctor who": "doctor",
    "amy": "amy pond",
    "amelia": "amy pond",
    "clara": "clara oswald",
    "rose": "rose tyler",
    "martha": "martha jones",
    "donna": "donna noble",
    "river": "river song",
    "the master": "master",
    "daleks": "daleks",
    "cybermen": "cybermen",
    "weeping angels": "weeping angels",
}

STOP_ENTITIES = {
    "one", "two", "they", "them", "him", "her", "it",
    "something", "anything", "people", "human", "humans"
}

BAD_RELATIONS = {
    "be", "have", "do", "say", "go", "get",
    "make", "take", "want", "need", "try"
}

RELATION_MAP = {
    "fight": "attack",
    "fought": "attack",
    "fighting": "attack",
    "attack": "attack",
    "attacked": "attack",

    "go": "move",
    "went": "move",
    "travel": "move",
    "travelled": "move",

    "see": "observe",
    "saw": "observe",
    "look": "observe",

    "say": "communicate",
    "said": "communicate",
    "tell": "communicate",

    "be": "is",
    "was": "is",
    "were": "is",
}

# -------------------------
# Helpers
# -------------------------
def normalize_label(label):
    return STANDARD_LABELS.get(label.upper(), label.upper())


def canonical(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"^the\s+", "", text)
    text = " ".join(text.split())
    return ALIASES.get(text, text)


def normalize_relation(verb: str) -> str:
    verb = verb.lower().strip()
    verb = re.sub(r"[^a-z]", "", verb)
    return RELATION_MAP.get(verb, verb)

# -------------------------
# Entity extraction
# -------------------------
def extract_entities(doc, gliner_entities):

    entity_dict = {}
    token_lookup = {}

    def add(text, label, start=None, end=None):

        key = canonical(text)

        if key in STOP_ENTITIES or len(key) <= 1:
            return

        label = normalize_label(label).upper()

        entity_dict.setdefault(key, set()).add(label)

        if start is not None and end is not None:
            for i in range(start, end):
                token_lookup[i] = key

    for ent in doc.ents:
        add(ent.text, ent.label_, ent.start, ent.end)

    for ent in gliner_entities:
        if ent.get("score", 1.0) < 0.55:
            continue

        span = doc.char_span(ent["start"], ent["end"], alignment_mode="expand")
        if span is None:
            continue

        add(span.text, ent["label"], span.start, span.end)

    return entity_dict, token_lookup

# -------------------------
# Dependency helpers
# -------------------------
def expand_conj(token):
    items = [token]
    items.extend([c for c in token.children if c.dep_ == "conj"])
    return items


def find_entity(token, token_lookup):
    return token_lookup.get(token.i)

# -------------------------
# Relations (FIXED + PASSIVE HANDLING)
# -------------------------
def extract_relations(doc, token_lookup):

    relations = set()

    for sent in doc.sents:

        for token in sent:

            if token.pos_ != "VERB":
                continue

            verb = normalize_relation(token.lemma_)

            if verb in BAD_RELATIONS:
                continue

            subjects = []
            objects = []
            passive = False

            for child in token.children:

                if child.dep_ in ("nsubj", "nsubjpass"):
                    subjects.extend(expand_conj(child))
                    if child.dep_ == "nsubjpass":
                        passive = True

                elif child.dep_ in ("dobj", "attr", "dative", "oprd"):
                    objects.extend(expand_conj(child))

                elif child.dep_ == "prep":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            objects.extend(expand_conj(pobj))

            for s in subjects:
                s_ent = find_entity(s, token_lookup)
                if not s_ent:
                    continue

                for o in objects:
                    o_ent = find_entity(o, token_lookup)
                    if not o_ent:
                        continue

                    if s_ent == o_ent:
                        continue

                    if passive:
                        relations.add((o_ent, verb, s_ent))
                    else:
                        relations.add((s_ent, verb, o_ent))

    return list(relations)

# -------------------------
# Graph
# -------------------------
G = nx.MultiDiGraph()

# -------------------------
# Main loop
# -------------------------
for episode_id, episode in corpus.items():

    text = f"{episode['title']}. {episode['description']}"
    doc = nlp(text)

    gliner_entities = model.predict_entities(
        text,
        GLINER_LABELS,
        threshold=0.55
    )

    entity_dict, token_lookup = extract_entities(doc, gliner_entities)
    relations = extract_relations(doc, token_lookup)

    # -------------------------
    # nodes
    # -------------------------
    for ent, labels in entity_dict.items():

        if not G.has_node(ent):
            G.add_node(ent, labels="", episode_count=0, _seen=set())

        node = G.nodes[ent]

        existing_labels = set(node["labels"].split(",")) if node["labels"] else set()
        existing_labels.update(labels)
        node["labels"] = ",".join(sorted(existing_labels))

        if episode_id not in node["_seen"]:
            node["_seen"].add(episode_id)
            node["episode_count"] += 1

    # -------------------------
    # edges
    # -------------------------
    for s, r, o in relations:

        if s == o:
            continue

        if G.has_edge(s, o):

            for k in G[s][o]:
                edge = G[s][o][k]

                if edge["relation"] == r:
                    edge["weight"] += 1

                    episodes = set(edge["episodes"].split(",")) if edge["episodes"] else set()
                    episodes.add(str(episode_id))
                    edge["episodes"] = ",".join(sorted(episodes))
                    break
            else:
                G.add_edge(s, o, relation=r, weight=1, episodes=str(episode_id))

        else:
            G.add_edge(s, o, relation=r, weight=1, episodes=str(episode_id))

# -------------------------
# Clean GraphML-safe export
# -------------------------
for n, d in G.nodes(data=True):
    if isinstance(d.get("_seen"), set):
        d["_seen"] = ",".join(sorted(d["_seen"]))

# -------------------------
# Stats
# -------------------------
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# -------------------------
# Visualization
# -------------------------
net = Network(height="900px", width="100%", bgcolor="#111111", font_color="white", directed=True)
net.barnes_hut()

for node, data in G.nodes(data=True):

    freq = data.get("episode_count", 1)
    labels = data.get("labels", "")

    color = "#97c2fc"

    if "PERSON" in labels:
        color = "#ff6666"
    elif "LOCATION" in labels:
        color = "#66ff99"
    elif "ORGANIZATION" in labels:
        color = "#ffcc66"
    elif "ALIEN" in labels:
        color = "#cc66ff"

    net.add_node(node, label=node, title=f"{node}<br>{labels}", value=freq, color=color)

for s, o, d in G.edges(data=True):
    net.add_edge(
        s,
        o,
        label=d.get("relation", ""),
        title=f"{d.get('relation','')} ({d.get('weight',1)})",
        value=d.get("weight", 1)
    )

nx.write_graphml(G, "graph_depo/doctor_who_kg.graphml")
net.save_graph("graph_depo/doctor_who_kg.html")

print("Saved graph + visualization")
