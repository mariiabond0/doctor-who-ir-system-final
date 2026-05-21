import json
import spacy
import networkx as nx

from pathlib import Path
from gliner import GLiNER
from pyvis.network import Network
from config import CORPUS_PATH

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)

# -------------------------
# NLP
# -------------------------
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer")

model = GLiNER.from_pretrained("urchade/gliner_base")

GLINER_LABELS = [
    "PERSON",
    "LOCATION",
    "TIME",
    "ORGANIZATION",
    "DATE",
    "ALIEN"
]

# -------------------------
# Normalization
# -------------------------
STANDARD_LABELS = {
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "LOCATION": "LOCATION",
    "ORGANIZATION": "ORGANIZATION",
    "DATE": "DATE",
    "TIME": "TIME",
    "ALIEN": "ALIEN",
}

ALIASES = {
    "the doctor": "doctor",
    "doctor who": "doctor",
    "amy": "amy pond",
    "rory": "rory williams",
    "clara": "clara oswald",
    "rose": "rose tyler",
    "jack": "captain jack harkness",
    "martha": "martha jones",
    "donna": "donna noble",
    "the master": "master",
    'the daleks': 'daleks',
    'the cybermen': 'cybermen',
    'the weeping angels': 'weeping angels',
    'angel': 'weeping angels',
    "angels": "weeping angels",
    "the silence": "silence",
    "river": "river song",
    'professor river song': 'river song',
    "amelia pond": "amy pond",
    "amelia": "amy pond",
    'pond': "amy pond", 
    "the doctor": "doctor",
    'vincent': "vincent van gogh",
    "clara": "clara oswald",
    "oswald": "clara oswald",
    "oswin": "clara oswald",
    "jackie": "jackie tyler",
    "mickey": "mickey smith",
    "smith": "mickey smith",
    "pete": "pete tyler",
    "rose's father": "pete tyler",
    "rose's mother": "jackie tyler",
    "bill": "bill potts",
    "the brigadier": "brigadier lethbridge-stewart",
    "lethbridge-stewart": "brigadier lethbridge-stewart",
    "the sicarax": "sicarax",
    "the zygon": "zygon",
    "the judoon": "judoon",
    "the ood": "ood",
    "the silurian": "silurian",
    "the autons": "autons",
    "auton": "autons",
}

STOP_ENTITIES = {
    "one",
    "two",
    "they",
    "them",
    "him",
    "her",
    "who",
    "which",
    "it",
    "something",
    "anything"
}

BAD_RELATIONS = {
    "be",
    "have",
    "do",
    "say",
    "go",
    "get"
}

# -------------------------
# Helpers
# -------------------------
def normalize_label(label):
    return STANDARD_LABELS.get(label, label)

def canonical(text):
    text = " ".join(text.lower().split())
    return ALIASES.get(text, text)

# -------------------------
# Entity extraction
# -------------------------
def extract_entities(doc, gliner_entities):

    entity_dict = {}

    def add_entity(text, label):

        key = canonical(text)

        if key in STOP_ENTITIES:
            return

        if len(key) <= 1:
            return

        label = normalize_label(label)

        if key not in entity_dict:
            entity_dict[key] = set()

        entity_dict[key].add(label)

    # spaCy entities
    for ent in doc.ents:
        add_entity(ent.text, ent.label_)

    # GLiNER entities
    for ent in gliner_entities:
        add_entity(ent["text"], ent["label"])

    return entity_dict

# -------------------------
# Relation extraction
# -------------------------
def extract_relations(doc, entity_dict):

    relations = []

    valid_entities = set(entity_dict.keys())

    for sent in doc.sents:

        for token in sent:

            if token.pos_ != "VERB":
                continue

            verb = token.lemma_.lower()

            if verb in BAD_RELATIONS:
                continue

            subjects = [
                w for w in token.lefts
                if w.dep_ in ("nsubj", "nsubjpass")
            ]

            objects = [
                w for w in token.rights
                if w.dep_ in ("dobj", "pobj", "attr")
            ]

            for subj in subjects:
                for obj in objects:

                    subj_text = canonical(subj.text)
                    obj_text = canonical(obj.text)

                    if subj_text not in valid_entities:
                        continue

                    if obj_text not in valid_entities:
                        continue

                    if subj_text == obj_text:
                        continue

                    relations.append(
                        (subj_text, verb, obj_text)
                    )

    return relations

# -------------------------
# Build graph
# -------------------------
G = nx.MultiDiGraph()

# -------------------------
# Small subset for testing
# -------------------------
#subset = {
#    k: v for k, v in corpus.items()
#    if v["season"] == 5 and v["number"] <= 5
#}

# -------------------------
# Main loop
# -------------------------
for episode_id, episode in corpus.items():

    text = f"{episode['title']}. {episode['description']}"

    doc = nlp(text)

    gliner_entities = model.predict_entities(
        text,
        GLINER_LABELS
    )

    entity_dict = extract_entities(doc, gliner_entities)

    relations = extract_relations(doc, entity_dict)

    # -------------------------
    # Add nodes
    # -------------------------
    for entity, labels in entity_dict.items():

        G.add_node(
            entity,
            labels=",".join(labels)
        )

    # -------------------------
    # Add edges
    # -------------------------
    for subj, rel, obj in relations:

        G.add_edge(
            subj,
            obj,
            relation=rel,
            episode=episode_id
        )

# -------------------------
# Stats
# -------------------------
print("\n===== GRAPH STATS =====")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# -------------------------
# Visualization
# -------------------------
net = Network(
    height="900px",
    width="100%",
    bgcolor="#111111",
    font_color="white",
    directed=True
)

# physics layout
net.barnes_hut()

# -------------------------
# Add nodes
# -------------------------
for node, data in G.nodes(data=True):

    labels = ", ".join(data.get("labels", []))

    # simple coloring
    color = "#97c2fc"

    if "PERSON" in labels:
        color = "#ff6666"

    elif "LOCATION" in labels:
        color = "#66ff99"

    elif "ORGANIZATION" in labels:
        color = "#ffcc66"

    elif "ALIEN" in labels:
        color = "#cc66ff"

    net.add_node(
        node,
        label=node,
        title=f"{node}<br>{labels}",
        color=color
    )

# -------------------------
# Add edges
# -------------------------
for source, target, data in G.edges(data=True):

    relation = data.get("relation", "")

    net.add_edge(
        source,
        target,
        title=relation,
        label=relation
    )

# -------------------------
# Save outputs
# -------------------------
nx.write_graphml(G, "doctor_who_kg.graphml")

net.save_graph("doctor_who_kg.html")

print("\nSaved:")
print("- doctor_who_kg.graphml")
print("- doctor_who_kg.html")