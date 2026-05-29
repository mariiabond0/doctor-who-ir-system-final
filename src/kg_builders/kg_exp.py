import json
import spacy
from config import CORPUS_PATH
from gliner import GLiNER

# -------------------------
# Load corpus
# -------------------------
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)

# -------------------------
# NLP
# -------------------------
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer")

model = GLiNER.from_pretrained("urchade/gliner_base")

GLINER_LABELS = ["PERSON", "LOCATION", "TIME", "ORGANIZATION", "DATE", "ALIEN"]

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

# -------------------------
# Helpers
# -------------------------


def normalize_label(label: str) -> str:
    return STANDARD_LABELS.get(label, label)


def canonical(text: str) -> str:
    return " ".join(text.lower().split())


# -------------------------
# entity builder
# -------------------------
def build_entities(doc, gliner_entities):
    entity_dict = {}

    def add(text, label):
        k = canonical(text)
        entity_dict.setdefault(k, set()).add(normalize_label(label))

    # spaCy entities
    for ent in doc.ents:
        add(ent.text, ent.label_)

    # GLiNER entities
    for ent in gliner_entities:
        add(ent["text"], ent["label"])

    return entity_dict


# -------------------------
# Relation extraction
# -------------------------
def extract_relations(doc, entity_dict):

    relations = []

    for sent in doc.sents:
        ents = list(sent.ents)
        for ent1 in ents:
            for ent2 in ents:
                if ent1 == ent2:
                    continue
                token1 = ent1.root
                token2 = ent2.root
                path1 = {token1}
                path2 = {token2}
                while token1.head != token1:
                    token1 = token1.head
                    path1.add(token1)
                while token2.head != token2:
                    token2 = token2.head
                    path2.add(token2)
                common = path1.intersection(path2)
                verbs = [t for t in common if t.pos_ == "VERB"]
                if verbs:
                    verb = verbs[0].lemma_
                    relations.append((ent1.text, verb, ent2.text))
    return relations


# -------------------------
# Filter corpus
# -------------------------
short_corpus = {k: v for k, v in corpus.items() if (v["season"] == 3 and v["number"] <= 3)}

# -------------------------
# Main loop
# -------------------------
for episode_id, episode in short_corpus.items():

    raw_text = f"{episode['title']}. {episode['description']}"
    doc = nlp(raw_text)

    gliner_entities = model.predict_entities(raw_text, GLINER_LABELS)

    # -------------------------
    # Entity dictionary (canonical node model)
    # -------------------------
    entity_dict = build_entities(doc, gliner_entities)

    print(f"\n=== {episode_id} | S{episode['season']}E{episode['number']} ===")

    print("\nExtracted entities:")
    for ent, labels in sorted(entity_dict.items()):
        print(f"{ent} -> {sorted(labels)}")

    relations = extract_relations(doc, entity_dict)

    print("\nRelations (dependency-based heuristic):")
    for subj, verb, obj in relations:
        print(f"{canonical(subj)} --{verb}--> {canonical(obj)}")
