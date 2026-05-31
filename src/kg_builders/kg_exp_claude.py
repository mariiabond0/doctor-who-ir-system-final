import json
import spacy
import networkx as nx

from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from gliner import GLiNER
from pyvis.network import Network
from config import CORPUS_PATH

# -------------------------
# Load data & models
# -------------------------
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)

nlp = spacy.load("en_core_web_trf")
nlp.add_pipe("sentencizer")

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
]


# -------------------------
# Config / lookup tables
# -------------------------

LABEL_MAP = {
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "LOCATION": "LOCATION",
    "PLANET": "LOCATION",
    "DATE": "DATE",
    "TIME": "TIME",
    "ALIEN SPECIES": "ALIEN",
    "ALIEN": "ALIEN",
    "SPACESHIP": "ORGANIZATION",
}

ALIASES = {
    "the doctor": "doctor",
    "doctor who": "doctor",
    "amy": "amy pond",
    "amelia": "amy pond",
    "amelia pond": "amy pond",
    "pond": "amy pond",
    "rory": "rory williams",
    "clara": "clara oswald",
    "oswald": "clara oswald",
    "oswin": "clara oswald",
    "rose": "rose tyler",
    "rose's mother": "jackie tyler",
    "rose's father": "pete tyler",
    "jack": "captain jack harkness",
    "martha": "martha jones",
    "donna": "donna noble",
    "the master": "master",
    "the daleks": "daleks",
    "the cybermen": "cybermen",
    "the weeping angels": "weeping angels",
    "angel": "weeping angels",
    "angels": "weeping angels",
    "the silence": "silence",
    "river": "river song",
    "professor river song": "river song",
    "vincent": "vincent van gogh",
    "jackie": "jackie tyler",
    "mickey": "mickey smith",
    "smith": "mickey smith",
    "pete": "pete tyler",
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
    "anything",
}

SKIP_VERBS = {"be", "have", "do", "say", "go", "get"}

PREP_LABELS = {
    "in": "located_in",
    "on": "located_on",
    "at": "located_at",
    "from": "from",
    "to": "travels_to",
    "with": "with",
    "against": "opposes",
    "for": "acts_for",
    "of": "part_of",
    "into": "enters",
    "through": "travels_through",
    "across": "crosses",
    "during": "occurs_during",
}

NODE_COLORS = {
    "PERSON": "#ff6b6b",
    "LOCATION": "#51cf66",
    "ORGANIZATION": "#ffd43b",
    "ALIEN": "#cc5de8",
    "DATE": "#74c0fc",
    "TIME": "#74c0fc",
}

# -------------------------
# Visualization settings
# -------------------------

# Only show nodes with at least this many connections.
# Raise to 5 to see only the most central characters.
MIN_DEGREE = 3

# Set to True to also show weak co-occurrence edges.
SHOW_COOCCURRENCE = False
OUTPUT_PREFIX = "doctor_who_kg_claude"


# -------------------------
# Helper functions
# -------------------------


def to_canonical(text):
    text = " ".join(text.lower().split())
    return ALIASES.get(text, text)


def get_entity_at_token(token, span_lookup, last_person, last_org):
    for (start, end), entity in span_lookup.items():
        if start <= token.i < end:
            return entity
    t = token.text.lower()
    if t in {"he", "she", "him", "her", "his", "hers"}:
        return last_person
    if t in {"they", "them", "it"}:
        return last_person or last_org
    return None


def get_conjunct_entities(token, span_lookup, last_person, last_org):
    entities = []
    queue = [token]
    visited = set()
    while queue:
        t = queue.pop()
        if t.i in visited:
            continue
        visited.add(t.i)
        e = get_entity_at_token(t, span_lookup, last_person, last_org)
        if e:
            entities.append(e)
        for child in t.children:
            if child.dep_ == "conj":
                queue.append(child)
    return entities


# -------------------------
# Main extraction functions
# -------------------------


def extract_entities(doc, gliner_entities):
    entity_dict = {}
    span_lookup = {}

    def register(text, raw_label, start=None, end=None):
        name = to_canonical(text)
        if name in STOP_ENTITIES or len(name) <= 1:
            return
        label = LABEL_MAP.get(raw_label.upper(), raw_label.upper())
        entity_dict.setdefault(name, set()).add(label)
        if start is not None:
            span_lookup[(start, end)] = name

    for ent in doc.ents:
        register(ent.text, ent.label_, ent.start, ent.end)

    for ent in gliner_entities:
        if ent["score"] < 0.55:
            continue
        span = doc.char_span(ent["start"], ent["end"], alignment_mode="expand")
        if span:
            register(span.text, ent["label"], span.start, span.end)

    return entity_dict, span_lookup


def extract_relations(doc, span_lookup):
    relations = []
    last_person = [None]
    last_org = [None]

    def update_referents(name, labels):
        if "PERSON" in labels:
            last_person[0] = name
        elif "ORGANIZATION" in labels:
            last_org[0] = name

    for name, labels in extract_relations._entity_dict.items():
        update_referents(name, labels)

    for sent in doc.sents:
        for token in sent:
            if token.pos_ != "VERB" or token.lemma_.lower() in SKIP_VERBS:
                continue

            verb = token.lemma_.lower()
            subjects = []
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subjects += get_conjunct_entities(
                        child, span_lookup, last_person[0], last_org[0]
                    )

            if not subjects:
                continue

            for child in token.children:
                if child.dep_ in ("dobj", "attr", "dative", "oprd"):
                    for obj in get_conjunct_entities(
                        child, span_lookup, last_person[0], last_org[0]
                    ):
                        for subj in subjects:
                            if subj != obj:
                                relations.append((subj, verb, obj))

            for child in token.children:
                if child.dep_ == "prep":
                    rel_label = PREP_LABELS.get(
                        child.lemma_.lower(), f"{verb}_{child.lemma_.lower()}"
                    )
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            for obj in get_conjunct_entities(
                                pobj, span_lookup, last_person[0], last_org[0]
                            ):
                                for subj in subjects:
                                    if subj != obj:
                                        relations.append((subj, rel_label, obj))

    syntactic_pairs = {(s, o) for s, _, o in relations} | {(o, s) for s, _, o in relations}
    for sent in doc.sents:
        seen = []
        for token in sent:
            e = get_entity_at_token(token, span_lookup, last_person[0], last_org[0])
            if e and e not in seen:
                seen.append(e)
        for i, a in enumerate(seen):
            for b in seen[i + 1 :]:
                if (a, b) not in syntactic_pairs:
                    relations.append((a, "co_occurs_with", b))
                    relations.append((b, "co_occurs_with", a))

    return relations


extract_relations._entity_dict = {}


def merge_near_duplicates(G):
    nodes = list(G.nodes())
    merge_map = {}

    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            shorter, longer = sorted([a, b], key=len)
            is_prefix = longer.startswith(shorter) and len(shorter) >= 4
            is_similar = SequenceMatcher(None, a, b).ratio() >= 0.85
            if (is_prefix or is_similar) and shorter not in merge_map:
                merge_map[shorter] = longer

    if not merge_map:
        return G

    def resolve(name):
        seen = set()
        while name in merge_map and name not in seen:
            seen.add(name)
            name = merge_map[name]
        return name

    H = nx.MultiDiGraph()
    for node, data in G.nodes(data=True):
        canon = resolve(node)
        if canon not in H:
            H.add_node(canon, labels=set())
        H.nodes[canon]["labels"].update(data.get("labels", set()))
    for src, tgt, data in G.edges(data=True):
        H.add_edge(resolve(src), resolve(tgt), **data)

    print(f"  Merged {len(merge_map)} near-duplicate nodes: {merge_map}")
    return H


# -------------------------
# Build the graph
# -------------------------
G = nx.MultiDiGraph()
edge_freq = defaultdict(int)

for episode_id, episode in corpus.items():
    text = f"{episode['title']}. {episode['description']}"
    doc = nlp(text)

    gliner_ents = model.predict_entities(text, GLINER_LABELS, threshold=0.55)
    entity_dict, span_lookup = extract_entities(doc, gliner_ents)

    extract_relations._entity_dict = entity_dict
    relations = extract_relations(doc, span_lookup)

    for name, labels in entity_dict.items():
        if name not in G:
            G.add_node(name, labels=set())
        G.nodes[name]["labels"].update(labels)

    seen = set()
    for subj, rel, obj in relations:
        if subj == obj:
            continue
        key = (subj, rel, obj)
        if key not in seen:
            seen.add(key)
            G.add_edge(subj, obj, relation=rel, episode=episode_id)
            edge_freq[key] += 1

print("\nMerging near-duplicate nodes...")
G = merge_near_duplicates(G)

# Recompute edge frequencies after node merge so visualization width/labels stay accurate.
edge_freq = defaultdict(int)
for src, tgt, data in G.edges(data=True):
    edge_freq[(src, data.get("relation"), tgt)] += 1

print("\n===== GRAPH STATS =====")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
top = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 by degree:")
for name, deg in top:
    print(f"  {name}: {deg}")


# -------------------------
# Visualization
# -------------------------

# Stringify sets for GraphML export
for node, data in G.nodes(data=True):
    if isinstance(data.get("labels"), set):
        data["labels"] = ", ".join(sorted(data["labels"]))

graphml_path = Path(f"{OUTPUT_PREFIX}.graphml")
nx.write_graphml(G, graphml_path.as_posix())

# Filter to well-connected nodes only
degree_map = dict(G.degree())
keep_nodes = {n for n, d in degree_map.items() if d >= MIN_DEGREE}
VIZ_G = G.subgraph(keep_nodes)

print(
    f"\nVisualization: {VIZ_G.number_of_nodes()} nodes, "
    f"{VIZ_G.number_of_edges()} edges "
    f"(MIN_DEGREE={MIN_DEGREE}, co-occurrence={'on' if SHOW_COOCCURRENCE else 'off'})"
)

# Build Pyvis network
# NOTE: show_buttons is intentionally removed — it injects an iframe that
# prevents the canvas from rendering in many browser/Pyvis combinations.
net = Network(height="100vh", width="100%", bgcolor="#0d1117", font_color="#e6edf3", directed=True)

net.set_options("""{
  "physics": {
    "solver": "barnesHut",
    "barnesHut": {
      "gravitationalConstant": -4000,
      "centralGravity": 0.2,
      "springLength": 200,
      "springConstant": 0.02,
      "damping": 0.09
    },
    "stabilization": {
      "enabled": true,
      "iterations": 200,
      "updateInterval": 10,
      "fit": true
    }
  }
}""")

max_degree = max(degree_map.values(), default=1)
max_freq = max(edge_freq.values(), default=1)

for node, data in VIZ_G.nodes(data=True):
    labels_str = data.get("labels", "")
    deg = degree_map.get(node, 1)
    size = int(10 + 40 * (deg / max_degree))
    color = next(
        (NODE_COLORS[lbl.strip()] for lbl in labels_str.split(",") if lbl.strip() in NODE_COLORS),
        "#adb5bd",
    )
    net.add_node(
        node,
        label=node,
        title=f"<b>{node}</b><br>{labels_str}<br>Connections: {deg}",
        color=color,
        size=size,
    )

for src, tgt, data in VIZ_G.edges(data=True):
    rel = data.get("relation", "")
    if rel == "co_occurs_with" and not SHOW_COOCCURRENCE:
        continue
    # Skip edges whose endpoints were filtered out
    if src not in keep_nodes or tgt not in keep_nodes:
        continue
    freq = edge_freq.get((src, rel, tgt), 1)
    is_weak = rel == "co_occurs_with"
    net.add_edge(
        src,
        tgt,
        title=f"{rel} (×{freq})",
        label="",
        width=1.0 + 3.0 * (freq / max_freq),
        color="#4a5568" if is_weak else "#718096",
        dashes=is_weak,
    )

# ---- Inject custom HTML/JS into the saved file ----

LEGEND_HTML = """
<div style="position:fixed;top:16px;left:16px;z-index:999;background:#161b22;
            border:1px solid #30363d;border-radius:8px;padding:12px 16px;
            font-family:monospace;font-size:13px;color:#e6edf3;
            box-shadow:0 4px 16px rgba(0,0,0,.5)">
  <b>ENTITY TYPES</b><br><br>
  <span style="color:#ff6b6b">&#9632;</span> Person<br>
  <span style="color:#51cf66">&#9632;</span> Location<br>
  <span style="color:#ffd43b">&#9632;</span> Organization / Ship<br>
  <span style="color:#cc5de8">&#9632;</span> Alien<br>
  <span style="color:#74c0fc">&#9632;</span> Date / Time<br>
  <span style="color:#adb5bd">&#9632;</span> Other<br>
  <br><small>Node size = connections<br>Line width = frequency<br>Hover edge = relation</small>
</div>"""

SEARCH_HTML = """
<div style="position:fixed;top:16px;right:16px;z-index:999">
  <input id="kg-search" placeholder="Search entity…"
         style="padding:7px 12px;border-radius:6px;border:1px solid #30363d;
                background:#161b22;color:#e6edf3;font-size:13px;
                width:200px;outline:none">
</div>"""

# Single polled block — waits for the vis.js `network` global before
# attaching any handlers, and hides the loading bar on stabilization.
EXTRA_JS = """
<script>
(function () {
  var poll = setInterval(function () {
    if (typeof network === "undefined") return;
    clearInterval(poll);

    // Hide the loading bar as soon as stabilization finishes
    network.on("stabilizationProgress", function (params) {
      var bar = document.getElementById("loadingBar");
      if (bar) {
        var pct = Math.round(params.iterations / params.total * 100);
        var fill = bar.querySelector(".bar") || bar.querySelector("[style*='width']");
        if (fill) fill.style.width = pct + "%";
      }
    });
    network.once("stabilizationIterationsDone", function () {
      network.setOptions({ physics: { enabled: false } });
      var bar = document.getElementById("loadingBar");
      if (bar) bar.style.display = "none";
    });

    // Search / highlight
    var input = document.getElementById("kg-search");
    if (!input) return;
    input.addEventListener("input", function () {
      var q = this.value.toLowerCase().trim();
      if (!q) { network.selectNodes([]); return; }
      var ids = Object.values(network.body.data.nodes._data)
        .filter(function (n) { return n.label && n.label.toLowerCase().includes(q); })
        .map(function (n) { return n.id; });
      network.selectNodes(ids);
      if (ids.length === 1) network.focus(ids[0], { scale: 1.4, animation: true });
    });
  }, 150);
})();
</script>"""

html_path = Path(f"{OUTPUT_PREFIX}.html")

# write_html is preferred; fall back to save_graph for older Pyvis versions
try:
    net.write_html(str(html_path))
except AttributeError:
    net.save_graph(str(html_path))

html = html_path.read_text(encoding="utf-8")
html = html.replace("<body>", f"<body>\n{LEGEND_HTML}\n{SEARCH_HTML}", 1)
html = html.replace("</body>", f"{EXTRA_JS}\n</body>", 1)
html_path.write_text(html, encoding="utf-8")

print(f"\nSaved: {graphml_path.name}, {html_path.name}")
