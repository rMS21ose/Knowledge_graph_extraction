# Install dependencies before running:
# pip install spacy networkx matplotlib
# python -m spacy download en_core_web_sm

import spacy
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

def extract_entities_and_relationships(text):
    doc = nlp(text)

    # Directed graph for relationships
    graph = nx.DiGraph()

    # Dictionary to hold named entities
    entities = {}
    # Dictionary to hold subject-verb-object triples
    relationships = defaultdict(list)

    # Extract named entities
    for ent in doc.ents:
        entities[ent.text] = ent.label_

    # Extract subject-verb-object relationships
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            subject = token.text
            verb = token.head.text

            # Search for direct object of the verb
            object_ = None
            for child in token.head.children:
                if child.dep_ == "dobj":
                    object_ = child.text
                    break

            if object_:
                graph.add_edge(subject, object_, label=verb)
                relationships[subject].append((verb, object_))

    return graph, entities, relationships

# Sample input text
text = """
Elon Musk is the CEO of SpaceX. He founded Tesla in 2003. Tesla produces electric cars.
"""

# Extract data
graph, entities, relationships = extract_entities_and_relationships(text)

# Display Entities
print("Entities:")
for entity, label in entities.items():
    print(f"{entity}: {label}")

# Display Relationships
print("\nRelationships:")
for subject, rels in relationships.items():
    for verb, obj in rels:
        print(f"{subject} --[{verb}]--> {obj}")

# Visualize the Knowledge Graph
pos = nx.spring_layout(graph)

plt.figure(figsize=(8, 6))
nx.draw(graph, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray")
edge_labels = nx.get_edge_attributes(graph, 'label')
nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color="red")
plt.title("Knowledge Graph", fontsize=16)
plt.axis("off")
plt.show()
