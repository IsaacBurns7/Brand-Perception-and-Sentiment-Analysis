from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Use better embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize BERTopic
topic_model = BERTopic(
    embedding_model=embedding_model,
    language="english",
    calculate_probabilities=True,
    verbose=True
)

# Fit model
topics, probs = topic_model.fit_transform(df["full_text"].tolist())

# Save topics
df["bertopic_topic"] = topics

# View topic summary
print(topic_model.get_topic_info())
