import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords

data_dir = Path("data")
df = pd.read_csv(data_dir / "rating.csv")

pd.set_option("display.max_columns", None)
# pd.set_option("display.max_colwidth", None)
pd.set_option("display.expand_frame_repr", False)

# # === basic EDA Commands ===
# print("df.head() =")
# print(df.head(), "\n")

# print("df.tail() =")
# print(df.tail(), "\n")

# print("df.sample(5) =")
# print(df.sample(5), "\n")

# print("df.columns =")
# print(df.columns, "\n")

# print("df.index =")
# print(df.index, "\n")

# print("df.shape =")
# print(df.shape, "\n")

# print("len(df) =")
# print(len(df), "\n")

# print("df.dtypes =")
# print(df.dtypes, "\n")

# print("df.info() =")
# df.info()  # info prints directly, no need to wrap
# print("\n")

# print("df.isnull() =")
# print(df.isnull(), "\n")

# print("df.isnull().sum() =")
# print(df.isnull().sum(), "\n")

# print("df.describe() =")
# print(df.describe(), "\n")

# sample_col = df.columns[0] if len(df.columns) > 0 else None
# if sample_col:
#     print(f"df['{sample_col}'].value_counts() = ")
#     print(df[sample_col].value_counts(), "\n")

#     print(f"df['{sample_col}'].unique() =")
#     print(df[sample_col].unique(), "\n")

#     print(f"df['{sample_col}'].nunique() =")
#     print(df[sample_col].nunique(), "\n")

# # Example filtering
# if sample_col:
#     print(f"df[df['{sample_col}'] == df['{sample_col}'].iloc[0]] =") #because not all article id's are unique 
#     print(df[df[sample_col] == df[sample_col].iloc[0]], "\n")

# print("df.sort_values(by=df.columns[0]) =")
# print(df.sort_values(by=df.columns[0]), "\n")

# print("df.groupby sample (first column) size =")
# print(df.groupby(df.columns[0]).size(), "\n")

# # For text-like columns
# text_cols = df.select_dtypes(include="object").columns
# for col in text_cols:
#     print(f"df['{col}'].str.len() =")
#     print(df[col].str.len(), "\n")
#     print(f"df['{col}'].str.contains('a') =")
#     print(df[col].str.contains("a"), "\n")

print("=" * 20)
print("TEXT STUFF")
print("=" * 20)

text_col = "article" #all the below can be done for other text cols too 

# 1. Length of each text and descriptive stats
print("df[text_col].str.len().describe() =")
print(df[text_col].str.len().describe(), "\n")

# 2. Add text length column
df["text_length"] = df[text_col].str.len()
print("df['text_length'] =")
print(df["text_length"].head(), "\n")

# 3. Histogram of text length
plt.figure(figsize=(8, 5))

plt.hist(df["text_length"], bins=50)
plt.xlabel("Text Length")
plt.ylabel("Frequency")
plt.title("Distribution of Text Length")

plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Word count per text
df["word_count"] = df[text_col].str.split().str.len()
print("df['word_count'].describe() =")
print(df["word_count"].describe(), "\n")

# 5. Proportion of texts containing '@'
print("df[text_col].str.contains('@').mean() =")
print(df[text_col].str.contains("@").mean(), "\n")

# 6. Proportion of texts containing 'http'
print("df[text_col].str.contains('http').mean() =")
print(df[text_col].str.contains("http").mean(), "\n")

# 7. Missing values
print("df[text_col].isnull().sum() =")
print(df[text_col].isnull().sum(), "\n")

# 8. Empty strings (after stripping whitespace)
print("(df[text_col].str.strip() == '').sum() =")
print((df[text_col].str.strip() == "").sum(), "\n")

# 9. Top 20 words
all_words = df[text_col].str.lower().str.split().explode()
print("all_words.value_counts().head(20) =")
print(all_words.value_counts().head(20), "\n")

vocab_size = all_words.nunique()
print("Vocabulary size:", vocab_size)

# 10. Unique text proportion
print("df[text_col].nunique() / len(df) =")
print(df[text_col].nunique() / len(df), "\n")

# 11. Duplicated texts count
print("df[text_col].duplicated().sum() =")
print(df[text_col].duplicated().sum(), "\n")

# 12. ASCII-only text proportion
print("df[text_col].str.isascii().mean() =")
print(df[text_col].str.isascii().mean(), "\n")

# 13. 1st and 99th percentile of text length
print("df[text_length].quantile([0.01, 0.99]) =")
print(df["text_length"].quantile([0.01, 0.99]), "\n")

# Shannon entropy per document (rough measure of information content)
def shannon_entropy(text):
    counts = np.array(list(Counter(text.split()).values()))
    probs = counts / counts.sum()
    return -(probs * np.log2(probs)).sum() if len(probs) > 0 else 0

df["entropy"] = df[text_col].apply(shannon_entropy)
print("Document entropy stats:\n", df["entropy"].describe())

# --- 3. TF-IDF sparsity check ---
# --- 3. TF-IDF sparsity diagnostic (expanded) ---

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df[text_col])

# Matrix shape
n_docs, n_features = X.shape

# Global sparsity
sparsity = 1.0 - X.nnz / (n_docs * n_features)

# Per-document statistics (how many features are active per document)
nnz_per_doc = X.getnnz(axis=1)

print("TF-IDF matrix shape:", X.shape)
print("Global sparsity:", sparsity)

print("\nNonzero feature stats per document:")
print("Mean active features per document:", nnz_per_doc.mean())
print("Std active features per document:", nnz_per_doc.std())
print("Min active features per document:", nnz_per_doc.min())
print("25th percentile:", np.percentile(nnz_per_doc, 25))
print("Median:", np.median(nnz_per_doc))
print("75th percentile:", np.percentile(nnz_per_doc, 75))
print("Max:", nnz_per_doc.max())

# Feature-level sparsity (how many documents each term appears in)
nnz_per_feature = X.getnnz(axis=0)

print("\nFeature activation stats:")
print("Mean document frequency per feature:", nnz_per_feature.mean())
print("Std document frequency per feature:", nnz_per_feature.std())
print("Min document frequency:", nnz_per_feature.min())
print("25th percentile:", np.percentile(nnz_per_feature, 25))
print("Median:", np.median(nnz_per_feature))
print("75th percentile:", np.percentile(nnz_per_feature, 75))
print("Max document frequency:", nnz_per_feature.max())

# Per-document nonzero feature counts
nnz_per_doc = X.getnnz(axis=1)

# Plot histogram (single plot only, no subplots, no color specification)
plt.figure(figsize=(8, 5))
plt.hist(nnz_per_doc, bins=50)
plt.xlabel("Number of Active TF-IDF Features per Document")
plt.ylabel("Frequency")
plt.title("Distribution of TF-IDF Feature Activation per Document")
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 4. Duplicate detection & similarity ---
duplicates = df[text_col].duplicated().sum()
print("Duplicate documents:", duplicates)
unique_ratio = df[text_col].nunique() / len(df)
print("Unique doc ratio:", unique_ratio)

# Optional: compute Jaccard similarity for first 5 docs as sanity check
def jaccard_similarity(s1, s2):
    a = set(s1.split())
    b = set(s2.split())
    return len(a & b) / len(a | b) if len(a | b) > 0 else 0

print("Sample Jaccard similarities (first 5 docs):")
for i in range(5):
    for j in range(i+1,5):
        print(f"Doc {i} vs Doc {j}:", jaccard_similarity(df[text_col].iloc[i], df[text_col].iloc[j]))

# --- 5. Stopword dominance ---
# You can optionally load NLTK stopwords
stop_words = set(stopwords.words("english"))
df["stopword_ratio"] = df[text_col].apply(lambda t: sum(1 for w in t.lower().split() if w in stop_words) / max(len(t.split()),1))
print("Stopword ratio stats:\n", df["stopword_ratio"].describe())