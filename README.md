# Vector Database Implementation using FAISS and Milvus Lite

This repository demonstrates how to build and query vector databases using **FAISS** and **Milvus Lite**.

The project includes examples of:

* Building FAISS indexes
* Saving and loading indexes
* Generating random vectors
* Creating Milvus collections
* Generating text embeddings
* Performing similarity search

Two Jupyter notebooks are provided to compare **local vector indexing (FAISS)** and **persistent vector storage (Milvus Lite)**.

---

# 📌 Overview

Vector databases store high-dimensional vectors and enable fast similarity search.
This project demonstrates core vector database workflows including:

* Vector generation
* Index creation
* Data insertion
* Similarity search
* Index persistence

The notebooks simulate common vector database tasks used in:

* Semantic search
* Document retrieval
* Recommendation systems
* AI assistants
* Retrieval-Augmented Generation (RAG)

---

# 📂 Repository Structure

```bash
Vector-DB/
│
├── faiss.ipynb          # FAISS index creation and vector search
├── milvus.ipynb         # Milvus Lite collections and search
└── README.md
```

---

# ⚙️ Technologies Used

* Python
* NumPy
* FAISS
* Milvus Lite
* pymilvus
* Sentence Embedding Model (`paraphrase-albert-small-v2`)

---

# 📘 Notebook Details

---

# 🔹 FAISS Implementation (`faiss.ipynb`)

This notebook demonstrates building a FAISS vector index using **IndexFlatIP** (Inner Product similarity).

---

## Configuration

```python
DIM = 512
TOTAL_VECS = 100
TOPK = 3
QUERY_SIZE = 1
INDEX_FILE = "flat.index"
```

Meaning:

* Vector dimension: **512**
* Number of stored vectors: **100**
* Top search results: **3**

---

## Index Type

The FAISS index used:

```python
faiss.IndexFlatIP(DIM)
```

This uses:

* **Inner Product similarity**
* Often used for **cosine similarity**

---

## Workflow

### Step 1 — Build Index

Creates FAISS index:

```python
build_flat_index()
```

---

### Step 2 — Add Random Vectors

Random vectors are generated:

```python
xb = np.random.random((TOTAL_VECS, DIM))
```

Vectors added to index:

```python
index.add(xb)
```

---

### Step 3 — Save Index

Index is saved to disk:

```python
faiss.write_index(index, "flat.index")
```

This allows reuse without rebuilding.

---

### Step 4 — Load Existing Index

If the index exists:

```python
faiss.read_index("flat.index")
```

---

### Step 5 — Search

Query vectors are generated:

```python
xq = np.random.random((QUERY_SIZE, DIM))
```

Search operation:

```python
D, I = index.search(xq, TOPK)
```

Output example:

```text
First query result:
[(39, 136.16), (96, 135.36), (63, 135.18)]
```

Where:

* `I` → vector IDs
* `D` → similarity scores

---

# 🔹 Milvus Lite Implementation (`milvus.ipynb`)

This notebook demonstrates storing vectors persistently using **Milvus Lite**.

Milvus Lite stores vectors locally in:

```text
milvus_demo.db
```

---

# Collection 1 — Text Embedding Search

Collection Name:

```text
text_collection
```

---

## Vector Dimension

```python
dimension = 768
```

Embeddings generated using:

```python
model.DefaultEmbeddingFunction()
```

This downloads:

```text
paraphrase-albert-small-v2
```

---

## Text Dataset

The notebook includes 10 text documents such as:

* Gutenberg printing press
* Isaac Newton
* Charles Darwin
* Ada Lovelace
* Python programming language

Each record includes:

```python
{
  "id": int,
  "vector": embedding,
  "text": string,
  "subject": "history"
}
```

---

## Insert Data

```python
client.insert(
    collection_name="text_collection",
    data=data
)
```

Example result:

```text
{'insert_count': 10}
```

---

## Text Search Example

Query:

```python
"Who is Alan Turing?"
```

Search:

```python
client.search(
    collection_name="text_collection",
    limit=2
)
```

Example result:

```text
Ada Lovelace is often regarded as the world’s first computer programmer.
Python, created by Guido van Rossum in 1991...
```

---

# Collection 2 — Random Vector Search

Collection Name:

```text
random_num_collection
```

---

## Configuration

```python
DIM = 512
TOTAL_VECS = 100
TOPK = 3
```

---

## Random Vector Creation

```python
xb = np.random.random((TOTAL_VECS, DIM))
```

Inserted as:

```python
{"id": i, "vector": xb[i]}
```

---

## Vector Search

Query:

```python
query_vectors = np.random.random((1, DIM))
```

Search:

```python
client.search(
    collection_name="random_num_collection",
    limit=TOPK
)
```

Returns:

* Nearest vectors
* Similarity distances

---

# 🔍 Key Concepts Demonstrated

This project covers:

* Vector indexing
* Similarity search
* Persistent storage
* Embedding generation
* Index saving/loading
* Cosine similarity search

---

# 📊 FAISS vs Milvus Lite

| Feature           | FAISS             | Milvus Lite              |
| ----------------- | ----------------- | ------------------------ |
| Storage           | Memory / File     | Local Database           |
| Persistence       | Manual Save       | Automatic                |
| Index Type        | IndexFlatIP       | COSINE                   |
| Text Support      | No                | Yes                      |
| Embedding Support | External          | Built-in                 |
| Use Case          | Fast Local Search | Persistent Search System |


| Metric           | FAISS     | Milvus Lite |
| ---------------- | --------- | ----------- |
| Index Creation   | Very Fast | Fast        |
| Vector Insertion | Fast      | Fast        |
| Search Latency   | Very Low  | Low         |
| Persistence      | Manual    | Automatic   |
| Scalability      | Medium    | Medium      |

From this benchmark:

* **FAISS** is faster for small in-memory workloads
* **Milvus Lite** provides persistence and structured storage
* Both systems support efficient similarity search
* Milvus becomes more useful as dataset size increases

This demonstrates the trade-off between:

```text
Speed vs Persistence
```


---

# 📚 Learning Objectives

This project helps understand:

* Vector similarity search
* FAISS index management
* Milvus collection design
* Text embedding workflows
* Persistent vector databases

---

# 👩‍💻 Author

**Nabilah Hannania**

GitHub:
https://github.com/nabilahannania

