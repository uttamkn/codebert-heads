# Statistical Analysis of Attention Head Activations in Code Retrieval

**Objective:**
Investigate which attention heads and layers in BERT-based models (such as CodeBERT or GraphCodeBERT) are most influential in mapping natural language queries to code snippets.

**Research Tasks:**

* **Data Collection:** Curate a dataset of paired natural language queries and corresponding code snippets from public repositories (e.g., GitHub).  
* **Model Probing:** Use pre-trained code retrieval models and extract attention weights from each layer and head when processing the query–code pairs.  
* **Statistical Analysis:**
  * Compute the distribution of activations across layers and attention heads.  
  * Identify statistically significant patterns (e.g., which heads consistently show high activation when certain types of code structures or natural language instructions are processed).
* **Visualization & Interpretation:** Use clustering or dimensionality reduction (e.g., PCA) to visualize activation patterns, and interpret how these might correlate with semantic features in the code.
