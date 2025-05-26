# codebert-heads

A research project exploring which attention heads in CodeBERT are most critical for mapping natural language to code.

---

## **Statistical Analysis of Attention Head Activations in Code Retrieval**

### Objective

Investigate which attention heads and layers in BERT-based models (such as **CodeBERT** or **GraphCodeBERT**) are most influential in mapping natural language queries to code snippets.

### Research Tasks

#### 1. Data Collection

- Curate a dataset of paired natural language queries and corresponding code snippets.
- Use publicly available datasets from repositories like GitHub, CodeSearchNet, or curated academic benchmarks.

#### 2. Model Probing

- Utilize pre-trained code retrieval models (e.g., CodeBERT, GraphCodeBERT).
- Pass query–code pairs through the model and extract **attention weights** from each layer and head.
- Focus on both query-to-code and code-to-query attention when applicable.

#### 3. Statistical Analysis

- Compute and analyze the **distribution of attention activations** across layers and heads.
- Identify **statistically significant attention patterns**, such as:
  - Heads that consistently show high activation for specific structures (e.g., loops, functions).
  - Layers where most meaningful semantic alignment occurs.
- Use statistical tests (e.g., ANOVA, t-tests) to assess significance.

#### 4. Visualization & Interpretation

- Apply **clustering algorithms** or **dimensionality reduction** techniques like **PCA** or **t-SNE** to visualize attention patterns.
- Correlate observed patterns with:
  - Types of code constructs.
  - Natural language instruction patterns.
- Interpret the role of specific heads in semantic matching between query and code.

---

## Dataset Used

- **CodeSearchNet**: A large-scale dataset of code snippets and their corresponding natural language descriptions.
- **Synthetic Data**: [Kaggle](https://www.kaggle.com/datasets/mohitnair512/python-code-query-pair-dataset)
