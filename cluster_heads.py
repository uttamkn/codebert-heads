from crossbert import AttentionAnalyzer
from crossbert import load_synthetic_dataset, load_code_search_net_sample
from crossbert import ModelConfig
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
from tqdm import tqdm
import umap
import os
import numpy as np


def get_code_pairs(language="python"):
    domains = load_synthetic_dataset(language)
    csn_data = load_code_search_net_sample(language)
    data = []
    for _, v in domains.items():
        data.extend(v)

    # add csn data to synthetic
    data.extend(csn_data)
    return data


def get_q2c_attn_score_for_all_pairs(language="python", num_examples=100):
    data = get_code_pairs(language)
    analyzer = AttentionAnalyzer(
        ModelConfig(
            name="microsoft/codebert-base",
            path="model/python",
            is_finetuned=True,
            language="python",
        )
    )

    # Create tokens and extract sep indices
    encoded_inputs_list, sep_indices_list = analyzer.preprocess(data)

    # Limit examples
    encoded_inputs_list["input_ids"] = encoded_inputs_list["input_ids"][:num_examples]
    encoded_inputs_list["attention_mask"] = encoded_inputs_list["attention_mask"][
        :num_examples
    ]
    sep_indices_list = sep_indices_list[:num_examples]

    # Get attention scores batch-wise
    batch_size = 128
    qc = []  # this will have q2c attn scores for all attn heads (flattened to a vector of size 144 (12x12))
    for i in tqdm(
        range(0, len(sep_indices_list), batch_size),
        total=(len(sep_indices_list) // batch_size),
    ):
        inp_ids = encoded_inputs_list["input_ids"][i : i + batch_size]
        att_masks = encoded_inputs_list["attention_mask"][i : i + batch_size]
        sep_list = sep_indices_list[i : i + batch_size]

        batch_cluster_input = analyzer.get_q2c_attn_scores(
            {"input_ids": inp_ids, "attention_mask": att_masks}, sep_list
        )

        qc.extend(batch_cluster_input)
    return np.array(qc)


def elbow_plot(values, max_clusters=15, new_dim=20):
    reducer = umap.UMAP(n_neighbors=45, min_dist=0.1, n_components=new_dim)
    umap_embedding = reducer.fit_transform(values)

    wcss = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(umap_embedding)
        wcss.append(kmeans.inertia_)

    x = np.arange(1, max_clusters + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(x, wcss, "bo-")
    plt.xlabel("Number of clusters")
    plt.ylabel("Within-cluster Sum of Squares (WCSS)")
    plt.title("Elbow Method for Optimal Number of Clusters")
    plt.grid(True)

    os.makedirs("cluster_heads", exist_ok=True)
    plt.savefig("cluster_heads/elbow.png")
    plt.close()


def cluster_heads_kmeans(values, n_clusters, new_dim_size):
    # reduce dimensions
    reducer = umap.UMAP(n_neighbors=45, min_dist=0.1, n_components=new_dim_size)
    umap_embedding = reducer.fit_transform(values)

    # apply kmeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(umap_embedding)

    return kmeans.labels_


def visualize_clusters(values, labels, n_clusters):
    reducer = umap.UMAP(n_neighbors=45, min_dist=0.1, n_components=2)
    X_2d = reducer.fit_transform(values)

    norm = mcolors.Normalize(vmin=0, vmax=n_clusters - 1)
    cmap = plt.colormaps.get_cmap("tab10").resampled(n_clusters)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap=cmap, norm=norm, s=10)
    ax.set_title("UMAP of Attention Head Activations")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Add colorbar with axis explicitly provided
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Cluster ID")

    # Save and close
    os.makedirs("cluster_heads", exist_ok=True)
    file_path = "cluster_heads/kmeans.png"
    plt.savefig(file_path)
    plt.close(fig)


def get_corresponding_pairs(labels, n_clusters, language="python"):
    cluster_to_pairs = [[] for _ in range(n_clusters)]
    pairs = get_code_pairs(language)
    for i, label in enumerate(labels):
        cluster_to_pairs[label].append(pairs[i])

    return cluster_to_pairs


if __name__ == "__main__":
    import pickle as bigle

    # data = get_q2c_attn_score_for_all_pairs("python", num_examples=5000)
    # with open("something.pkl", "wb") as f:
    #     bigle.dump(data, f)

    data = bigle.load(open("something.pkl", "rb"))
    elbow_plot(data, new_dim=7)

    n_clusters = 3
    labels = cluster_heads_kmeans(data, n_clusters, new_dim_size=144)
    visualize_clusters(data, labels, n_clusters)
