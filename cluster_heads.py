from crossbert import AttentionAnalyzer
from crossbert import load_code_search_net_sample
from crossbert import ModelConfig
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
from tqdm import tqdm
import umap
import numpy as np


def get_inputs(language="python", num_examples=100):
    data = load_code_search_net_sample(language)
    analyzer = AttentionAnalyzer(
        ModelConfig(
            name="microsoft/codebert-base",
            path="microsoft/codebert-base",
            is_finetuned=False,
            language="multilingual",
        )
    )

    encoded_inputs_list, sep_indices_list = analyzer.preprocess(data)

    encoded_inputs_list["input_ids"] = encoded_inputs_list["input_ids"][:num_examples]
    encoded_inputs_list["attention_mask"] = encoded_inputs_list["attention_mask"][
        :num_examples
    ]
    sep_indices_list = sep_indices_list[:num_examples]

    batch_size = 128
    qc = []
    for i in tqdm(
        range(0, len(sep_indices_list), batch_size),
        total=(len(sep_indices_list) // batch_size),
    ):
        inp_ids = encoded_inputs_list["input_ids"][i : i + batch_size]
        att_masks = encoded_inputs_list["attention_mask"][i : i + batch_size]
        sep_list = sep_indices_list[i : i + batch_size]

        batch_cluster_input = analyzer.get_cluster_inputs(
            {"input_ids": inp_ids, "attention_mask": att_masks}, sep_list
        )

        qc.extend(batch_cluster_input["query_to_code"])

    qc = np.array(qc)

    return {"query_to_code": qc}


def visualize_clusters(labels, n_clusters, metric, values):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(values)

    norm = mcolors.Normalize(vmin=0, vmax=n_clusters - 1)
    cmap = plt.colormaps.get_cmap("tab10").resampled(n_clusters)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap=cmap, norm=norm, s=10)
    ax.set_title(f"PCA of Attention Head Activations for {metric}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # Add colorbar with axis explicitly provided
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Cluster ID")

    # Save and close
    file_path = f"cluster_heads_{metric}.png"
    plt.savefig(file_path)
    plt.close(fig)


def elbow_plot(values, max_clusters=15):
    pca_reduced = PCA(n_components=50).fit_transform(values)

    reducer = umap.UMAP(n_neighbors=45, min_dist=0.1, n_components=20)
    umap_embedding = reducer.fit_transform(pca_reduced)

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

    plt.savefig("elbow.png")
    plt.close()


def cluster_heads_hdbscan(attention_metrics):
    import hdbscan

    res = []
    for k, v in attention_metrics.items():
        print(f"Clustering {k} with HDBSCAN")

        pca = PCA(n_components=20)
        v = pca.fit_transform(v)

        # scaler = StandardScaler()
        # v = scaler.fit_transform(v)

        hdb = hdbscan.HDBSCAN()
        hdb.fit(v)
        labels = hdb.labels_
        res.append({"metric": k, "labels": labels})

    return res


def cluster_heads_kmeans(attention_metrics, n_clusters=12):
    res = []
    for k, v in attention_metrics.items():
        print(f"Clustering {k} with KMeans")

        pca_reduced = PCA(n_components=50).fit_transform(v)

        reducer = umap.UMAP(n_neighbors=45, min_dist=0.1, n_components=20)
        umap_embedding = reducer.fit_transform(pca_reduced)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(umap_embedding)
        labels = kmeans.labels_
        res.append(
            {"metric": k, "labels": labels, "centroids": kmeans.cluster_centers_}
        )

    return res


if __name__ == "__main__":
    import pickle as bigle

    # data = get_inputs("python", num_examples=5000)
    # with open("something.pkl", "wb") as f:
    #     pickle.dump(data, f)

    data = bigle.load(open("something.pkl", "rb"))
    elbow_plot(data["query_to_code"])

    # n_clusters = 3
    # res = cluster_heads_kmeans(data, n_clusters)
    # for r in res:
    #     visualize_clusters(r["labels"], n_clusters, r["metric"], data[r["metric"]])
