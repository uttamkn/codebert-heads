import umap
import os
import numpy as np
import argparse
import pickle
from pathlib import Path
from typing import Any
from google import genai
from crossbert import (
    AttentionAnalyzer,
    load_synthetic_dataset,
    load_code_search_net_sample,
    ModelConfig,
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


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


def get_avg_attn_scores(labels, data, n_clusters):
    cluster_to_attn = [[] for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        cluster_to_attn[label].append(data[i])

    avg_attn_scores = []
    for cluster in cluster_to_attn:
        avg_attn_scores.append(np.mean(cluster, axis=0))

    print(f"avg_attn_scores shape: {np.array(avg_attn_scores).shape}")
    return avg_attn_scores


def visualize_avg_attn_scores(avg_attn_scores, output_dir="cluster_heads"):
    if not avg_attn_scores:
        print("No average attention scores to visualize.")
        return
    for i, scores in enumerate(avg_attn_scores):
        scores = scores.reshape(12, 12)
        plt.figure(figsize=(10, 6))
        sns.heatmap(scores, annot=True, fmt=".2f", cmap="viridis", cbar=True)
        plt.title(f"Average Attention Scores for Cluster {i + 1}")
        plt.xlabel("Attention Head Index")
        plt.ylabel("Attention Layer Index")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"avg_attn_scores_cluster_{i + 1}.png"))
        plt.close()


def ask_gemini_to_infer_cluster_description(cluster_pairs, n_clusters):
    system_prompt = """
    You are a senior software engineer with 15 years of experience in software development. 
    You are given a set of unlabelled clusters of natural language queries and their corresponding code snippets.
    Your task is to infer the cluster description for each cluster.
    """

    user_prompt = "Here are the clustered code pairs:\n\n"

    LIMIT = 25
    for i in range(n_clusters):
        user_prompt += f"cluster_{i + 1}:\n\n"
        user_prompt += "\n\n".join(
            [f"Query: {c[0]}\nCode: {c[1]}" for c in cluster_pairs[i][:LIMIT]]
        )
        user_prompt += "\n\n"

    client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
    model = "gemini-2.0-flash-exp"
    try:
        response = client.models.generate_content(
            model=model,
            contents=[user_prompt],
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
            ),
        )
        if not response.text:
            raise ValueError("Gemini LLM response text is empty")

        print(response.text)
        return response.text
    except Exception as e:
        print(e)
        return None


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Cluster attention heads based on Q2C attention patterns"
    )

    # Main arguments
    parser.add_argument(
        "--generate-data", action="store_true", help="Generate attention scores data"
    )
    parser.add_argument(
        "--load-data",
        type=str,
        default=None,
        help="Load attention scores from pickle file",
    )
    parser.add_argument(
        "--elbow-plot",
        action="store_true",
        help="Generate elbow plot for determining optimal clusters",
    )
    parser.add_argument(
        "--cluster", action="store_true", help="Run clustering and visualization"
    )
    parser.add_argument(
        "--ask-gemini",
        action="store_true",
        help="Ask Gemini to infer cluster description",
    )

    # Optional parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="cluster_heads",
        help="Output directory for saving results",
    )
    parser.add_argument(
        "--num-examples", type=int, default=5000, help="Number of examples to process"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="python",
        help="Programming language for code pairs",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=3, help="Number of clusters for K-means"
    )
    parser.add_argument(
        "--dim-size", type=int, default=144, help="Dimensionality for UMAP reduction"
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=15,
        help="Maximum number of clusters for elbow plot",
    )

    return parser.parse_args()


def ensure_output_dir(output_dir: str) -> None:
    """Ensure the output directory exists."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def save_data(data: Any, filepath: str) -> None:
    """Save data to a pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def save_text_data(data: str, filepath: str) -> None:
    """Save text data to a file."""
    with open(filepath, "w") as f:
        f.write(data)


def load_data(filepath: str) -> Any:
    """Load data from a pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def main():
    args = parse_arguments()

    # Ensure output directory exists
    ensure_output_dir(args.output_dir)

    if args.generate_data:
        print(f"Generating attention scores for {args.num_examples} examples...")
        data = get_q2c_attn_score_for_all_pairs(
            args.language, num_examples=args.num_examples
        )
        output_file = os.path.join(args.output_dir, "attention_scores.pkl")
        save_data(data, output_file)
        print(f"Data saved to {output_file}")
    elif args.load_data:
        print(f"Loading data from {args.load_data}")
        data = load_data(args.load_data)
    else:
        print("Please specify either --generate-data or --load-data")
        return

    if args.elbow_plot:
        print("Generating elbow plot...")
        elbow_plot(data, max_clusters=args.max_clusters, new_dim=min(50, data.shape[1]))
        print(f"Elbow plot saved to {os.path.join(args.output_dir, 'elbow.png')}")

    if args.cluster:
        print(f"Running K-means clustering with {args.n_clusters} clusters...")
        labels = cluster_heads_kmeans(data, args.n_clusters, new_dim_size=args.dim_size)
        visualize_clusters(data, labels, args.n_clusters)
        avg_attn_scores = get_avg_attn_scores(labels, data, args.n_clusters)
        visualize_avg_attn_scores(avg_attn_scores, output_dir=args.output_dir)
        print(
            f"Clustering visualization saved to {os.path.join(args.output_dir, 'kmeans.png')}"
        )

        if args.ask_gemini:
            print("Asking Gemini to infer cluster description...")
            conclusion = ask_gemini_to_infer_cluster_description(
                get_corresponding_pairs(labels, args.n_clusters, args.language),
                args.n_clusters,
            )
            save_text_data(
                conclusion or "Gemini did not return a valid response.",
                os.path.join(args.output_dir, "gemini_cluster_descriptions.txt"),
            )
            print(
                f"Cluster descriptions saved to {os.path.join(args.output_dir, 'gemini_cluster_descriptions.txt')}"
            )


if __name__ == "__main__":
    main()
