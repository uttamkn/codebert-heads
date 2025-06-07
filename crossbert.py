import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
from transformers import RobertaTokenizerFast, RobertaModel
from tokenizers import processors


@dataclass
class ModelConfig:
    name: str
    path: str
    is_finetuned: bool = False
    language: str = "multilingual"

    def load_model(self):
        tokenizer = RobertaTokenizerFast.from_pretrained(
            self.path if self.is_finetuned else self.name
        )
        tokenizer._tokenizer.post_processor = processors.BertProcessing(
            sep=("</s>", tokenizer._tokenizer.token_to_id("</s>")),
            cls=("<s>", tokenizer._tokenizer.token_to_id("<s>")),
        )
        model = RobertaModel.from_pretrained(
            self.path if self.is_finetuned else self.name,
            output_attentions=True,
            attn_implementation="eager",
        )
        model.eval()
        return model, tokenizer


class AttentionAnalyzer:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.model, self.tokenizer = model_config.load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def preprocess(
        self, pairs: List[Tuple[str, str]], limit: int = 512, max_len: int = 256
    ):
        pairs = [
            pair for pair in pairs if len(pair[0]) <= limit and len(pair[1]) <= limit
        ]
        inputs = self.tokenizer(
            pairs,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        sep_indices = [
            [i for i, token_id in enumerate(inputs["input_ids"][j]) if token_id == 2]
            for j in range(len(inputs["input_ids"]))
        ]
        return inputs, sep_indices

    def get_entropy(self, attention_matrix: torch.Tensor):
        flat = attention_matrix.flatten()
        prob = flat / (flat.sum() + 1e-10)
        return -torch.sum(prob * torch.log(prob + 1e-10)).item()

    def get_std(self, attention_matrix: torch.Tensor):
        """
        Calculate and return the standard deviation of the attention matrix.

        Args:
            attention_matrix (torch.Tensor): The attention matrix to analyze.

        Returns:
            float: The standard deviation of the attention matrix.
        """
        return attention_matrix.flatten().std().item()

    def get_max(self, attention_matrix: torch.Tensor):
        """
        Calculate and return the maximum value in the attention matrix.

        Args:
            attention_matrix (torch.Tensor): The attention matrix to analyze.

        Returns:
            float: The maximum value in the attention matrix.
        """
        return attention_matrix.max().item()

    # more spread less sparsity
    def get_sparsity(self, attention_matrix: torch.Tensor, threshold=0.01):
        flat = attention_matrix.flatten()
        return ((flat < threshold).sum().item()) / len(flat)

    def cross_model_attn(self, attention_matrix: torch.Tensor, sep_indices: List[int]):
        query_start = 1
        query_end = sep_indices[0]
        code_start = sep_indices[0] + 1
        code_end = sep_indices[1]

        return {
            "query_to_code": attention_matrix[
                query_start:query_end, code_start:code_end
            ]
            .max()
            .item(),
            "code_to_query": attention_matrix[
                code_start:code_end, query_start:query_end
            ]
            .max()
            .item(),
            "code_to_code": attention_matrix[code_start:code_end, code_start:code_end]
            .max()
            .item(),
            "query_to_query": attention_matrix[
                query_start:query_end, query_start:query_end
            ]
            .max()
            .item(),
        }

    def analyze_attention(self, encoded_inputs, sep_indices):
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)

        attention_data = outputs.attentions
        num_layers = len(attention_data)
        num_heads = attention_data[0].shape[1]
        batch_size = attention_data[0].shape[0]

        stats = {
            "entropy": np.zeros((num_layers, num_heads)),
            "sparsity": np.zeros((num_layers, num_heads)),
            "std": np.zeros((num_layers, num_heads)),
            "max": np.zeros((num_layers, num_heads)),
            "query_to_code": np.zeros((num_layers, num_heads)),
            "code_to_query": np.zeros((num_layers, num_heads)),
            "code_to_code": np.zeros((num_layers, num_heads)),
            "query_to_query": np.zeros((num_layers, num_heads)),
        }

        for i in range(num_layers):
            for k in range(num_heads):
                ent, sp, std, max_, q2c, c2q, c2c, q2q = [], [], [], [], [], [], [], []
                for j in range(batch_size):
                    matrix = attention_data[i][j][k]
                    ent.append(self.get_entropy(matrix))
                    sp.append(self.get_sparsity(matrix))
                    std.append(self.get_std(matrix))
                    max_.append(self.get_max(matrix))
                    cross = self.cross_model_attn(matrix, sep_indices[j])
                    q2c.append(cross["query_to_code"])
                    c2q.append(cross["code_to_query"])
                    c2c.append(cross["code_to_code"])
                    q2q.append(cross["query_to_query"])

                stats["entropy"][i, k] = np.mean(ent)
                stats["sparsity"][i, k] = np.mean(sp)
                stats["std"][i, k] = np.mean(std)
                stats["max"][i, k] = np.mean(max_)
                stats["query_to_code"][i, k] = np.mean(q2c)
                stats["code_to_query"][i, k] = np.mean(c2q)
                stats["code_to_code"][i, k] = np.mean(c2c)
                stats["query_to_query"][i, k] = np.mean(q2q)

        return stats

    def plot_heatmaps(self, stats, output_dir="output"):
        os.makedirs(output_dir, exist_ok=True)
        for name, matrix in stats.items():
            plt.figure(figsize=(10, 6))
            sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", cbar=True)
            plt.title(f"{name} Heatmap")
            plt.xlabel("Heads")
            plt.ylabel("Layers")
            plt.tight_layout()
            file_path = os.path.join(
                output_dir, f"{name.lower().replace('_', '_')}_heatmap.png"
            )
            plt.savefig(file_path)
            plt.close()

    def run(self, pairs, output_dir="output", limit=30):
        print(
            f"[*] Analyzing with model: {self.model_config.name} ({self.model_config.language})"
        )
        print(f"[*] Preprocessing {min(len(pairs), limit)} pairs...")

        # Create output directory with model name
        model_output_dir = os.path.join(
            output_dir, self.model_config.name.replace("/", "_")
        )
        if self.model_config.is_finetuned:
            model_output_dir = os.path.join(
                output_dir, f"{self.model_config.language}_finetuned"
            )
        os.makedirs(model_output_dir, exist_ok=True)

        # Process the data
        encoded_inputs_list, sep_indices_list = self.preprocess(pairs[:limit])

        # Move inputs to device
        encoded_inputs_list = {
            k: v.to(self.device) for k, v in encoded_inputs_list.items()
        }

        # Analyze attention
        stats = self.analyze_attention(encoded_inputs_list, sep_indices_list)

        # Save stats to JSON for later comparison
        stats_path = os.path.join(model_output_dir, "attention_stats.json")
        with open(stats_path, "w") as f:
            json.dump({k: v.tolist() for k, v in stats.items()}, f)

        # Generate visualizations
        self.plot_heatmaps(stats, output_dir=model_output_dir)

        return stats, model_output_dir


def load_code_search_net_sample(language="python"):
    """Helper function to load a sample from CodeSearchNet dataset."""
    try:
        from datasets import load_dataset

        dataset = load_dataset("code_search_net", language, split="validation")
        pair_tokens = [
            (d["func_documentation_tokens"], d["func_code_tokens"])  # type: ignore
            for d in dataset
        ]
        return [(" ".join(doc), " ".join(code)) for doc, code in pair_tokens]
    except ImportError:
        print(
            "datasets library not available. Please install with: pip install datasets"
        )
        return []


def create_sample_data(language="python"):
    """Create sample data for different programming languages."""
    if language.lower() == "python":
        return [
            ("Calculate the sum of two numbers", "def add(a, b): return a + b"),
            ("Find maximum value in list", "def find_max(lst): return max(lst)"),
            (
                "Sort a list in ascending order",
                "def sort_list(lst): return sorted(lst)",
            ),
            (
                "Check if number is prime",
                "def is_prime(n): return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))",
            ),
            ("Reverse a string", "def reverse_string(s): return s[::-1]"),
        ]
    elif language.lower() == "javascript":
        return [
            ("Calculate the sum of two numbers", "const add = (a, b) => a + b"),
            ("Find maximum value in array", "const findMax = arr => Math.max(...arr)"),
            (
                "Sort an array in ascending order",
                "const sortArray = arr => [...arr].sort((a, b) => a - b)",
            ),
            (
                "Check if number is prime",
                "const isPrime = n => { for(let i = 2, s = Math.sqrt(n); i <= s; i++) if(n % i === 0) return false; return n > 1; }",
            ),
            (
                "Reverse a string",
                "const reverseString = s => [...s].reverse().join('')",
            ),
        ]
    else:
        return [
            ("Sample function 1", "function sample1() { return 'Hello'; }"),
            ("Sample function 2", "function sample2() { return 42; }"),
        ]


def load_synthetic_dataset(file_path, language="Python"):
    import orjson
    from collections import defaultdict
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    download_path = "./data"
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    api.dataset_download_files(
        "mohitnair512/synthetic-code-query-pairs-golang-js-and-python",
        path=download_path,
        unzip=True,
    )
    file_path = os.path.join(download_path, "{language}_unlabelled.jsonl")

    with open(file_path, "r") as f:
        res = defaultdict(list)
        for line in f:
            data = orjson.loads(line)
            res[data["domain"].lower().replace("/", "_")].append(
                (data["query"], data["code"])
            )

    return res


def cluster_analysis(file_name, limit, models, output_dir="output"):
    """
    Analyze clusters of code pairs using the specified models.

    Args:
        file_name: Path to the file containing clustered data
        limit: Maximum number of pairs to process per cluster
        models: List of ModelConfig objects to use for analysis
        output_dir: Base directory for output
    """
    clustered_data_pairs = load_synthetic_dataset(file_name)
    if not clustered_data_pairs:
        print("[!] No clustered data found or error loading data")
        return []

    print(f"[*] Loaded {len(clustered_data_pairs)} clusters from JSON data.")

    all_cluster_results = {}

    for cluster_key, pairs in clustered_data_pairs.items():
        print(
            f"\n[*] Analyzing cluster: {cluster_key} with {min(len(pairs), limit)} pairs..."
        )
        cluster_output_dir = os.path.join(output_dir, f"cluster_{cluster_key}")

        # Run analysis for this cluster with all models
        cluster_results = {}
        for model in models:
            print(f"  - Using model: {model.name} ({model.language})")
            analyzer = AttentionAnalyzer(model)
            try:
                stats, model_output_dir = analyzer.run(
                    pairs,
                    limit=limit,
                    output_dir=os.path.join(cluster_output_dir, model.language),
                )
                cluster_results[model.name] = {
                    "stats": stats,
                    "output_dir": model_output_dir,
                    "config": model.__dict__,
                }
            except Exception as e:
                print(f"    [!] Error analyzing with {model.name}: {str(e)}")

        all_cluster_results[cluster_key] = cluster_results

        # Generate comparison plots for this cluster
        if len(cluster_results) > 1:  # Only compare if we have multiple models
            plot_comparison(cluster_results, cluster_output_dir)

    return all_cluster_results


def compare_models(models_configs, data, output_dir="output/comparison"):
    """Compare attention patterns across multiple models."""
    os.makedirs(output_dir, exist_ok=True)
    all_stats = {}

    # Run analysis for each model
    for config in models_configs:
        analyzer = AttentionAnalyzer(config)
        stats, model_output_dir = analyzer.run(data, output_dir=output_dir)
        all_stats[config.name] = {
            "stats": stats,
            "output_dir": model_output_dir,
            "config": config.__dict__,
        }

    # Generate comparison visualizations
    plot_comparison(all_stats, output_dir)

    return all_stats


def plot_comparison(all_stats, output_dir):
    """Generate comparison plots across models."""
    metrics = ["entropy", "sparsity", "std", "max", "query_to_code", "code_to_query"]

    for metric in metrics:
        plt.figure(figsize=(12, 6))

        for model_name, data in all_stats.items():
            stats = data["stats"]
            if metric in stats:
                # Take mean across all layers and heads for this metric
                mean_values = np.mean(stats[metric], axis=(0, 1))
                plt.plot(mean_values, label=f"{model_name}")

        plt.title(f"Comparison of {metric} across models")
        plt.xlabel("Layer")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(output_dir, f"comparison_{metric}.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()


def load_finetuned_models(base_dir="model"):
    """Load all available fine-tuned models from the model directory."""
    models = []
    base_path = Path(base_dir)

    # Add base CodeBERT model
    models.append(
        ModelConfig(
            name="microsoft/codebert-base",
            path="microsoft/codebert-base",
            is_finetuned=False,
            language="multilingual",
        )
    )

    # Look for language-specific fine-tuned models
    for lang_dir in base_path.iterdir():
        if lang_dir.is_dir() and (lang_dir / "config.json").exists():
            models.append(
                ModelConfig(
                    name=f"codebert-{lang_dir.name}",
                    path=str(lang_dir),
                    is_finetuned=True,
                    language=lang_dir.name,
                )
            )

    return models


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare CodeBERT attention patterns"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="sample",
        choices=["sample", "code_search_net", "clustered", "synthetic"],
        help="Data source for analysis ('sample': predefined samples, 'code_search_net': CodeSearchNet dataset, 'clustered': topic-clustered data, 'synthetic': topic-wise synthetic data)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="python",
        help="Programming language",
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Maximum number of examples to process"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory for results"
    )
    parser.add_argument(
        "--codebert_only",
        action="store_true",
        help="Run analysis only on the base CodeBERT model",
    )

    args = parser.parse_args()

    # Load data
    if args.data == "sample":
        data = create_sample_data(args.language)
    elif args.data == "code_search_net":
        data = load_code_search_net_sample(args.language)
    elif args.data == "clustered":
        # Load models first
        if args.codebert_only:
            print("[*] Using only the base CodeBERT model as requested")
            models = [
                ModelConfig(
                    name="microsoft/codebert-base",
                    path="microsoft/codebert-base",
                    is_finetuned=False,
                    language="multilingual",
                )
            ]
        else:
            models = load_finetuned_models()
            if not models:
                print("No models found. Using base CodeBERT model.")
                models = [
                    ModelConfig(
                        name="microsoft/codebert-base",
                        path="microsoft/codebert-base",
                        is_finetuned=False,
                        language="multilingual",
                    )
                ]

        # Run cluster analysis with the selected models # Change this line, the file name will turn out to be wrong
        cluster_analysis("topicwise_pairs.json", args.limit, models, args.output_dir)
        return  # Exit after clustered analysis
    elif args.data == "synthetic":
        # here also, Change this line, the file name will turn out to be wrong
        data = load_synthetic_dataset("topicwise_pairs.json")
        if not data:
            print("[!] No synthetic data found or error loading data")
            return

    # Load models
    if args.codebert_only:
        print("[*] Using only the base CodeBERT model as requested")
        models = [
            ModelConfig(
                name="microsoft/codebert-base",
                path="microsoft/codebert-base",
                is_finetuned=False,
                language="multilingual",
            )
        ]
    else:
        models = load_finetuned_models()
        if not models:
            print("No models found. Using base CodeBERT model.")
            models = [
                ModelConfig(
                    name="microsoft/codebert-base",
                    path="microsoft/codebert-base",
                    is_finetuned=False,
                    language="multilingual",
                )
            ]

    print(f"[*] Found {len(models)} models for comparison")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model.name} ({model.language})")

    # Run comparison
    compare_models(models, data[: args.limit], output_dir=args.output_dir)
    print(f"\n[+] Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
