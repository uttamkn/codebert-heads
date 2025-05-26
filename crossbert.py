import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers.models.roberta import RobertaTokenizerFast, RobertaModel
from tokenizers import processors
from typing import List, Tuple


class AttentionAnalyzer:
    def __init__(self, model_name="microsoft/codebert-base"):
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        self.tokenizer._tokenizer.post_processor = processors.BertProcessing(
            sep=("</s>", self.tokenizer._tokenizer.token_to_id("</s>")),
            cls=("<s>", self.tokenizer._tokenizer.token_to_id("<s>")),
        )
        self.model = RobertaModel.from_pretrained(
            model_name, output_attentions=True, attn_implementation="eager"
        )
        self.model.eval()

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

    def get_entropy(self, attention_matrix):
        flat = attention_matrix.flatten()
        prob = flat / (flat.sum() + 1e-10)
        return -torch.sum(prob * torch.log(prob + 1e-10)).item()

    # more spread less sparsity
    def get_sparsity(self, attention_matrix, threshold=0.01):
        flat = attention_matrix.flatten()
        return ((flat < threshold).sum().item()) / len(flat)

    def cross_model_attn(self, attention_matrix, sep_indices: List[int]):
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
            "query_to_code": np.zeros((num_layers, num_heads)),
            "code_to_query": np.zeros((num_layers, num_heads)),
            "code_to_code": np.zeros((num_layers, num_heads)),
            "query_to_query": np.zeros((num_layers, num_heads)),
        }

        for i in range(num_layers):
            for k in range(num_heads):
                ent, sp, q2c, c2q, c2c, q2q = [], [], [], [], [], []
                for j in range(batch_size):
                    matrix = attention_data[i][j][k]
                    ent.append(self.get_entropy(matrix))
                    sp.append(self.get_sparsity(matrix))
                    cross = self.cross_model_attn(matrix, sep_indices[j])
                    q2c.append(cross["query_to_code"])
                    c2q.append(cross["code_to_query"])
                    c2c.append(cross["code_to_code"])
                    q2q.append(cross["query_to_query"])

                stats["entropy"][i, k] = np.mean(ent)
                stats["sparsity"][i, k] = np.mean(sp)
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
        print("[*] Preprocessing...")
        encoded_inputs_list, sep_indices_list = self.preprocess(pairs[:limit])
        stats = self.analyze_attention(encoded_inputs_list, sep_indices_list)
        self.plot_heatmaps(stats, output_dir=output_dir)


def load_code_search_net_sample():
    """Helper function to load a sample from CodeSearchNet dataset."""
    try:
        from datasets import load_dataset

        dataset = load_dataset("code_search_net", "python", split="validation")
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


def create_sample_data():
    """Create sample data for testing."""
    return [
        ("Calculate the sum of two numbers", "def add(a, b): return a + b"),
        ("Find maximum value in list", "def find_max(lst): return max(lst)"),
        ("Sort a list in ascending order", "def sort_list(lst): return sorted(lst)"),
        (
            "Check if number is prime",
            "def is_prime(n): return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))",
        ),
        ("Reverse a string", "def reverse_string(s): return s[::-1]"),
    ]


def load_json_data(file_path, limit=20):
    """Load JSON data from a file."""
    import json
    from collections import defaultdict

    with open(file_path, "r") as f:
        data = json.load(f)

        res = defaultdict(list)
        for key, value in data.items():
            res[int(key)] = [(v[0]["query"], v[0]["code"]) for v in value]

        return {i: res[i] for i in range(limit)}


def load_synthetic_dataset(file_path):
    import json
    from collections import defaultdict

    with open(file_path, "r") as f:
        res = defaultdict(list)
        for line in f:
            data = json.loads(line)
            res[data["domain"].lower().replace("/", "_")].append(
                (data["query"], data["code"])
            )

    return res


def cluster_analysis(file_name, limit):
    # clustered_data_pairs = load_json_data(file_name, limit)
    clustered_data_pairs = load_synthetic_dataset(file_name)
    print(f"[*] Loaded {len(clustered_data_pairs)} clusters from JSON data.")
    for key, pairs in clustered_data_pairs.items():
        print(f"[*] Analyzing cluster: {key} with {limit} pairs...")
        analyzer = AttentionAnalyzer()
        analyzer.run(
            pairs,
            limit=limit,
            output_dir=f"output/cluster_{key}",
        )


def main():
    pairs = create_sample_data()
    analyzer = AttentionAnalyzer()
    analyzer.run(pairs, output_dir="output")


if __name__ == "__main__":
    cluster_analysis("data/topicwise_pairs.json", 30)
