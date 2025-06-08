from crossbert import AttentionAnalyzer
from crossbert import load_code_search_net_sample
from crossbert import ModelConfig


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
    encoded_inputs_list["attention_mask"] = encoded_inputs_list["attention_mask"][:num_examples]
    sep_indices_list = sep_indices_list[:num_examples]

    num_pairs = min(num_examples, len(sep_indices_list))

    batch_size = 128
    cluster_input = {
        "query_to_code": [],
        "query_to_code_entropy": [],
    }
    for i in range(0, num_pairs, batch_size):
        encoded_inputs_list["input_ids"] = encoded_inputs_list["input_ids"][i:i+batch_size]
        encoded_inputs_list["attention_mask"] = encoded_inputs_list["attention_mask"][i:i+batch_size]
        sep_indices_list = sep_indices_list[i:i+batch_size]

        batch_cluster_input = analyzer.get_cluster_inputs(encoded_inputs_list, sep_indices_list)

        cluster_input["query_to_code"].append(batch_cluster_input["query_to_code"])
        cluster_input["query_to_code_entropy"].append(batch_cluster_input["query_to_code_entropy"])

    cluster_input["query_to_code"] = np.array(cluster_input["query_to_code"])
    cluster_input["query_to_code_entropy"] = np.array(cluster_input["query_to_code_entropy"])

    return cluster_input


if __name__ == "__main__":
    print(get_inputs(num_examples=100)["query_to_code"].shape)
