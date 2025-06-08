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

    encoded_inputs_list, sep_indices_list = analyzer.preprocess(data[:200])

    cluster_input = analyzer.get_cluster_inputs(encoded_inputs_list, sep_indices_list)

    return cluster_input


if __name__ == "__main__":
    print(get_inputs()["query_to_code"].shape)
