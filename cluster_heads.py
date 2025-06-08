from crossbert import AttentionAnalyzer
from crossbert import load_code_search_net_sample
from crossbert import ModelConfig
from tqdm import tqdm
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
    qc, qce = [], []
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
        qce.extend(batch_cluster_input["query_to_code_entropy"])

    qc = np.array(qc)
    qce = np.array(qce)

    return {"query_to_code": qc, "query_to_code_entropy": qce}


if __name__ == "__main__":
    import pickle

    data = get_inputs("python", num_examples=5000)

    print(data["query_to_code"].shape)
    with open("something.pkl", "wb") as f:
        pickle.dump(data, f)
