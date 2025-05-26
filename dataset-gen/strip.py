import orjson

input_file = "topicwise_pairs.json"
output_file = "topicwise_pairs_unlabelled.json"

with open(input_file, "rb") as infile, open(output_file, "wb") as outfile:
    for line in infile:
        if line.strip():
            data = orjson.loads(line)
            filtered = {"query": data["query"], "code": data["code"]}
            outfile.write(orjson.dumps(filtered) + b"\n")

print("Done!")
