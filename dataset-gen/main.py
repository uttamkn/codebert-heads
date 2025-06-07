import asyncio
import os
from typing import List

import orjson
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

load_dotenv()


class Output(BaseModel):
    domain: str
    query: str
    code: str

    def to_labelled_dict(self):
        return self.model_dump()

    def to_unlabelled_dict(self):
        # Exclude the domain (topic) for unlabelled version
        return {"query": self.query, "code": self.code}


LANGUAGES = ["Go", "Python", "JavaScript"]

CODE_DOMAINS = [
    "data structures and algorithms",
    "control flow and function logic",
    "object-oriented programming",
    "web server code",
    "unit testing",
    "data analysis",
    "file handling",
    "machine learning code",
    "scientific and mathematical code",
    "image processing code",
    "database operations",
    "network programming",
]

client = genai.Client(api_key=os.getenv("API_KEY"))


def gen_prompt(domain: str, language: str, batch_size: int = 16) -> str:
    return f"""
Generate {batch_size} natural language query and {language} code snippet from the domain "{domain}".

## Output Requirements:
- `domain`: "{domain}" (Should be the exact same as the domain provided to you in the prompt)
- `query`: A clear, concise single sentence describing what the code does.
- `code`: Clean {language} code without comments or docstrings.

## Quality Guidelines:
- Code should be functional and demonstrate the domain concept
- Do not add import statements. If any libraries used, use it just like that without the import statement(s).
- Natural language should be assertive and specific
- Vary complexity from beginner to intermediate level
- Total output must stay under 512 tokens
- Focus on practical, real-world examples
"""


async def generate_code_query_pairs(
    domain: str,
    language: str,
    batch_size: int = 16,
) -> List[Output]:
    prompt = gen_prompt(domain, language, batch_size)

    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "domain": {"type": "string"},
                "query": {"type": "string"},
                "code": {"type": "string"},
            },
            "required": ["domain", "query", "code"],
        },
    }

    response = await client.aio.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[prompt],
        config={
            "response_mime_type": "application/json",
            "response_schema": schema,
        },
    )
    await asyncio.sleep(4)
    data = orjson.loads(response.text)
    return [Output(**item) for item in data]


async def write_to_file(file_path: str, data: List[dict]):
    """Write data to file asynchronously"""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(orjson.dumps(item).decode("utf-8") + "\n")


async def run_all_languages_pipeline():
    output_dir = "code_query_pairs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_language_pairs = {lang: [] for lang in LANGUAGES}
    num_batches = 6

    for language in LANGUAGES:
        for domain in CODE_DOMAINS:
            print(f"Processing {domain} for {language}...")
            for i in range(num_batches):
                batch = await generate_code_query_pairs(domain, language, 16)
                all_language_pairs[language].extend(batch)
                print(f"  {language} - {domain}: Batch {i + 1}/{num_batches} completed")

    for language, pairs in all_language_pairs.items():
        labelled_file_name = f"{language}_labelled.jsonl"
        unlabelled_file_name = f"{language}_unlabelled.jsonl"
        labelled_file_path = os.path.join(output_dir, labelled_file_name)
        unlabelled_file_path = os.path.join(output_dir, unlabelled_file_name)

        labelled_data = [pair.to_labelled_dict() for pair in pairs]
        unlabelled_data = [pair.to_unlabelled_dict() for pair in pairs]

        await write_to_file(labelled_file_path, labelled_data)
        await write_to_file(unlabelled_file_path, unlabelled_data)

        print(
            f"✓ {language}: {len(pairs)} labelled pairs written to {labelled_file_path}"
        )
        print(
            f"✓ {language}: {len(pairs)} unlabelled pairs written to {unlabelled_file_path}"
        )

    print(
        f"\n🎉 Generated labelled and unlabelled pairs for all {len(LANGUAGES)} languages"
    )
    print(f"Files saved in '{output_dir}' directory")


if __name__ == "__main__":
    asyncio.run(run_all_languages_pipeline())
