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


CODE_DOMAINS = [
    "Data Structures & Algorithms",
    "Control Flow & Function Logic",
    "Object-Oriented Programming",
    "Web Server Code",
    "Unit Testing",
    "Data Analysis",
    "File Handling",
    "Machine Learning Code",
    "Scientific and Mathematical Code",
    "Image Processing Code",
    "Database Operations",
    "Network Programming",
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


async def run_pipeline():
    all_pairs = []
    num_batches = 6

    for domain in CODE_DOMAINS:
        print(f"Processing {domain}...")
        domain_pairs = []

        for i in range(num_batches):
            batch = await generate_code_query_pairs(domain, "Python", 16)
            domain_pairs.extend(batch)
            print(f"  Batch {i + 1}/{num_batches} completed")

        all_pairs.extend(domain_pairs)
        print(f"✓ {domain}: {len(domain_pairs)} pairs")

    with open("topicwise_pairs.json", "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(orjson.dumps(pair.model_dump()).decode("utf-8") + "\n")

    print(f"\n🎉 Generated {len(all_pairs)} total pairs in topicwise_pairs.json")


if __name__ == "__main__":
    asyncio.run(run_pipeline())
