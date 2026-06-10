import random
import json
import argparse
from pathlib import Path

# Technical vocabulary pool to create realistic-looking documentation tokens
VOCABULARY = [
    "FastAPI", "Uvicorn", "Pydantic", "CORS", "routing", "asyncio", "middleware", 
    "dependency", "injection", "database", "SQLAlchemy", "OAuth2", "security", 
    "response", "request", "validation", "schemas", "background", "tasks", 
    "retrieval", "augmented", "generation", "vector", "embeddings", "Qdrant", 
    "late-interaction", "ColBERT", "sparse", "dense", "hybrid", "reranking"
]

PAGE_PATHS = [
    "tutorial/cors", "tutorial/security", "tutorial/background-tasks", 
    "advanced/custom-middleware", "advanced/dependency-injection", 
    "deployment/docker", "deployment/kubernetes", "database/sql-databases"
]

HEADINGS = [
    "Introduction to CORS Configuration", "Setting up OAuth2 password bearer",
    "Defining asynchronous background tasks", "Custom middleware execution order",
    "Advanced dependency overrides for testing", "Containerizing with multi-stage Dockerfiles",
    "Running migrations with Alembic", "Vector search configuration thresholds"
]

def generate_sentence() -> str:
    words_count = random.randint(8, 18)
    words = [random.choice(VOCABULARY) for _ in range(words_count)]
    sentence = " ".join(words)
    return sentence.capitalize() + "."

def generate_paragraph(sentence_count: int = 5) -> str:
    return " ".join([generate_sentence() for _ in range(sentence_count)])

def generate_chunk(index: int) -> dict:
    page_path = random.choice(PAGE_PATHS)
    heading = f"{random.choice(HEADINGS)} (Part {random.randint(1, 5)})"
    clean_heading = heading.lower().replace(" ", "-").replace("(", "").replace(")", "")
    
    return {
        "chunk_text": generate_paragraph(random.randint(3, 7)),
        "heading_text": heading,
        "page_id": f"{page_path}_{index}",
        "section_url": f"https://fastapi.tiangolo.com/{page_path}/#{clean_heading}"
    }

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic documentation data for load testing.")
    parser.add_argument("--count", "-c", type=int, default=1000, help="Number of chunks to generate")
    parser.add_argument("--output", "-o", type=str, default="tests/synthetic_data.json", help="Output JSON path")
    args = parser.parse_args()

    print(f"Generating {args.count} synthetic documentation chunks...")
    chunks = [generate_chunk(i) for i in range(args.count)]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Successfully wrote synthetic data to {output_path}")

if __name__ == "__main__":
    main()
