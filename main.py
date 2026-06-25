import requests
from pathlib import Path
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct

client = QdrantClient(url="http://localhost:6333")


def create_collection(collection_name):

    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )


dummy_data = [
    "My name is Charly.",
    "I am a software engineer.",
    "I love programming.",
    "I like to play guitar.",
    "I enjoy hiking.",
    "I am a foodie.",
    "I like to travel.",
    "I am a cat person.",
    "I enjoy reading books.",
    "I like to watch movies.",
    "I am a coffee lover.",
    "I enjoy cooking.",
    "I like to play video games.",
    "I am a dog person.",
    "I enjoy swimming.",
    "I like to go to the beach.",
    "I am a music lover.",
    "I enjoy painting.",
    "I like to dance.",
    "I am a nature enthusiast.",
    "I enjoy photography.",
    "I like to write.",
    "I am a sports fan.",
    "I enjoy gardening.",
    "I like to meditate.",
    "I am a history buff.",
    "I enjoy learning new languages.",
    "I like to volunteer.",
    "I am a technology enthusiast.",
    "I enjoy attending concerts.",
    "I like to go camping.",
    "I am a movie buff.",
    "I enjoy playing board games.",
    "I like to go fishing.",
    "I am a wine enthusiast.",
    "I enjoy going to museums.",
    "I like to go skiing.",
    "I am a theater lover.",
    "I enjoy going to the gym.",
    "I like to go for long walks.",
    "I am a puzzle enthusiast.",
    "I enjoy going to the zoo.",
    "I like to go to amusement parks.",
    "I am a beach lover.",
    "I enjoy going to the library.",
    "I like to go to the park.",
    "My name is also Jarly.",
]


def add_dummy_data(collection_name):
    for i, text in enumerate(dummy_data):
        response = requests.post(
            url="http://localhost:11434/api/embed",
            json={"model": "mxbai-embed-large:335m", "input": text},
        )

        data = response.json()
        embeddings = data["embeddings"]

        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=i,
                    vector=embeddings[0],
                    payload={"text": text},
                )
            ],
        )


def search_with_mxbai_embed_large(collection_name, prompt, top_k=5):
    prompt_prefix = "Represent this sentence for searching relevant passages:"
    adjusted_prompt = f"{prompt_prefix} {prompt}"

    response = requests.post(
        url="http://localhost:11434/api/embed",
        json={"model": "mxbai-embed-large:335m", "input": adjusted_prompt},
    )

    data = response.json()
    embeddings = data["embeddings"][0]

    search_results = client.query_points(
        collection_name=collection_name,
        query=embeddings,
        with_payload=True,
        limit=top_k,
    )

    return search_results


def ask_llm(prompt, model="llama3.2:latest", stream=False):
    response = requests.post(
        url="http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {"num_ctx": 10000},
        },
    )

    data = response.json()
    return data


def get_relevant_context_from_vector_db(collection_name, prompt, top_k=5):
    print(
        f"Searching for relevant context in collection '{collection_name}' for prompt: '{prompt}'"
    )
    results = search_with_mxbai_embed_large(collection_name, prompt, top_k)
    return results.points


def show_contexts_from_vector_db(contexts):
    print("Contexts retrieved from vector database:")
    for context in contexts:
        print(
            f"ID: {context.id}, Score: {context.score}, Text: {context.payload['text']}"
        )


def clean_vector_db_contexts(contexts):
    cleaned_contexts = []
    for context in contexts:
        text = context.payload.get("text", "")
        if text:
            cleaned_contexts.append(text)
    return cleaned_contexts


def prepare_augmented_prompt_for_llm(cleaned_contexts, prompt):
    cleaned_llm_context = "\n".join(cleaned_contexts)
    augmented_llm_prompt = f"Answer the following question based on the context provided:\n\nContext:\n{cleaned_llm_context}\n\nQuestion:\n{prompt}"
    return augmented_llm_prompt


def clean_llm_response(llm_response):
    cleaned_response = llm_response.get("response", "").strip()
    return cleaned_response


def print_divider():
    print("\n" + "-" * 50 + "\n")


def extract_metadata_from_text(file_path):
    """Extract metadata from the top of a text file.

    Supports lines like "Title: ..." and "Author: ..." and retains any
    additional key/value pairs encountered before the first blank line.
    """
    file_path = Path(file_path)
    metadata = {
        "path": str(file_path),
        "name": file_path.name,
        "title": None,
        "author": None,
        "other": {},
    }

    if not file_path.exists() or not file_path.is_file():
        return metadata

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                break
            if stripped.startswith("-") and len(set(stripped)) == 1:
                continue
            if ":" in stripped:
                key, value = stripped.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                if not value:
                    continue
                if key == "title":
                    metadata["title"] = value
                elif key == "author":
                    metadata["author"] = value
                else:
                    metadata["other"][key] = value
            elif stripped.lower().startswith("file:"):
                metadata["other"]["file"] = stripped[5:].strip()
    return metadata


def extract_content_from_text(file_path):
    """Return the main content after the metadata header in a text file."""
    file_path = Path(file_path)
    if not file_path.exists() or not file_path.is_file():
        return ""

    raw_text = file_path.read_text(encoding="utf-8")
    lines = raw_text.splitlines()
    header_seen = False
    content_start = 0

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if header_seen:
                content_start = index + 1
                break
            continue

        if stripped.startswith("-") and len(set(stripped)) == 1:
            if header_seen:
                content_start = index + 1
                break
            continue

        if ":" in stripped:
            header_seen = True
            continue

        if header_seen:
            content_start = index
            break
        else:
            # No metadata header found; treat entire file as content.
            return raw_text.strip()

    content_lines = lines[content_start:]
    return "\n".join(content_lines).strip()


def extract_from_text(file_path):
    """Extract both metadata and content from a text file."""
    metadata = extract_metadata_from_text(file_path)
    content = extract_content_from_text(file_path)
    return metadata, content


def create_chunks(text, chunk_size=200, chunk_overlap=50):
    """Split text into overlapping chunks for vector storage.

    Each chunk contains up to `chunk_size` words, with `chunk_overlap`
    words shared between consecutive chunks.
    """
    if not text:
        return []

    words = text.split()
    if not words:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be 0 or greater")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = end - chunk_overlap

    return chunks


def prepare_context_for_llm(text, metadata, chunk_size=200, chunk_overlap=50):
    """Split text into chunks and attach metadata and UUIDs for each chunk.

    Returns a list of payload dicts with keys: `id`, `text`, and `meta`.
    The `meta` dict contains the original file-level metadata plus
    `chunk_index` and `source` information.
    """
    chunks = create_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    payloads = []
    for idx, chunk_text in enumerate(chunks):
        uid = uuid.uuid4().hex
        meta = {
            "title": metadata.get("title"),
            "author": metadata.get("author"),
            "source": metadata.get("path"),
            "name": metadata.get("name"),
            "chunk_index": idx,
            **(metadata.get("other", {}) or {}),
        }
        payload = {
            "id": uid,
            "text": chunk_text,
            "meta": meta,
        }
        payloads.append(payload)

    return payloads


def upsert_chunks_to_qdrant(
    collection_name, payloads, embeddings, vector_field_name=None
):
    """Upsert chunk payloads and corresponding embeddings into Qdrant.

    - `payloads` should be a list of dicts as returned by
      `prepare_context_for_llm` (with `id`, `text`, `meta`).
    - `embeddings` should be a list of vectors with the same order/length
      as `payloads`.
    """
    if not payloads:
        return None
    if not embeddings:
        raise ValueError("embeddings list is empty")
    if len(payloads) != len(embeddings):
        raise ValueError("Number of payloads and embeddings must match")

    points = []
    for payload, vector in zip(payloads, embeddings):
        p = PointStruct(
            id=payload["id"],
            vector=vector,
            payload={**payload["meta"], "text": payload["text"]},
        )
        points.append(p)

    client.upsert(collection_name=collection_name, points=points)
    return True


def create_text_embeddings(texts, model="mxbai-embed-large:335m"):
    """Create embeddings for a list of texts using the specified model."""
    if not texts:
        return []

    response = requests.post(
        url="http://localhost:11434/api/embed",
        json={"model": model, "input": texts},
    )

    data = response.json()
    embeddings = data.get("embeddings", [])
    return embeddings


def add_new_context_to_vector_db(
    collection_name,
    file_path=None,
    folder_path=None,
    chunk_size=200,
    chunk_overlap=50,
    model="mxbai-embed-large:335m",
):
    """Add new file(s) to the Qdrant vector DB for LLM context.

    Either `file_path` or `folder_path` must be provided. When a folder is
    provided, every `.txt` file in the folder and subfolders is processed.
    """
    create_collection(collection_name)

    if bool(file_path) == bool(folder_path):
        raise ValueError("Provide exactly one of file_path or folder_path")

    files_to_process = []
    if file_path:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise ValueError(f"File not found: {file_path}")
        files_to_process = [path]
    else:
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Folder not found: {folder_path}")
        files_to_process = sorted(folder.rglob("*.txt"))

    total_chunks = 0
    for path in files_to_process:
        metadata, content = extract_from_text(path)
        payloads = prepare_context_for_llm(
            content,
            metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not payloads:
            continue

        embeddings = create_text_embeddings(
            [payload["text"] for payload in payloads], model=model
        )
        upsert_chunks_to_qdrant(collection_name, payloads, embeddings)
        total_chunks += len(payloads)

    return total_chunks


def main():
    # collection_name = "test"
    # create_collection(collection_name)
    # add_dummy_data(collection_name)

    collection_name = "articles"
    create_collection(collection_name)

    # metadata, content = extract_from_text(
    #     "dataset/articles/A_Coding_Agent_That_Gets_Out_Of_The_Way.txt"
    # )

    # payloads = prepare_context_for_llm(
    #     content, metadata, chunk_size=200, chunk_overlap=50
    # )

    # embeddings = create_text_embeddings(
    #     [payload["text"] for payload in payloads], model="mxbai-embed-large:335m"
    # )

    # upsert_chunks_to_qdrant(collection_name, payloads, embeddings)

    # chunks_count = add_new_context_to_vector_db(
    #     collection_name="articles",
    #     folder_path="dataset/articles",
    #     chunk_size=200,
    #     chunk_overlap=50,
    #     model="mxbai-embed-large:335m",
    # )
    # print(f"Total chunks added to vector DB: {chunks_count}")

    # exit()

    prompt = input("Enter a search prompt: ")
    # prompt = "What do I like to do?"
    contexts = get_relevant_context_from_vector_db(collection_name, prompt, top_k=5)

    print_divider()

    show_contexts_from_vector_db(contexts)

    print_divider()

    # cleaned_llm_context = "\n".join([text.payload["text"] for text in contexts])
    cleaned_llm_context = clean_vector_db_contexts(contexts)
    print("Cleaned LLM Context:")
    print(cleaned_llm_context)

    print_divider()

    augmented_llm_prompt = prepare_augmented_prompt_for_llm(cleaned_llm_context, prompt)
    print("Augmented LLM Prompt:")
    print(augmented_llm_prompt)

    print_divider()

    llm_model = "gemma3:12b-it-qat"
    llm_response = ask_llm(augmented_llm_prompt, model=llm_model)
    print("Raw LLM Response:")
    print(llm_response)

    print_divider()

    cleaned_response = clean_llm_response(llm_response)
    print("Cleaned LLM Response:")
    print(cleaned_response)

    print_divider()


if __name__ == "__main__":
    main()
