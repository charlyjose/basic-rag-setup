import requests
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


def main():
    collection_name = "test"
    # create_collection(collection_name)
    # add_dummy_data(collection_name)

    # prompt = input("Enter a search prompt: ")
    prompt = "What do I like to do?"
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

    llm_response = ask_llm(augmented_llm_prompt, model="llama3.2:latest")
    # print("Raw LLM Response:")
    # print(llm_response)

    print_divider()

    cleaned_response = clean_llm_response(llm_response)
    print("Cleaned LLM Response:")
    print(cleaned_response)

    print_divider()


if __name__ == "__main__":
    main()
