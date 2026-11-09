# This is a simple RAG agent that runs in the terminal.
# It uses the tool called rag_search to find text in ChromaDB,
# and then it gives the answer to the user.

import json
import os
import sys

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sentence_transformers import SentenceTransformer

# the same collection that we filled in index_documents.py
COLLECTION_NAME = "finalproject_docs"

# the embedding model we use for the text
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# how many chunks we want from ChromaDB
RESULT_COUNT = 10


# this model checks the input of our tool
class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, description="Question about the documents.")


# this model checks the output of our tool
class RagResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context: str = Field(min_length=1, description="The text found in the documents.")


# the RAG search tool
def rag_search(collection, model, arguments):
    # the agent sends the arguments as a JSON string
    # so first we read the JSON, then we validate it with pydantic
    try:
        data = json.loads(arguments)
        request = Question.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as error:
        return "error: " + str(error)

    # turn the question into an embedding
    question_vector = model.encode(
        request.question,
        normalize_embeddings=True,
    ).tolist()

    # search the 10 closest chunks in ChromaDB
    result = collection.query(
        query_embeddings=[question_vector],
        n_results=RESULT_COUNT,
        include=["documents", "metadatas"],
    )

    # build one simple string with all the chunks
    pieces = []
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]

    for i in range(len(documents)):
        text = documents[i]
        metadata = metadatas[i]

        # add the source and the page, so the agent can cite them
        piece = text + "\nSource: " + metadata["source"] + ", page " + str(metadata["page"])
        pieces.append(piece)

    context = "\n\n".join(pieces)

    # if nothing was found, we still return a simple string
    if context == "":
        context = "No relevant text was found in the documents."

    # validate the output string with pydantic
    try:
        rag_result = RagResult(context=context)
    except ValidationError as error:
        return "error: " + str(error)

    # return the plain string
    return rag_result.context


# one normal call to the LLM
def ask_agent(client, messages, tools):
    response = client.chat.completions.create(
        model=os.getenv("OPEN_ROUTER_MODEL_NAME", "openrouter/free"),
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    return response.choices[0].message


def main():
    # make sure special characters from the LLM can be printed on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    # read the .env file
    load_dotenv()

    # check the API key before starting
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        print("error: OPEN_ROUTER_API_KEY is missing in .env")
        return

    # connect to ChromaDB
    client_db = chromadb.HttpClient(host="localhost", port=8000)
    collection = client_db.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # load the embedding model
    print("Loading the embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # connect to OpenRouter using the key from .env
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # describe our tool to the agent
    tools = [
        {
            "type": "function",
            "function": {
                "name": "rag_search",
                "description": "Search the indexed documents and return the most relevant text.",
                "parameters": Question.model_json_schema(),
            },
        }
    ]

    # the system message tells the agent how to behave
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use the rag_search tool for "
                "questions about the documents. Answer only from the context "
                "you receive and mention the source and the page."
            ),
        }
    ]

    # ask the user questions until they type quit
    while True:
        question_text = input("you> ").strip()

        if question_text == "quit":
            break

        # validate the user question with pydantic
        try:
            request = Question.model_validate({"question": question_text})
        except ValidationError as error:
            print("error:", error)
            continue

        # add the question to the messages
        messages.append({"role": "user", "content": request.question})

        # ask the agent what to do
        response = ask_agent(client, messages, tools)
        messages.append(response.model_dump(exclude_none=True))

        # keep asking the agent while it wants to use tools
        rounds = 0
        while response.tool_calls and rounds < 5:
            rounds += 1

            for tool_call in response.tool_calls:
                if tool_call.function.name == "rag_search":
                    context = rag_search(collection, model, tool_call.function.arguments)
                else:
                    context = "error: unknown tool"

                # put the tool output in the messages
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": context,
                    }
                )

            # ask the agent again, now it has the context from the tool
            response = ask_agent(client, messages, tools)
            messages.append(response.model_dump(exclude_none=True))

        # show the answer in the terminal
        if response.content:
            print("agent>", response.content)
        else:
            print("agent> The model did not return an answer.")


# start the program
if __name__ == "__main__":
    main()
