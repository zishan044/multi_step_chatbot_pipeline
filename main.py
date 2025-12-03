import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain.agents import create_agent

def main():

    llm = ChatOllama(
        model="llama3.1",
        base_url="http://localhost:11434",
        temperature=0.5
    )

    embeddings = OllamaEmbeddings(model="llama3.1")

    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"
    )

    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    assert len(docs) == 1
    print(f"Total characters: {len(docs[0].page_content)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"Split blog post into {len(all_splits)} sub-documents.")

    document_ids = vector_store.add_documents(documents=all_splits)
    print(document_ids[:3])

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information to help answer a query."""
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    
    tools = [retrieve_context]

    prompt = (
        "You have access to a tool that retrieves context from a blog post. "
        "Use the tool to help answer user queries."
    )

    agent = create_agent(llm, tools, system_prompt=prompt)

    query = (
        "What is the standard method for Task Decomposition?\n\n"
        "Once you get the answer, look up common extensions of that method."
    )

    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()


    # response = llm.invoke("What is LangChain? very short answer.")
    # print(response)


if __name__ == "__main__":
    main()
