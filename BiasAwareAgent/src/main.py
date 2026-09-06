import chromadb
import json

from langchain_chroma import Chroma
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage


import tools
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def load_config():
    # Load Configuration
    with open("config/config.json") as json_data_file:
        config = json.load(json_data_file)
    
    return config


def init_vector_store(config):
    chroma_config = config["chroma"]
    chroma_client = chromadb.HttpClient(host=chroma_config["host"], port=chroma_config["port"])

    

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma(client=chroma_client,
           collection_name=chroma_config["collection"],
           embedding_function=embeddings,
           create_collection_if_not_exists=True)
    
    
    return vector_store


def add_documents(vector_store: Chroma):
    print("******** add_documents() CALLED ********")

    file_path = "data/news_articles.txt"
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    print("Documents loaded:", len(documents))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=25
    )

    texts = text_splitter.split_documents(documents)

    print("Chunks:", len(texts))

    vector_store.add_documents(texts)

    print("Documents after insert:", vector_store._collection.count())

def reasoner(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

if __name__ == '__main__':
    # Load Configuration
    config = load_config()
    # Initialize Chroma Vector Store
    vector_store = init_vector_store(config=config)
    # Add Knowledge to the store if enabled
    if config["chroma"]["add_knowledge"]:
        add_documents(vector_store)
    # Tools that are available for Bias Aware Agent
    available_tools = [tools.news_articles_retrieval_tool(vector_store), tools.bias_detector]
    # Choose the LLM to use

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )
    # Bind tools with LLM
    llm_with_tools = llm.bind_tools(available_tools)
    # System message
    sys_msg = SystemMessage(content="You are a highly advanced bias detection system designed to analyze retrieved news articles for bias. Your task is to:"
                            "1. Answer the query based on the content of the article in a concise and factual manner."
                            "2. Analyze the retrieved content for bias by utilizing tools available"
                            "3. Provide a bias evaluation: If bias is output: This content contains bias. Include a brief explanation of why the content is biased, citing specific examples. If no bias is detected, output: This content appears unbiased")
    
    # Graph
    builder = StateGraph(MessagesState)
    # Add nodes
    builder.add_node("reasoner", reasoner)
    builder.add_node("tools", ToolNode(available_tools))
    # Add edges
    builder.add_edge(START, "reasoner")
    builder.add_conditional_edges(
        "reasoner",
        tools_condition,
    )
    builder.add_edge("tools", "reasoner")
    react_graph = builder.compile()
    import sys

    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        print("Paste your query. Press Enter twice when finished:")

        lines = []
        
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)

        query = " ".join(lines)
    messages = [HumanMessage(content=query)]
    messages = react_graph.invoke({"messages": messages})
    for m in messages['messages']:
        m.pretty_print()