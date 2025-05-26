


I want to create an mcp service that uses crawl4ai to go through a github repo  like SDL OpenGL PyGame LangChain OpenAI

API documents and have solid knowledge base to pull expertise from.

using deepcrawl_example.py and rag_langchain I can start by combining them so I can point it at a site and stuff it into rag. I'll need to account for size constraints of embedding at some point. then I can take the mcp_langchain.py script and use the mcp stuff to make an ask an expert on the api ai mcp agent thingy.
Could you rewrite that so it would make more sense to an ai model.


I'll restructure this into a clearer project specification:

# Project Specification: Technical Documentation RAG System

## Objective
Create an MCP (Multi-Agent Communication Protocol) service that leverages crawl4ai to extract documentation from GitHub repositories (SDL, OpenGL, PyGame, LangChain, OpenAI) and build a comprehensive knowledge base to serve as an API expertise system.

## Implementation Plan
1. **Data Collection**: 
   - Utilize deepcrawl_example.py to scrape technical documentation from target repositories
   - Configure crawler to focus on API documentation, tutorials, and examples

2. **Knowledge Base Construction**:
   - Implement `rag_langchain` to process and store the crawled data
   - Apply chunking strategies to handle embedding size limitations
   - Create efficient vector store indexing for quick retrieval

3. **Service Integration**:
   - Extend `mcp_langchain.py` to create a multi-agent system
   - Develop specialized agents with domain expertise for each technology
   - Implement query routing and response synthesis capabilities

4. **User Interface**:
   - Create an "Ask the API Expert" interface that routes questions to appropriate domain agents
   - Support contextual follow-up questions and code examples

## Technical Considerations
- Manage token limits for embeddings and context windows
- Implement caching for frequently accessed documentation
- Ensure regular updates as repository documentation changes

F






xpatchdev: # Project Specification: Technical Documentation RAG System

## Objective
Create an MCP (Multi-Agent Communication Protocol) service that leverages crawl4ai to extract documentation from GitHub repositories (SDL, OpenGL, PyGame, LangChain, OpenAI) and build a comprehensive knowledge base to serve as an API expertise system.

## Implementation Plan
1. **Data Collection**: 
   - Utilize deepcrawl_example.py to scrape technical documentation from target repositories
   - Configure crawler to focus on API documentation, tutorials, and examples

2. **Knowledge Base Construction**:
   - Implement `rag_langchain` to process and store the crawled data
   - Apply chunking strategies to handle embedding size limitations
   - Create efficient vector store indexing for quick retrieval

3. **Service Integration**:
   - Extend `mcp_langchain.py` to create a multi-agent system
   - Develop specialized agents with domain expertise for each technology
   - Implement query routing and response synthesis capabilities

4. **User Interface**:
   - Create an "Ask the API Expert" interface that routes questions to appropriate domain agents
   - Support contextual follow-up questions and code examples


   start by using the  rag_langchain.py script and deepcrawl_example.py file and start putting a draft together in ask_an_expert.py

GitHub Copilot: I'll help you create an MCP service that leverages crawl4ai for technical documentation extraction and RAG for retrieval. Let's first examine the existing files to understand what we're working with.Now I'll examine the deepcrawl_example.py file to understand how we can collect documentation:I'll check if there's an existing MCP file structure we can build upon:

xpatchdev: todo_langchain.py has MCP structure

GitHub Copilot: I'll help you create the ask_an_expert.py file by using the MCP structure from todo_langchain.py. First, let me examine the content of both todo_langchain.py, rag_langchain.py, and deepcrawl_example.py to understand how to integrate them for your technical documentation RAG system.I'll examine the rag_langchain.py and deepcrawl_example.py files to understand how to integrate them for our technical documentation RAG system.Let me continue examining the deepcrawl_example.py file to understand how it works for our technical documentation RAG system:I've examined the necessary files for creating our Technical Documentation RAG System. Now I'll implement the ask_an_expert.py file by combining the MCP structure from todo_langchain.py, the RAG capabilities from rag_langchain.py, and the web crawling functionality from deepcrawl_example.py.

Let's create the ask_an_expert.py file:
````
import math
import json
import os
import uuid
import asyncio
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import argparse

# LangChain imports
from langchain_core.tools import Tool, tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Crawl4AI imports
from crawl4ai import CrawlerRunConfig, AsyncWebCrawler, CacheMode
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import (
    FilterChain,
    URLPatternFilter,
    DomainFilter,
    ContentTypeFilter,
    ContentRelevanceFilter
)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer

# Simple spinner for loading indication
import Spinner

# MCP server import
from mcp.server.fastmcp import FastMCP

# Parse command line arguments for environment variables
parser = argparse.ArgumentParser(description="Technical Documentation RAG System")
parser.add_argument("--server-api-host", type=str, default=None, help="Server API host URL")
parser.add_argument("--model", type=str, default=None, help="Model name")
parser.add_argument("--embedding-model", type=str, default=None, help="Embedding model name")
parser.add_argument("--repository", type=str, default=None, help="GitHub repository to crawl")
parser.add_argument("--persist-dir", type=str, default=None, help="Directory to persist vector store")
parser.add_argument("--mode", type=str, default=None, choices=["CLI", "MCP"], help="Run mode")

# Only parse args if running as main
if __name__ == "__main__":
    args, unknown = parser.parse_known_args()
else:
    class DummyArgs:
        server_api_host = None
        model = None
        embedding_model = None
        repository = None
        persist_dir = None
        mode = None
    args = DummyArgs()

# Environment variables with command line override
SERVER_API_HOST = args.server_api_host or os.getenv("SERVER_API_HOST", "http://localhost:11434")
MODEL = args.model or os.getenv("MODEL", "qwen2.5-0.5b-instruct")
EMBEDDING_MODEL = args.embedding_model or os.getenv("EMBEDDING_MODEL", "qwen2.5-0.5b-instruct")
SERVER_TYPE = os.getenv("SERVER_TYPE", "openai")  # Default to openai if not specified
REPOSITORY = args.repository or os.getenv("REPOSITORY", None)
PERSIST_DIR = args.persist_dir or os.getenv("PERSIST_DIR", "chroma_db/api_expert")
MODE = args.mode or os.getenv("MODE", "CLI").upper()

# Enable chat history by default
ENABLE_CHAT_HISTORY = os.getenv("ENABLE_CHAT_HISTORY", "true").lower() == "true"

# Target repositories - these are our knowledge domains
TARGET_REPOS = {
    "SDL": "https://github.com/libsdl-org/SDL/tree/main/docs",
    "OpenGL": "https://github.com/KhronosGroup/OpenGL-Registry/tree/main/specs",
    "PyGame": "https://github.com/pygame/pygame/tree/main/docs",
    "LangChain": "https://github.com/langchain-ai/langchain/tree/master/docs",
    "OpenAI": "https://github.com/openai/openai-cookbook"
}

# Dictionary to store vector stores for each domain
DOMAIN_VECTORSTORES = {}

async def crawl_repository(repo_url: str, keywords: List[str] = None) -> List[Dict]:
    """
    Crawl a GitHub repository for technical documentation.
    
    Args:
        repo_url (str): URL of the GitHub repository to crawl
        keywords (List[str], optional): Keywords to prioritize content. Defaults to None.
        
    Returns:
        List[Dict]: List of crawled content with metadata
    """
    print(f"Crawling repository: {repo_url}")
    
    # Default keywords if none provided
    if keywords is None:
        keywords = ["api", "documentation", "tutorial", "example", "guide", "reference"]
    
    # Create filters to focus on documentation files
    filter_chain = FilterChain([
        URLPatternFilter(patterns=["*.md", "*.rst", "*.txt", "*/docs/*", "*/examples/*"]),
        ContentTypeFilter(allowed_types=["text/plain", "text/markdown", "text/x-rst", "text/html"]),
    ])
    
    # Create scorer to prioritize documentation content
    keyword_scorer = KeywordRelevanceScorer(keywords=keywords, weight=1.0)
    
    # Configure the crawler
    config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=3,
            include_external=False,
            filter_chain=filter_chain,
            url_scorer=keyword_scorer,
            max_pages=100  # Limit pages to avoid extremely large crawls
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        stream=True,
        verbose=True,
        cache_mode=CacheMode.PERSISTENT
    )
    
    results = []
    
    async with AsyncWebCrawler() as crawler:
        async for result in await crawler.arun(url=repo_url, config=config):
            # Extract relevant data from result
            doc_data = {
                "url": result.url,
                "title": result.title,
                "content": result.text,
                "metadata": result.metadata,
                "domain": get_domain_from_url(repo_url)
            }
            results.append(doc_data)
            print(f"Crawled: {result.url}")
    
    print(f"Completed crawling {len(results)} documents from {repo_url}")
    return results

def get_domain_from_url(url: str) -> str:
    """
    Extract the domain/topic from a repository URL.
    
    Args:
        url (str): Repository URL
        
    Returns:
        str: Domain name
    """
    for domain, repo_url in TARGET_REPOS.items():
        if repo_url in url:
            return domain
    
    # Extract domain from GitHub URL if not in predefined list
    if "github.com" in url:
        parts = url.split("/")
        try:
            repo_index = parts.index("github.com") + 2
            if len(parts) > repo_index:
                return parts[repo_index]
        except ValueError:
            pass
    
    return "general"

def create_rag_from_crawled_data(crawled_data: List[Dict], persist_directory: Optional[str] = None) -> Chroma:
    """
    Create a RAG from crawled documentation data.
    
    Args:
        crawled_data (List[Dict]): Crawled repository data
        persist_directory (Optional[str]): Directory to persist the vector store
        
    Returns:
        Chroma: Vector store containing the processed documents
    """
    from langchain_core.documents import Document
    
    # Convert crawled data to LangChain documents
    documents = []
    for doc in crawled_data:
        documents.append(
            Document(
                page_content=doc["content"],
                metadata={
                    "source": doc["url"],
                    "title": doc.get("title", ""),
                    "domain": doc.get("domain", "general"),
                }
            )
        )
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Ensure chunks have proper text content
    for chunk in chunks:
        if not isinstance(chunk.page_content, str):
            chunk.page_content = str(chunk.page_content)
    
    # Create embeddings based on SERVER_TYPE
    if SERVER_TYPE.lower() == "ollama":
        embeddings = OllamaEmbeddings(
            base_url=SERVER_API_HOST,
            model=EMBEDDING_MODEL
        )
    else:  # Default to OpenAI embeddings
        embeddings = OpenAIEmbeddings(
            api_key="lm-studio",  # For LM Studio compatibility
            base_url=f"{SERVER_API_HOST}/v1",
            model=EMBEDDING_MODEL
        )
    
    # Create and return the vector store
    if persist_directory:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vectorstore.persist()
    else:
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    
    return vectorstore

def load_or_create_domain_vectorstores() -> Dict[str, Chroma]:
    """
    Load existing vector stores or create new ones for each domain.
    
    Returns:
        Dict[str, Chroma]: Dictionary of domain vector stores
    """
    domain_vectorstores = {}
    
    for domain, repo_url in TARGET_REPOS.items():
        domain_dir = f"{PERSIST_DIR}/{domain.lower()}"
        
        try:
            # Try to load existing vector store
            if os.path.exists(domain_dir):
                print(f"Loading existing vector store for {domain}...")
                
                # Create embeddings
                if SERVER_TYPE.lower() == "ollama":
                    embeddings = OllamaEmbeddings(
                        base_url=SERVER_API_HOST,
                        model=EMBEDDING_MODEL
                    )
                else:
                    embeddings = OpenAIEmbeddings(
                        api_key="lm-studio",
                        base_url=f"{SERVER_API_HOST}/v1",
                        model=EMBEDDING_MODEL
                    )
                
                # Load the vector store
                vectorstore = Chroma(persist_directory=domain_dir, embedding_function=embeddings)
                domain_vectorstores[domain] = vectorstore
                print(f"✅ Loaded vector store for {domain}")
            else:
                print(f"Vector store for {domain} not found. Please run the crawler first.")
        except Exception as e:
            print(f"Error loading vector store for {domain}: {str(e)}")
    
    return domain_vectorstores

def custom_input(prompt_text=""):
    """
    Custom input function with input history management.
    
    Args:
        prompt_text (str): The prompt to display before input
        
    Returns:
        str: The user input string
    """
    try:
        import termios
        import tty
        import sys
        
        # Save old terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        
        # Initialize buffer and history position
        buffer = []
        history_pos = len(input_history)
        
        # Print prompt
        sys.stdout.write(prompt_text)
        sys.stdout.flush()
        
        try:
            tty.setraw(sys.stdin.fileno())
            
            while True:
                # Read a single character
                char = sys.stdin.read(1)
                
                # Check for Ctrl+J (down) - ASCII code 10
                if ord(char) == 10:  # Ctrl+J
                    if history_pos < len(input_history) - 1:
                        history_pos += 1
                        # Clear current line
                        sys.stdout.write('\r' + ' ' * (len(prompt_text) + len(buffer)) + '\r')
                        buffer = list(input_history[history_pos])
                        sys.stdout.write(prompt_text + ''.join(buffer))
                    elif history_pos == len(input_history) - 1:
                        history_pos = len(input_history)
                        # Clear current line
                        sys.stdout.write('\r' + ' ' * (len(prompt_text) + len(buffer)) + '\r')
                        buffer = []
                        sys.stdout.write(prompt_text)
                        
                # Check for Ctrl+K (up) - ASCII code 11
                elif ord(char) == 11:  # Ctrl+K
                    if history_pos > 0 and input_history:
                        history_pos -= 1
                        # Clear current line
                        sys.stdout.write('\r' + ' ' * (len(prompt_text) + len(buffer)) + '\r')
                        buffer = list(input_history[history_pos])
                        sys.stdout.write(prompt_text + ''.join(buffer))
                
                # Handle backspace
                elif ord(char) == 127:
                    if buffer:
                        buffer.pop()
                        # Erase last character
                        sys.stdout.write('\b \b')
                
                # Handle Enter/Return
                elif ord(char) == 13:  # Carriage return
                    sys.stdout.write('\n')
                    break
                
                # Handle normal character input
                elif ord(char) >= 32:  # Printable characters
                    buffer.append(char)
                    sys.stdout.write(char)
                
                sys.stdout.flush()
            
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            
        return ''.join(buffer)
        
    except (ImportError, AttributeError):
        # Fall back to standard input if terminal handling fails
        print("\nWarning: Custom input mode not available. Using standard input.")
        return input(prompt_text)

def get_available_server():
    """
    Check which server from SERVER_API_HOST is available.
    
    Returns:
        str: Available server URL
    """
    server_hosts = SERVER_API_HOST.split(",")
    for host in server_hosts:
        host = host.strip()
        print(f"Trying server at: {host}")
        try:
            response = requests.get(f"{host}/", timeout=5)
            if response.status_code == 200:
                return host
        except requests.RequestException:
            continue
    raise ConnectionError("No available servers found in SERVER_API_HOST list.")

@tool("crawl_repository", description="Crawl a GitHub repository for technical documentation")
async def crawl_repository_tool(repo_url: str, keywords: List[str] = None) -> str:
    """
    Tool to crawl a GitHub repository for technical documentation.
    
    Args:
        repo_url (str): URL of the GitHub repository to crawl
        keywords (List[str], optional): Keywords to prioritize content. Defaults to None.
        
    Returns:
        str: Summary of crawled content
    """
    crawled_data = await crawl_repository(repo_url, keywords)
    
    # Get domain from URL
    domain = get_domain_from_url(repo_url)
    
    # Create or update vector store
    domain_dir = f"{PERSIST_DIR}/{domain.lower()}"
    os.makedirs(domain_dir, exist_ok=True)
    
    vectorstore = create_rag_from_crawled_data(crawled_data, domain_dir)
    
    # Update global dictionary
    DOMAIN_VECTORSTORES[domain] = vectorstore
    
    return f"Successfully crawled {len(crawled_data)} documents from {repo_url} for the {domain} domain."

@tool("query_documentation", description="Query technical documentation by domain")
def query_documentation(query: str, domain: Optional[str] = None) -> str:
    """
    Query technical documentation with an optional domain filter.
    
    Args:
        query (str): The query to search for in the documentation
        domain (Optional[str]): Specific domain to query (e.g., SDL, OpenGL, PyGame)
        
    Returns:
        str: Query results from the documentation
    """
    # Load vector stores if not already loaded
    global DOMAIN_VECTORSTORES
    if not DOMAIN_VECTORSTORES:
        DOMAIN_VECTORSTORES = load_or_create_domain_vectorstores()
    
    # If no vector stores available, return error
    if not DOMAIN_VECTORSTORES:
        return "No documentation has been crawled yet. Use the crawl_repository tool first."
    
    # Get LLM
    if SERVER_TYPE.lower() == "ollama":
        llm = ChatOllama(
            model=MODEL,
            base_url=SERVER_API_HOST,
            format="json",
            temperature=0.1,
        )
    else:
        llm = ChatOpenAI(
            api_key="lm-studio",  # For LM Studio compatibility
            base_url=f"{SERVER_API_HOST}/v1",
            model=MODEL,
            temperature=0.1,
        )
    
    # Create prompt template for RAG
    prompt = ChatPromptTemplate(
        input_variables=["context", "question"],
        messages=[
            SystemMessage(content="""You are an API Documentation Expert. 
            Use the following pieces of technical documentation to answer the user's question.
            If you don't know the answer, say so clearly. Provide code examples when possible.
            Format your response in markdown."""),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=["context", "question"],
                    template="Question: {question}\n\nRelevant Documentation:\n{context}\n\nAnswer:"
                )
            )
        ]
    )
    
    results = ""
    
    # If domain specified, query only that domain
    if domain and domain in DOMAIN_VECTORSTORES:
        retriever = DOMAIN_VECTORSTORES[domain].as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10}
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        response = qa_chain({"query": query})
        sources = set()
        for doc in response.get("source_documents", []):
            if "source" in doc.metadata:
                sources.add(doc.metadata["source"])
        
        domain_results = f"### {domain} Documentation Results\n\n"
        domain_results += response["result"].strip()
        domain_results += "\n\n#### Sources:\n"
        for source in sources:
            domain_results += f"- {source}\n"
        
        results = domain_results
    else:
        # Query all domains and compile results
        for domain_name, vectorstore in DOMAIN_VECTORSTORES.items():
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3, "fetch_k": 5}
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
            
            try:
                response = qa_chain({"query": query})
                
                # Only include domains with relevant results
                if response.get("result", "").strip():
                    sources = set()
                    for doc in response.get("source_documents", []):
                        if "source" in doc.metadata:
                            sources.add(doc.metadata["source"])
                    
                    domain_results = f"### {domain_name} Documentation Results\n\n"
                    domain_results += response["result"].strip()
                    domain_results += "\n\n#### Sources:\n"
                    for source in sources:
                        domain_results += f"- {source}\n"
                    
                    results += domain_results + "\n\n"
            except Exception as e:
                print(f"Error querying {domain_name}: {str(e)}")
    
    return results if results else "No relevant information found in the documentation."

@tool("list_available_domains", description="List all available documentation domains")
def list_available_domains() -> str:
    """
    List all available documentation domains.
    
    Returns:
        str: A list of available documentation domains
    """
    # Load vector stores if not already loaded
    global DOMAIN_VECTORSTORES
    if not DOMAIN_VECTORSTORES:
        DOMAIN_VECTORSTORES = load_or_create_domain_vectorstores()
    
    if not DOMAIN_VECTORSTORES:
        return "No documentation has been crawled yet. Use the crawl_repository tool first."
    
    result = "### Available Documentation Domains\n\n"
    for domain, vectorstore in DOMAIN_VECTORSTORES.items():
        result += f"- {domain}\n"
    
    return result

@tool("add_domain", description="Add a new documentation domain")
async def add_domain(domain_name: str, repo_url: str, keywords: List[str] = None) -> str:
    """
    Add a new documentation domain by crawling a repository.
    
    Args:
        domain_name (str): Name for the new domain
        repo_url (str): URL of the GitHub repository to crawl
        keywords (List[str], optional): Keywords to prioritize content. Defaults to None.
        
    Returns:
        str: Result of the domain addition
    """
    # Add to TARGET_REPOS
    TARGET_REPOS[domain_name] = repo_url
    
    # Crawl the repository
    crawled_data = await crawl_repository(repo_url, keywords)
    
    # Create vector store
    domain_dir = f"{PERSIST_DIR}/{domain_name.lower()}"
    os.makedirs(domain_dir, exist_ok=True)
    
    vectorstore = create_rag_from_crawled_data(crawled_data, domain_dir)
    
    # Update global dictionary
    DOMAIN_VECTORSTORES[domain_name] = vectorstore
    
    return f"Successfully added new domain '{domain_name}' with {len(crawled_data)} documents from {repo_url}."

def create_tools():
    """
    Create LangChain tools from our functions.
    
    Returns:
        List: List of tools
    """
    return [
        crawl_repository_tool,
        query_documentation,
        list_available_domains,
        add_domain,
    ]

def chat_loop():
    """
    Main chat loop that processes user input and handles tool calls using LangChain.
    """
    # Display agent status on startup
    print("\n" + "="*50)
    print(" Technical Documentation RAG System Status ")
    print("="*50)
    
    SERVER_API_HOST = get_available_server()
    print(f"LLM API URL: {SERVER_API_HOST}")
    print(f"Model: {MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Provider: {SERVER_TYPE}")
    print(f"Persistence Directory: {PERSIST_DIR}")
    print(f"Chat History: {'Enabled' if ENABLE_CHAT_HISTORY else 'Disabled'}")
    print("="*50 + "\n")
    
    # Load domain vector stores
    global DOMAIN_VECTORSTORES
    DOMAIN_VECTORSTORES = load_or_create_domain_vectorstores()
    
    # Initialize LLM based on provider
    if SERVER_TYPE.lower() == "ollama":
        llm = ChatOllama(
            model=MODEL,
            base_url=SERVER_API_HOST,
            format="json",
            temperature=0.1,
        )
    else:  # OpenAI or compatible API
        llm = ChatOpenAI(
            api_key="lm-studio",  # for LM Studio compatibility
            base_url=f"{SERVER_API_HOST}/v1",
            model=MODEL,
            temperature=0.1,
        )

    # Create tools
    tools = create_tools()
    
    # Create the system prompt
    system_prompt = f"""The current date and time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.
    You are an API Documentation Expert, designed to help users understand and use various software libraries and APIs.
    
    You have access to documentation for the following domains:
    - SDL: Simple DirectMedia Layer, a cross-platform development library
    - OpenGL: A graphics rendering API
    - PyGame: A set of Python modules for writing games
    - LangChain: A framework for developing applications with LLMs
    - OpenAI: AI models and tools
    
    You can help users by:
    1. Answering questions about API functions, classes, and methods
    2. Providing code examples to demonstrate API usage
    3. Explaining technical concepts related to these domains
    4. Comparing approaches across different libraries
    
    Use the provided tools to:
    - Crawl repositories for additional documentation
    - Query the documentation for specific information
    - List available domains
    - Add new documentation domains
    
    Always be helpful, accurate, and provide code examples when possible.
    """
    
    # Create prompt messages, conditionally including chat history
    prompt_messages = [
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
    
    # Only include chat history if enabled
    if ENABLE_CHAT_HISTORY:
        prompt_messages.insert(1, MessagesPlaceholder(variable_name="chat_history"))
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages(prompt_messages)

    # Create an agent with tools
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    chat_history = []
    global input_history
    input_history = []
    
    print("API Documentation Expert (Type 'quit' to exit)")

    while True:
        try:
            user_input = custom_input("\nYou: ")
            
            user_input = user_input.strip()
            print("\n")
            
            if user_input.lower() == "quit":
                break
                
            # Add to history only if it's not empty and not the same as the last entry
            if user_input and (not input_history or user_input != input_history[-1]):
                input_history.append(user_input)

            try:
                with Spinner.Spinner("Thinking..."):
                    # Prepare invoke arguments
                    invoke_args = {"input": user_input}
                    
                    # Only include chat history if enabled
                    if ENABLE_CHAT_HISTORY:
                        invoke_args["chat_history"] = chat_history
                    
                    response = agent_executor.invoke(invoke_args)
                    result = response["output"]
                    
                    # Only add to chat history if enabled
                    if ENABLE_CHAT_HISTORY:
                        chat_history.append(HumanMessage(content=user_input))
                        chat_history.append(AIMessage(content=result))

                print(f"\nExpert: {result}")

            except Exception as e:
                print(f"\nError: {str(e)}")
                print(
                    f"\nError processing your request!\n\n"
                    f"Please ensure:\n"
                    f"1. Server is running at {SERVER_API_HOST}\n"
                    f"2. Model '{MODEL}' is available\n\n"
                )
        except KeyboardInterrupt:
            print("\nOperation cancelled by user. Enter a new query or type 'quit' to exit.")
            continue

# Initialize MCP server
mcp = FastMCP("Technical Documentation Expert MCP Server")

# Wrap LangChain tools for MCP
@mcp.tool(name=crawl_repository_tool.name, description=crawl_repository_tool.description)
async def crawl_repository_mcp(repo_url: str, keywords: List[str] = None) -> str:
    """MCP wrapper for crawl_repository_tool"""
    return await crawl_repository_tool.ainvoke(repo_url=repo_url, keywords=keywords)

@mcp.tool(name=query_documentation.name, description=query_documentation.description)
async def query_documentation_mcp(query: str, domain: Optional[str] = None) -> str:
    """MCP wrapper for query_documentation"""
    return query_documentation.invoke(query=query, domain=domain)

@mcp.tool(name=list_available_domains.name, description=list_available_domains.description)
async def list_available_domains_mcp() -> str:
    """MCP wrapper for list_available_domains"""
    return list_available_domains.invoke()

@mcp.tool(name=add_domain.name, description=add_domain.description)
async def add_domain_mcp(domain_name: str, repo_url: str, keywords: List[str] = None) -> str:
    """MCP wrapper for add_domain"""
    return await add_domain.ainvoke(domain_name=domain_name, repo_url=repo_url, keywords=keywords)

if __name__ == "__main__":
    # Create persist directory if it doesn't exist
    os.makedirs(PERSIST_DIR, exist_ok=True)
    
    if MODE == "MCP":
        # Run the MCP server with HTTP-SSE transport
        mcp.run(transport="sse")
    else:  # Default to CLI mode
        # Run the chat loop
        chat_loop()
````


Made changes.

I've successfully created the ask_an_expert.py file that implements a Technical Documentation RAG System following your specifications. This implementation combines:

1. The MCP (Multi-Agent Communication Protocol) structure from todo_langchain.py
2. The RAG capabilities from rag_langchain.py
3. The web crawling functionality from deepcrawl_example.py

The system has the following key features:

### Main Components
- **Web Crawler**: Uses Crawl4AI to scrape documentation from GitHub repositories
- **Knowledge Base**: Processes and stores the crawled data using LangChain's RAG functionality
- **Vector Store**: Creates efficient embeddings for quick retrieval of relevant documentation
- **Domain Experts**: Separate vector stores for each technology domain (SDL, OpenGL, PyGame, etc.)
- **Query Interface**: Supports contextual documentation queries with source attribution

### Available Tools
- `crawl_repository`: Crawls a GitHub repository for documentation
- `query_documentation`: Searches the knowledge base for specific information
- `list_available_domains`: Shows available technology domains
- `add_domain`: Add a new documentation domain to the system

### Running the System
You can run the system in two modes:
- **CLI Mode** (default): Interactive command-line interface
- **MCP Mode**: Runs as an MCP service that can be used by other agents

```bash
# CLI Mode
python ask_an_expert.py

# MCP Mode
python ask_an_expert.py --mode=MCP
```

### Customization
The system is highly configurable through environment variables or command-line arguments:
- `--server-api-host`: LLM API host URL
- `--model`: LLM model to use
- `--embedding-model`: Model for creating embeddings
- `--repository`: Specific GitHub repository to crawl
- `--persist-dir`: Directory to persist vector stores
- `--mode`: Run mode (CLI or MCP)

This implementation will serve as a solid foundation for your Technical Documentation RAG System. You can extend it further by adding more specialized domain experts or enhancing the query capabilities.