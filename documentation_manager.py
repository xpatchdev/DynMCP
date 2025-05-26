#!/usr/bin/env python3

import os
import asyncio
import sys
from typing import List, Dict, Any, Optional, Union

# LangChain imports
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document

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


class DocumentationManager:
    """
    Class to manage technical documentation retrieval, crawling, and querying.
    This encapsulates the RAG functionality for managing documentation sites.
    """
    
    def __init__(self, config=None):
        """
        Initialize the DocumentationManager with the provided configuration.
        
        Args:
            config: Dictionary containing configuration values or an instance with attributes
        """
        # Set up configuration 
        if config is None:
            config = {}
            
        # Handle both dictionary and object with attributes (for backward compatibility)
        if isinstance(config, dict):
            # Initialize from dictionary
            self.server_api_host = config.get("server-api-host") or os.getenv("SERVER_API_HOST", "http://172.23.192.1:11435")
            self.model = config.get("model") or os.getenv("MODEL", "qwen3:0.6b")  # Use available model
            self.embedding_model = config.get("embedding-model") or os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
            self.server_type = os.getenv("SERVER_TYPE", "ollama")
            self.repository = config.get("repository") or os.getenv("REPOSITORY", None)
            self.persist_dir = config.get("persist-dir") or os.getenv("PERSIST_DIR", "./books/")
            self.mode = config.get("mode") or os.getenv("MODE", "CLI").upper()
            self.rag_markdown_dir = config.get("rag-markdown-dir") or os.getenv("RAG_MARKDOWN_DIR", None)
        else:
            # For backward compatibility with object-based configuration
            self.server_api_host = getattr(config, "server_api_host", None) or os.getenv("SERVER_API_HOST", "http://172.23.192.1:11435")
            self.model = getattr(config, "model", None) or os.getenv("MODEL", "qwen3:0.6b")  # Use available model
            self.embedding_model = getattr(config, "embedding_model", None) or os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
            self.server_type = os.getenv("SERVER_TYPE", "ollama")
            self.repository = getattr(config, "repository", None) or os.getenv("REPOSITORY", None)
            self.persist_dir = getattr(config, "persist_dir", None) or os.getenv("PERSIST_DIR", "./books/")
            self.mode = getattr(config, "mode", None) or os.getenv("MODE", "CLI").upper()
            self.rag_markdown_dir = getattr(config, "rag_markdown_dir", None) or os.getenv("RAG_MARKDOWN_DIR", None)
        
        # Find available server
        self.server_api_host = self._get_available_server()
        
        # Dictionary to store vector stores for each domain
        self.domain_vectorstores = {}
        
        # Create persistence directory if it doesn't exist
        os.makedirs(self.persist_dir, exist_ok=True)
        
    def _get_available_server(self):
        """
        Check which server from server_api_host is available.
        
        Returns:
            str: Available server URL
        """
        import requests
        
        server_hosts = self.server_api_host.split(",")
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
    
    def _get_embeddings(self):
        """
        Create embeddings based on server type.
        
        Returns:
            Embeddings: A LangChain embeddings object
        """
        if self.server_type.lower() == "ollama":
            return OllamaEmbeddings(
                base_url=self.server_api_host,
                model=self.embedding_model
            )
        else:  # Default to OpenAI embeddings
            return OpenAIEmbeddings(
                base_url=f"{self.server_api_host}/v1",
                model=self.embedding_model
            )
    
    def _get_llm(self):
        """
        Create LLM based on server type.
        
        Returns:
            LLM: A LangChain LLM object
        """
        if self.server_type.lower() == "ollama":
            return ChatOllama(
                model=self.model,
                base_url=self.server_api_host,
                format="json",  # Use json for proper model output
                temperature=0.1,
            )
        else:
            return ChatOpenAI(
                base_url=f"{self.server_api_host}/v1",
                model=self.model,
                temperature=0.1,
            )
    
    def list_available_domains(self):
        """
        List all available documentation domains without loading the vector stores.
        This is more efficient than list_crawled_sites() as it only checks
        the directories rather than loading the vector stores.
        
        Returns:
            List[str]: A list of available documentation domain names
        """
        # First return any domains we've already loaded
        if self.domain_vectorstores:
            return list(self.domain_vectorstores.keys())
            
        # If no domains are loaded, just check the directory structure
        domain_names = []
        if os.path.exists(self.persist_dir):
            for domain_name in os.listdir(self.persist_dir):
                domain_dir = os.path.join(self.persist_dir, domain_name)
                if os.path.isdir(domain_dir):
                    # Skip the temporary markdown directory
                    if domain_name == "temp_markdown":
                        continue
                    # Only include directories that have FAISS index files
                    if os.path.exists(os.path.join(domain_dir, "index.faiss")):
                        domain_names.append(domain_name)
        
        return domain_names
        
    def _load_specific_domain_vectorstore(self, domain: str) -> bool:
        """
        Load a specific domain's vector store.
        
        Args:
            domain (str): Name of the domain to load
            
        Returns:
            bool: True if successful, False otherwise
        """
        domain_dir = os.path.join(self.persist_dir, domain)
        if not os.path.isdir(domain_dir):
            print(f"Error: Domain directory {domain_dir} does not exist")
            return False
            
        try:
            print(f"Loading vector store for domain: {domain}...")
            embeddings = self._get_embeddings()
            vectorstore = FAISS.load_local(domain_dir, embeddings, allow_dangerous_deserialization=True)
            self.domain_vectorstores[domain] = vectorstore
            print(f"✅ Loaded vector store for {domain}")
            return True
        except Exception as e:
            print(f"Error loading vector store for {domain}: {str(e)}")
            return False
            
    def query_crawled_sites(self, query: str, domain: Optional[str] = None):
        """
        Query technical documentation with an optional domain filter.
        
        Args:
            query (str): The query to search for in the documentation
            domain (Optional[str]): Specific domain to query (e.g., SDL, OpenGL, PyGame)
            
        Returns:
            str: Query results from the documentation
        """
        # Check available domains without loading vector stores
        available_domains = self.list_available_domains()
        if not available_domains:
            return "No documentation has been crawled yet. Use the crawl_site method first."
            
        # If domain is specified, make sure it exists
        if domain and domain not in available_domains:
            return f"Documentation for domain '{domain}' is not available. Available domains: {', '.join(available_domains)}"
            
        # Load only the necessary domains
        domains_to_query = []
        
        if domain:
            # If a specific domain is requested and it's not loaded yet, load it
            if domain not in self.domain_vectorstores:
                if not self._load_specific_domain_vectorstore(domain):
                    return f"Error loading documentation for domain '{domain}'."
            domains_to_query = [domain]
        else:
            # No specific domain requested - don't automatically load any domains
            # If user wants to query all domains, they should use the query_all_domains method
            return "Please specify a domain to query. Available domains: " + ", ".join(available_domains)
        
        # Get LLM
        llm = self._get_llm()
        
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
        
        # Query only the domains we've identified
        for domain_name in domains_to_query:
            if domain_name not in self.domain_vectorstores:
                continue
                
            vectorstore = self.domain_vectorstores[domain_name]
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
                # Use invoke() instead of __call__ to avoid deprecation warning
                response = qa_chain.invoke({"query": query})
                
                # Only include domains with relevant results
                if response.get("result", "").strip():
                    sources = set()
                    for doc in response.get("source_documents", []):
                        if "source" in doc.metadata:
                            sources.add(doc.metadata["source"])
                    
                    domain_results = f"### {domain_name} Documentation Results\n\n"
                    # Ensure the result is properly formatted as markdown
                    domain_results += response["result"].strip()
                    domain_results += "\n\n#### Sources:\n"
                    for source in sources:
                        domain_results += f"- {source}\n"
                    
                    results += domain_results + "\n\n"
            except Exception as e:
                print(f"Error querying {domain_name}: {str(e)}")
        
        return results if results else "No relevant information found in the documentation."
    
    def _get_domain_from_url(self, url: str) -> str:
        """
        Extract the domain/topic from a repository URL.
        
        Args:
            url (str): Repository URL
            
        Returns:
            str: Domain name
        """
        # Extract domain from GitHub URL if not in predefined list
        if "github.com" in url:
            parts = url.split("/")
            try:
                repo_index = parts.index("github.com") + 2
                if len(parts) > repo_index:
                    return parts[repo_index]
            except (ValueError, IndexError):
                # Handle cases where the URL format doesn't match expected pattern
                pass
        
        # Extract last part of the URL if possible
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            if domain:
                # Remove www. prefix and .com suffix
                if domain.startswith('www.'):
                    domain = domain[4:]
                return domain.split('.')[0]
        except Exception:
            pass
            
        # Fallback to simple extraction method
        try:
            clean_url = url.rstrip('/')
            last_part = clean_url.split('/')[-1]
            if last_part:
                return last_part
        except Exception:
            pass
            
        return "general"
            
    async def crawl_site(self, site_url: str, domain: Optional[str] = None, keywords: Optional[List[str]] = None, max_pages: int = 10):
        """
        Crawl a website for documentation using improved async handling.
        
        Args:
            site_url (str): URL of the website to crawl
            domain (Optional[str]): Domain name to use for the crawled data
            keywords (Optional[List[str]]): Keywords to prioritize content
            max_pages (int): Maximum number of pages to crawl, defaults to 10
            
        Returns:
            bool: True if crawling was successful, False otherwise
        """
        try:
            print(f"Crawling website: {site_url} with max_pages={max_pages}")
            
            # Get domain from URL if not provided
            if domain is None:
                domain = self._get_domain_from_url(site_url)
            
            # Default keywords if none provided
            if keywords is None:
                keywords = ["api", "documentation", "tutorial", "example", "guide", "reference"]
            
            # Get the base domain for filtering
            from urllib.parse import urlparse
            parsed_url = urlparse(site_url)
            base_domain = parsed_url.netloc
            if not base_domain:
                base_domain = site_url.split('/')[0]
            
            # Create filters to focus on documentation content
            filter_chain = FilterChain([
                DomainFilter(allowed_domains=[base_domain]),
                ContentTypeFilter(allowed_types=["text/html", "text/plain", "text/markdown"]),
            ])
            
            # Create scorer to prioritize documentation content
            keyword_scorer = KeywordRelevanceScorer(keywords=keywords, weight=1.0)
            
            # Configure the crawler with the max_pages parameter
            config = CrawlerRunConfig(
                deep_crawl_strategy=BestFirstCrawlingStrategy(
                    max_depth=3,
                    include_external=False,
                    filter_chain=filter_chain,
                    url_scorer=keyword_scorer,
                    max_pages=max_pages
                ),
                scraping_strategy=LXMLWebScrapingStrategy(),
                stream=True,
                verbose=True,
                cache_mode=CacheMode.ENABLED
            )
            
            # Crawled results collection
            crawled_data = []
            
            # Create a crawler instance
            crawler = AsyncWebCrawler()
            
            try:
                print("Starting crawler...")
                result_container = await crawler.arun(url=site_url, config=config)
                
                if result_container is None:
                    print("Error: Crawler returned None")
                    return False
                    
                print(f"Crawler returned result of type: {type(result_container)}")
                
                # Process the results
                count = 0
                async for page in result_container:
                    count += 1
                    print(f"\nProcessing document {count}:")
                    print(f"URL: {page.url}")
                    
                    title = getattr(page, "title", f"Document {count}")
                    print(f"Title: {title}")
                    
                    # Extract content from either text or HTML
                    content = ""
                    if hasattr(page, "text") and page.text:
                        content = page.text
                        content_source = "text attribute"
                    elif hasattr(page, "html") and page.html:
                        # Parse HTML to extract text
                        try:
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(page.html, 'html.parser')
                            content = soup.get_text()
                            content_source = "parsed HTML"
                        except ImportError:
                            print("Warning: BeautifulSoup not available, skipping HTML parsing")
                            content_source = "None - BS4 not available"
                        except Exception as e:
                            print(f"Error parsing HTML: {str(e)}")
                            content_source = f"Error parsing HTML: {str(e)}"
                    else:
                        content_source = "None available"
                    
                    content_length = len(content) if content else 0
                    print(f"Content: {content_length} chars from {content_source}")
                    
                    if content_length > 0:
                        # Save the document data
                        doc_data = {
                            "url": page.url,
                            "title": title,
                            "content": content,
                            "metadata": getattr(page, "metadata", {}),
                            "domain": domain
                        }
                        crawled_data.append(doc_data)
            
                print(f"\nCompleted crawling. Found {len(crawled_data)} documents with content.")
                
                # Check if we got any data
                if not crawled_data:
                    print(f"No data retrieved from {site_url}")
                    return False
                
                # Create vector store
                try:
                    # Process documents for RAG
                    from langchain_core.documents import Document
                    
                    # Convert crawled data to LangChain documents
                    documents = []
                    for doc in crawled_data:
                        content = doc.get("content", "")
                        if not content or content.strip() == "":
                            # Skip documents with no content, but log it
                            print(f"Warning: Empty content from {doc.get('url', 'unknown URL')}, skipping")
                            continue
                            
                        documents.append(
                            Document(
                                page_content=content,
                                metadata={
                                    "source": doc.get("url", "unknown"),
                                    "title": doc.get("title", "No Title"),
                                    "domain": domain,
                                }
                            )
                        )
                    
                    # Ensure we have at least one document
                    if not documents:
                        print("Warning: No valid documents to process")
                        return False
                    
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
                    
                    # Create embeddings
                    embeddings = self._get_embeddings()
                    
                    # Create the domain directory
                    domain_dir = os.path.join(self.persist_dir, domain.lower())
                    os.makedirs(domain_dir, exist_ok=True)
                    
                    # Create and save the vector store
                    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
                    vectorstore.save_local(domain_dir)
                    
                    # Update domain dictionary
                    self.domain_vectorstores[domain] = vectorstore
                    
                    print(f"Successfully crawled {len(crawled_data)} documents from {site_url} for the {domain} domain.")
                    return True
                except Exception as e:
                    print(f"Error creating vector store: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return False
                    
            except Exception as e:
                print(f"Error during crawling: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
        
        except Exception as e:
            print(f"Error crawling site: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def get_available_models(self, server_host: Optional[str] = None) -> list:
        """
        Fetch available models from Ollama server.
        
        Args:
            server_host (str): The Ollama server host URL. If None, uses self.server_api_host
            
        Returns:
            list: List of available model names
        """
        if server_host is None:
            server_host = self.server_api_host
            
        try:
            import requests
            response = requests.get(f"{server_host}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            
            models = []
            if "models" in data:
                for model in data["models"]:
                    if "name" in model:
                        models.append(model["name"])
            
            return sorted(models)
        except Exception as e:
            print(f"Warning: Could not fetch models from {server_host}: {e}")
            return []

    def set_model(self, model_name: str, model_type: str = "LLM"):
        """
        Update the model configuration.
        
        Args:
            model_name (str): Name of the model to use
            model_type (str): Type of model ("LLM" or "Embedding")
        """
        if model_type.lower() == "llm":
            self.model = model_name
            print(f"Updated LLM model to: {model_name}")
        elif model_type.lower() == "embedding":
            self.embedding_model = model_name
            print(f"Updated embedding model to: {model_name}")
        else:
            print(f"Unknown model type: {model_type}")

    def select_and_update_model(self, model_type: str = "LLM"):
        """
        Interactively select and update a model.
        
        Args:
            model_type (str): Type of model ("LLM" or "Embedding")
            
        Returns:
            bool: True if model was updated, False otherwise
        """
        # Removed interactive selection since select_model_interactively is no longer available
        print("Interactive model selection is not available.")
        return False


if __name__ == "__main__":
    # Simple test for the DocumentationManager
    async def test():
        try:
            dm = DocumentationManager()
            print("DocumentationManager initialized successfully")
            
            # Use pygame documentation website
            site_url = "https://pygame.org/docs"
            domain = "pygame_docs_test"
            
            success = await dm.crawl_site(site_url, domain=domain, max_pages=5)
            
            if success:
                print("Successfully crawled pygame.org/docs")
                result = dm.query_crawled_sites("How do I draw shapes?", domain=domain)
                print("\nQuery result:")
                print(result)
            else:
                print("Failed to crawl pygame.org/docs")
        except Exception as e:
            print(f"Test failed with error: {str(e)}")
            import traceback
            traceback.print_exc()

    # Handle asyncio event loop properly to avoid errors
    try:
        # Check if there's already a running event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                print("Event loop is already running, creating a new one")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # No event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Run the test function
        asyncio.run(test())
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            # Create a new event loop if needed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(test())
            finally:
                loop.close()
        else:
            print(f"Runtime error: {str(e)}")
            import traceback
            traceback.print_exc()
