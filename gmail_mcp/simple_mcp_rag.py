# simple_mcp_rag.py

"""
Simplified MCP Gmail RAG implementation that uses the server directly
without the complex client-server communication setup.
"""

import asyncio
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from .mcp_gmail_server import GmailMCPServer
from logger import get_logger

logger = get_logger(__name__)

class SimpleMCPGmailRAG:
    def __init__(self, db_dir: str = "./chromadb_mcp"):
        self.db_dir = db_dir
        self.mcp_server = GmailMCPServer()
        self.embedder = None
        self.collection = None
        
    async def initialize(self):
        """Initialize the simplified MCP Gmail RAG system"""
        logger.info("Initializing Simple MCP Gmail RAG system...")
        
        # Authenticate with Gmail via MCP server
        auth_result = await self.mcp_server.authenticate_gmail()
        if "Successfully authenticated" not in auth_result.content[0].text:
            raise Exception(f"Gmail authentication failed: {auth_result.content[0].text}")
        
        logger.info("Gmail authentication successful")
        
        # Initialize embedding model
        self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        logger.info("Embedding model loaded")
        
        # Initialize ChromaDB
        client = chromadb.PersistentClient(path=self.db_dir, settings=Settings())
        self.collection = client.get_or_create_collection(name="gmail_simple_mcp_rag")
        logger.info("ChromaDB initialized")
    
    def get_existing_message_ids(self):
        """Get message IDs that are already indexed in the collection"""
        try:
            # Get all existing IDs from the collection
            result = self.collection.get(include=['metadatas'])
            existing_ids = result['ids'] if result['ids'] else []
            logger.info(f"Found {len(existing_ids)} existing indexed emails")
            return set(existing_ids)
        except Exception as e:
            logger.warning(f"Could not retrieve existing message IDs: {e}")
            return set()

    async def index_emails_mcp(self, max_results: int = 100):
        """Index emails using MCP server directly"""
        if not self.embedder or not self.collection:
            raise Exception("System not initialized. Call initialize() first.")
            
        logger.info(f"Indexing {max_results} emails using MCP...")
        
        # Get existing indexed message IDs
        existing_ids = self.get_existing_message_ids()
        
        # Fetch emails via MCP server
        result = await self.mcp_server.fetch_emails({
            "max_results": max_results,
            "include_body": True
        })
        
        if "Error" in result.content[0].text:
            logger.error(f"Failed to fetch emails via MCP: {result.content[0].text}")
            return
        
        # Parse the JSON response
        try:
            response_text = result.content[0].text
            # Extract JSON from the response (skip the header)
            json_start = response_text.find('[')
            if json_start == -1:
                logger.error("No JSON data found in response")
                return
                
            emails_data = json.loads(response_text[json_start:])
            logger.info(f"Fetched {len(emails_data)} emails via MCP")
            
            # Filter out already indexed emails
            new_emails = [email for email in emails_data if email['id'] not in existing_ids]
            
            if not new_emails:
                logger.info("No new emails to index - all emails are already indexed")
                return
            
            logger.info(f"Found {len(new_emails)} new emails to index out of {len(emails_data)} total")
            
            indexed_count = 0
            for email in new_emails:
                # Extract text content
                text_content = email.get('body', email.get('snippet', ''))
                if not text_content.strip():
                    logger.warning(f"Skipping message {email['id']} - no text content")
                    continue
                
                try:
                    # Create embedding
                    embedding = self.embedder.encode(text_content, show_progress_bar=False).tolist()
                    
                    # Store in ChromaDB
                    self.collection.upsert(
                        ids=[email['id']],
                        embeddings=[embedding],
                        metadatas=[{
                            "subject": email.get('subject', ''),
                            "from": email.get('from', ''),
                            "date": email.get('date', ''),
                            "snippet": text_content[:200]
                        }]
                    )
                    indexed_count += 1
                    logger.debug(f"Successfully indexed message {email['id']}")
                except Exception as e:
                    logger.error(f"Failed to index message {email['id']}: {e}")
                    continue
            
            logger.info(f"Indexing complete: {indexed_count}/{len(new_emails)} new emails successfully indexed")
            
        except Exception as e:
            logger.error(f"Error processing MCP email data: {e}")
    
    def search_emails_mcp(self, query: str, top_k: int = 5):
        """Search emails using embeddings"""
        if not self.embedder or not self.collection:
            raise Exception("System not initialized. Call initialize() first.")
            
        logger.info(f"Searching for: '{query}'")
        
        # Create query embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        if not results or not results['ids'] or not results['ids'][0]:
            logger.info("No results found")
            return
            
        logger.info(f"Found {len(results['ids'][0])} results")
        
        # Display results
        ids = results['ids'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        for i, (email_id, metadata, distance) in enumerate(
            zip(ids, metadatas, distances), 1
        ):
            logger.info(f"Result {i}: {email_id} (score {distance:.4f})")
            logger.info(f"  Subject: {metadata.get('subject', 'N/A')}")
            logger.info(f"  From: {metadata.get('from', 'N/A')}")
            logger.info(f"  Snippet: {metadata.get('snippet', '')[:100]}...")
    
    async def search_emails_native_mcp(self, query: str, max_results: int = 10):
        """Search emails using Gmail's native search via MCP"""
        logger.info(f"Native Gmail search for: '{query}'")
        
        result = await self.mcp_server.search_emails({
            "query": query,
            "max_results": max_results
        })
        
        if "Error" not in result.content[0].text:
            logger.info(f"Native search results: {result.content[0].text[:200]}...")
        else:
            logger.error(f"Native search failed: {result.content[0].text}")

async def main():
    """Example usage of Simple MCP Gmail RAG"""
    rag = SimpleMCPGmailRAG()
    
    try:
        # Initialize the system
        await rag.initialize()
        
        # Index some emails
        await rag.index_emails_mcp(max_results=20)
        
        # Search using embeddings
        rag.search_emails_mcp("investment portfolio", top_k=3)
        
        # Search using native Gmail search
        await rag.search_emails_native_mcp("is:important", max_results=5)
        
    except Exception as e:
        logger.error(f"Error in Simple MCP Gmail RAG: {e}")

if __name__ == "__main__":
    asyncio.run(main())
