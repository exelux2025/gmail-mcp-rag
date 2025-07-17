# run_mcp_demo.py

"""
Simple launcher for the MCP Gmail RAG demo
"""

import asyncio
import sys
import os

# Add the gmail_mcp module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'gmail_mcp'))

from gmail_mcp.simple_mcp_rag import SimpleMCPGmailRAG
from logger import get_logger

logger = get_logger(__name__)

async def main():
    """Run the MCP Gmail RAG demo"""
    print("🚀 MCP Gmail RAG Demo")
    print("=" * 40)
    
    rag = SimpleMCPGmailRAG()
    
    try:
        # Initialize
        await rag.initialize()
        
        # Index emails
        await rag.index_emails_mcp(max_results=10)
        
        # Search with embeddings
        print("\n--- Semantic Search Results ---")
        rag.search_emails_mcp("financial investment", top_k=3)
        
        # Native Gmail search
        print("\n--- Native Gmail Search Results ---")
        await rag.search_emails_native_mcp("from:me", max_results=3)
        
        print("\n✅ MCP Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ MCP Demo failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
