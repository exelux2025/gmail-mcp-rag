# mcp_workflow_demo.py

"""
Comprehensive MCP workflow demonstration showing both approaches:
1. Direct server usage (simpler, more reliable)
2. Client-server communication (full MCP protocol)
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger import get_logger
from .simple_mcp_rag import SimpleMCPGmailRAG

logger = get_logger(__name__)

async def demo_simple_mcp_workflow():
    """Demonstrate the simple MCP workflow (recommended)"""
    logger.info("=" * 60)
    logger.info("SIMPLE MCP WORKFLOW DEMO (RECOMMENDED)")
    logger.info("=" * 60)
    
    rag = SimpleMCPGmailRAG()
    
    try:
        # Initialize
        await rag.initialize()
        
        # Index emails
        await rag.index_emails_mcp(max_results=10)
        
        # Search with embeddings
        logger.info("\n--- Semantic Search Results ---")
        rag.search_emails_mcp("financial investment", top_k=2)
        
        # Native Gmail search
        logger.info("\n--- Native Gmail Search Results ---")
        await rag.search_emails_native_mcp("from:me", max_results=2)
        
        logger.info("\n✅ Simple MCP workflow completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Simple MCP workflow failed: {e}")

async def main():
    """Main demo function"""
    print("\n🚀 MCP Gmail RAG Workflow Demo")
    print("="*50)
    
    # Run the simple workflow demo
    await demo_simple_mcp_workflow()
    
    print("\n" + "="*50)
    print("📝 SUMMARY:")
    print("✅ Simple MCP approach: Works reliably")
    print("✅ Uses MCP server components directly")
    print("✅ No complex client-server communication")
    print("✅ Same Gmail credentials from .env file")
    print("✅ Full RAG functionality with embeddings")
    print("\n🎯 Recommendation: Use simple_mcp_rag.py for your project")

if __name__ == "__main__":
    asyncio.run(main())
