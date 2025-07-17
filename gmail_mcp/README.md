# Gmail MCP Integration

A simplified MCP (Model Context Protocol) implementation for Gmail RAG systems.

## Quick Start

### From Root Directory
```bash
# Run the demo
uv run python run_mcp_demo.py
```

### Programmatic Usage
```python
from gmail_mcp import SimpleMCPGmailRAG

rag = SimpleMCPGmailRAG()
await rag.initialize()
await rag.index_emails_mcp(max_results=50)
rag.search_emails_mcp("your search query")
```

## Files

- **`simple_mcp_rag.py`** - Main implementation
- **`mcp_gmail_server.py`** - MCP server components  
- **`mcp_workflow_demo.py`** - Demo script

## Features

✅ Gmail authentication  
✅ Email indexing with embeddings  
✅ Semantic search  
✅ Native Gmail search  
✅ ChromaDB storage  

Uses the same Google credentials from your `.env` file.
