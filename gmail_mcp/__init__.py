# MCP Gmail Integration Package

"""
MCP (Model Context Protocol) Gmail integration for RAG systems.

This package provides a simplified MCP implementation that works reliably
with Gmail APIs for email indexing and search functionality.
"""

from .simple_mcp_rag import SimpleMCPGmailRAG
from .mcp_gmail_server import GmailMCPServer

__all__ = ['SimpleMCPGmailRAG', 'GmailMCPServer']
