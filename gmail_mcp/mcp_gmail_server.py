# mcp_gmail_server.py

import asyncio
import json
import base64
from typing import Any, Dict, List, Optional
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
)
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gmail API setup
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

class GmailMCPServer:
    def __init__(self):
        self.server = Server("gmail-mcp")
        self.gmail_service = None
        self.setup_handlers()
    
    def setup_handlers(self):
        @self.server.list_tools()
        async def handle_list_tools() -> ListToolsResult:
            return ListToolsResult(
                tools=[
                    Tool(
                        name="authenticate_gmail",
                        description="Authenticate with Gmail API using OAuth2",
                        inputSchema={
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    ),
                    Tool(
                        name="fetch_emails",
                        description="Fetch emails from Gmail with optional filters",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "max_results": {
                                    "type": "integer",
                                    "description": "Maximum number of emails to fetch",
                                    "default": 100
                                },
                                "query": {
                                    "type": "string",
                                    "description": "Gmail search query (e.g., 'from:example@gmail.com')",
                                    "default": ""
                                },
                                "include_body": {
                                    "type": "boolean",
                                    "description": "Whether to include email body content",
                                    "default": True
                                }
                            },
                            "required": []
                        }
                    ),
                    Tool(
                        name="search_emails",
                        description="Search emails using Gmail's native search",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query",
                                    "default": ""
                                },
                                "max_results": {
                                    "type": "integer",
                                    "description": "Maximum number of results",
                                    "default": 10
                                }
                            },
                            "required": ["query"]
                        }
                    ),
                    Tool(
                        name="get_email_details",
                        description="Get detailed information about a specific email",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "email_id": {
                                    "type": "string",
                                    "description": "Gmail message ID"
                                }
                            },
                            "required": ["email_id"]
                        }
                    ),
                    Tool(
                        name="list_labels",
                        description="List all Gmail labels",
                        inputSchema={
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    )
                ]
            )
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            try:
                if name == "authenticate_gmail":
                    return await self.authenticate_gmail()
                elif name == "fetch_emails":
                    return await self.fetch_emails(arguments)
                elif name == "search_emails":
                    return await self.search_emails(arguments)
                elif name == "get_email_details":
                    return await self.get_email_details(arguments)
                elif name == "list_labels":
                    return await self.list_labels()
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {str(e)}")]
                )
    
    async def authenticate_gmail(self) -> CallToolResult:
        """Authenticate with Gmail API"""
        try:
            flow = InstalledAppFlow.from_client_config(
                {
                    "installed": {
                        "client_id": os.environ["GOOGLE_CLIENT_ID"],
                        "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token"
                    }
                },
                SCOPES
            )
            
            creds = flow.run_local_server(port=0)
            self.gmail_service = build('gmail', 'v1', credentials=creds)
            
            return CallToolResult(
                content=[TextContent(type="text", text="Successfully authenticated with Gmail API")]
            )
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Authentication failed: {str(e)}")]
            )
    
    async def fetch_emails(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Fetch emails from Gmail"""
        if not self.gmail_service:
            return CallToolResult(
                content=[TextContent(type="text", text="Please authenticate with Gmail first")]
            )
        
        try:
            max_results = arguments.get("max_results", 100)
            query = arguments.get("query", "")
            include_body = arguments.get("include_body", True)
            
            # Build the request
            request = self.gmail_service.users().messages().list(
                userId='me',
                maxResults=max_results
            )
            
            if query:
                request = self.gmail_service.users().messages().list(
                    userId='me',
                    maxResults=max_results,
                    q=query
                )
            
            response = request.execute()
            messages = response.get('messages', [])
            
            emails = []
            for msg in messages:
                email_data = {
                    "id": msg['id'],
                    "threadId": msg['threadId']
                }
                
                if include_body:
                    # Get full message details
                    full_msg = self.gmail_service.users().messages().get(
                        userId='me', 
                        id=msg['id'], 
                        format='full'
                    ).execute()
                    
                    # Extract headers
                    headers = full_msg['payload']['headers']
                    email_data.update({
                        "subject": next((h['value'] for h in headers if h['name'] == 'Subject'), ''),
                        "from": next((h['value'] for h in headers if h['name'] == 'From'), ''),
                        "date": next((h['value'] for h in headers if h['name'] == 'Date'), ''),
                        "snippet": full_msg.get('snippet', '')
                    })
                    
                    # Extract body if available
                    if 'parts' in full_msg['payload']:
                        for part in full_msg['payload']['parts']:
                            if part.get('mimeType') == 'text/plain':
                                if 'data' in part['body']:
                                    email_data['body'] = base64.urlsafe_b64decode(
                                        part['body']['data']
                                    ).decode('utf-8')
                                    break
                    elif full_msg['payload'].get('mimeType') == 'text/plain':
                        if 'data' in full_msg['payload']['body']:
                            email_data['body'] = base64.urlsafe_b64decode(
                                full_msg['payload']['body']['data']
                            ).decode('utf-8')
                
                emails.append(email_data)
            
            return CallToolResult(
                content=[TextContent(
                    type="text", 
                    text=f"Fetched {len(emails)} emails:\n\n" + 
                         json.dumps(emails, indent=2, default=str)
                )]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error fetching emails: {str(e)}")]
            )
    
    async def search_emails(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Search emails using Gmail's native search"""
        if not self.gmail_service:
            return CallToolResult(
                content=[TextContent(type="text", text="Please authenticate with Gmail first")]
            )
        
        try:
            query = arguments["query"]
            max_results = arguments.get("max_results", 10)
            
            response = self.gmail_service.users().messages().list(
                userId='me',
                maxResults=max_results,
                q=query
            ).execute()
            
            messages = response.get('messages', [])
            
            search_results = []
            for msg in messages:
                full_msg = self.gmail_service.users().messages().get(
                    userId='me', 
                    id=msg['id'], 
                    format='full'
                ).execute()
                
                headers = full_msg['payload']['headers']
                search_results.append({
                    "id": msg['id'],
                    "subject": next((h['value'] for h in headers if h['name'] == 'Subject'), ''),
                    "from": next((h['value'] for h in headers if h['name'] == 'From'), ''),
                    "date": next((h['value'] for h in headers if h['name'] == 'Date'), ''),
                    "snippet": full_msg.get('snippet', '')
                })
            
            return CallToolResult(
                content=[TextContent(
                    type="text", 
                    text=f"Search results for '{query}':\n\n" + 
                         json.dumps(search_results, indent=2, default=str)
                )]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error searching emails: {str(e)}")]
            )
    
    async def get_email_details(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Get detailed information about a specific email"""
        if not self.gmail_service:
            return CallToolResult(
                content=[TextContent(type="text", text="Please authenticate with Gmail first")]
            )
        
        try:
            email_id = arguments["email_id"]
            
            full_msg = self.gmail_service.users().messages().get(
                userId='me', 
                id=email_id, 
                format='full'
            ).execute()
            
            headers = full_msg['payload']['headers']
            email_details = {
                "id": email_id,
                "subject": next((h['value'] for h in headers if h['name'] == 'Subject'), ''),
                "from": next((h['value'] for h in headers if h['name'] == 'From'), ''),
                "to": next((h['value'] for h in headers if h['name'] == 'To'), ''),
                "date": next((h['value'] for h in headers if h['name'] == 'Date'), ''),
                "snippet": full_msg.get('snippet', ''),
                "labels": full_msg.get('labelIds', [])
            }
            
            # Extract body
            if 'parts' in full_msg['payload']:
                for part in full_msg['payload']['parts']:
                    if part.get('mimeType') == 'text/plain':
                        if 'data' in part['body']:
                            email_details['body'] = base64.urlsafe_b64decode(
                                part['body']['data']
                            ).decode('utf-8')
                            break
            elif full_msg['payload'].get('mimeType') == 'text/plain':
                if 'data' in full_msg['payload']['body']:
                    email_details['body'] = base64.urlsafe_b64decode(
                        full_msg['payload']['body']['data']
                    ).decode('utf-8')
            
            return CallToolResult(
                content=[TextContent(
                    type="text", 
                    text=f"Email details:\n\n" + 
                         json.dumps(email_details, indent=2, default=str)
                )]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error getting email details: {str(e)}")]
            )
    
    async def list_labels(self) -> CallToolResult:
        """List all Gmail labels"""
        if not self.gmail_service:
            return CallToolResult(
                content=[TextContent(type="text", text="Please authenticate with Gmail first")]
            )
        
        try:
            response = self.gmail_service.users().labels().list(userId='me').execute()
            labels = response.get('labels', [])
            
            return CallToolResult(
                content=[TextContent(
                    type="text", 
                    text=f"Gmail Labels:\n\n" + 
                         json.dumps(labels, indent=2, default=str)
                )]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error listing labels: {str(e)}")]
            )

async def main():
    server = GmailMCPServer()
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="gmail-mcp",
                server_version="1.0.0",
                capabilities=server.server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main()) 