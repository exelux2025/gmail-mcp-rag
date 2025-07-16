# main.py

import os
import base64
from dotenv import load_dotenv

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from logger import get_logger, set_log_level

# Initialize logger
logger = get_logger(__name__)

def get_gmail_service():
    logger.info("Initializing Gmail service...")
    try:
        SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
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
        logger.debug("Starting OAuth flow...")
        creds = flow.run_local_server(port=0)
        logger.info("OAuth authentication successful")
        service = build('gmail', 'v1', credentials=creds)
        logger.info("Gmail service initialized successfully")
        return service
    except Exception as e:
        logger.error(f"Failed to initialize Gmail service: {e}")
        raise

def fetch_last_n_message_ids(service, n=100):
    logger.info(f"Fetching last {n} message IDs from Gmail...")
    try:
        resp = service.users().messages().list(userId='me', maxResults=n).execute()
        message_ids = [m['id'] for m in resp.get('messages', [])]
        logger.info(f"Successfully fetched {len(message_ids)} message IDs")
        return message_ids
    except Exception as e:
        logger.error(f"Failed to fetch message IDs: {e}")
        raise

def get_existing_message_ids(collection):
    """Get message IDs that are already indexed in the collection"""
    try:
        # Get all existing IDs from the collection
        result = collection.get(include=['metadatas'])
        existing_ids = result['ids'] if result['ids'] else []
        logger.info(f"Found {len(existing_ids)} existing indexed emails")
        return set(existing_ids)
    except Exception as e:
        logger.warning(f"Could not retrieve existing message IDs: {e}")
        return set()

def extract_text(service, msg_id):
    logger.debug(f"Extracting text from message ID: {msg_id}")
    try:
        msg = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
        parts = msg.get('payload', {}).get('parts', [])
        text = ""
        for p in parts:
            if p.get('mimeType') == 'text/plain' and 'data' in p.get('body', {}):
                data = p['body']['data']
                text += base64.urlsafe_b64decode(data).decode('utf-8')
        
        final_text = text or msg.get('snippet', '')
        logger.debug(f"Extracted {len(final_text)} characters from message {msg_id}")
        return final_text
    except Exception as e:
        logger.error(f"Failed to extract text from message {msg_id}: {e}")
        return ""

def index_emails(service, embedder, collection, db_dir, n=100):
    logger.info(f"Starting to index {n} emails...")
    try:
        # Get existing indexed message IDs
        existing_ids = get_existing_message_ids(collection)
        
        # Fetch all message IDs from Gmail
        all_ids = fetch_last_n_message_ids(service, n)
        
        # Filter out already indexed messages
        new_ids = [mid for mid in all_ids if mid not in existing_ids]
        
        if not new_ids:
            logger.info("No new emails to index - all emails are already indexed")
            return
        
        logger.info(f"Found {len(new_ids)} new emails to index out of {len(all_ids)} total")
        indexed_count = 0
        
        for i, mid in enumerate(new_ids, 1):
            logger.debug(f"Processing new email {i}/{len(new_ids)}: {mid}")
            txt = extract_text(service, mid)
            
            if not txt.strip():
                logger.warning(f"Skipping message {mid} - no text content")
                continue
                
            try:
                emb = embedder.encode(txt, show_progress_bar=False).tolist()
                collection.upsert(
                    ids=[mid],
                    embeddings=[emb],
                    metadatas=[{"snippet": txt[:200]}]
                )
                indexed_count += 1
                logger.debug(f"Successfully indexed message {mid}")
            except Exception as e:
                logger.error(f"Failed to index message {mid}: {e}")
                continue
        
        logger.info(f"Indexing complete: {indexed_count}/{len(new_ids)} new emails successfully indexed")
        logger.info(f"Chroma database location: {db_dir}")
    except Exception as e:
        logger.error(f"Failed to index emails: {e}")
        raise

def semantic_search(embedder, collection, query, top_k=5):
    logger.info(f"Performing semantic search for query: '{query}' (top_k={top_k})")
    try:
        q_emb = embedder.encode(query).tolist()
        results = collection.query(query_embeddings=[q_emb], n_results=top_k)
        
        logger.info(f"Found {len(results['ids'][0])} results")
        for i, (_id, meta, dist) in enumerate(zip(results["ids"][0], results["metadatas"][0], results["distances"][0]), 1):
            logger.info(f"Result {i}: {_id} (score {dist:.4f}): {meta['snippet']}…")
            print(f"→ {_id} (score {dist:.4f}): {meta['snippet']}…")
    except Exception as e:
        logger.error(f"Failed to perform semantic search: {e}")
        print(f"Error performing search: {e}")

def main():
    logger.info("Starting Gmail RAG application...")
    
    try:
        # Load environment variables
        logger.info("Loading environment variables...")
        load_dotenv()
        
        # Check required environment variables
        required_vars = ["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        logger.info("Environment variables loaded successfully")

        # 1. Gmail auth
        service = get_gmail_service()

        # 2. Embedding model
        MODEL_NAME = "BAAI/bge-small-en-v1.5"
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        try:
            embedder = SentenceTransformer(MODEL_NAME)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        # 3. Chroma setup
        db_dir = os.getenv("CHROMA_DB_DIR", "./chromadb")
        logger.info(f"Initializing ChromaDB at: {db_dir}")
        try:
            client = chromadb.PersistentClient(path=db_dir, settings=Settings())
            collection = client.get_or_create_collection(name="gmail_rag")
            logger.info("ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

        # 4. Index
        index_emails(service, embedder, collection, db_dir, n=100)

        # 5. Query loop
        logger.info("Starting interactive query loop...")
        while True:
            try:
                q = input("\nEnter search query (or 'exit'): ").strip()
                if q.lower() in ("exit", "quit"):
                    logger.info("User requested exit")
                    break
                if not q:
                    logger.warning("Empty query received, skipping...")
                    continue
                    
                semantic_search(embedder, collection, q, top_k=5)
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, exiting...")
                break
            except Exception as e:
                logger.error(f"Error in query loop: {e}")
                print(f"Error: {e}")
                
    except Exception as e:
        logger.error(f"Application failed: {e}")
        raise
    finally:
        logger.info("Gmail RAG application shutdown complete")

if __name__ == "__main__":
    main()
