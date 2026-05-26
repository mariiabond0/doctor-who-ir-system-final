"""
RAG (Retrieval Augmented Generation) module for Doctor Who IR system.
Combines retrieval with Ollama LLM for generating contextual answers.
"""

import logging
import json
import requests
import sqlite3
from typing import Dict, List, Optional
import config
from src.boolean_search import boolean_search_sqlite

logger = logging.getLogger(__name__)


def check_ollama_health() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get(
            f"{config.OLLAMA_BASE_URL}/api/tags",
            timeout=2
        )
        return response.status_code == 200
    except requests.ConnectionError:
        return False
    except Exception as e:
        logger.warning(f"Ollama health check failed: {e}")
        return False


def retrieve_context(query: str, top_k: int = None) -> tuple[List[Dict], str]:
    """
    Retrieve top-K documents using Boolean Search and format as context.
    
    Args:
        query: User query string
        top_k: Number of documents to retrieve (default: config.RAG_CONTEXT_SIZE)
    
    Returns:
        Tuple of (results list, formatted context string)
    """
    if top_k is None:
        top_k = config.RAG_CONTEXT_SIZE
    
    try:
        conn = sqlite3.connect(str(config.DB_PATH))
        results = boolean_search_sqlite(query, conn, top_n=top_k)
        conn.close()
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return [], "Error retrieving documents."
    
    if not results:
        return [], "No relevant documents found."
    
    context_parts = []
    for i, result in enumerate(results, 1):
        context_parts.append(
            f"Document {i}: {result}\n"
        )
    
    context = "\n---\n".join(context_parts)
    return [({"title": r, "score": 1.0}) for r in results], context


def build_prompt(query: str, context: str) -> str:
    """Build a prompt for the LLM with retrieval context."""
    prompt = f"""You are a knowledgeable assistant about Doctor Who episodes.
    
Based on the following Doctor Who episode information, answer the user's question accurately and concisely.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    return prompt


def query_ollama(prompt: str, temperature: float = None, max_tokens: int = None) -> Optional[str]:
    """
    Query Ollama LLM for answer generation.
    
    Args:
        prompt: Full prompt with context and question
        temperature: Generation temperature (default: config.RAG_TEMPERATURE)
        max_tokens: Maximum tokens in response (default: config.RAG_MAX_TOKENS)
    
    Returns:
        Generated answer string or None if error
    """
    if temperature is None:
        temperature = config.RAG_TEMPERATURE
    if max_tokens is None:
        max_tokens = config.RAG_MAX_TOKENS
    
    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            logger.error(f"Ollama API error: {response.status_code}")
            return None
            
    except requests.Timeout:
        logger.error("Ollama request timed out (30s)")
        return None
    except requests.ConnectionError:
        logger.error(f"Failed to connect to Ollama at {config.OLLAMA_BASE_URL}")
        return None
    except Exception as e:
        logger.error(f"Ollama query failed: {e}")
        return None


def rag_query(query: str) -> Dict:
    """
    Perform RAG query: retrieve documents and generate answer.
    
    Args:
        query: User query string
    
    Returns:
        Dictionary with:
            - 'query': Original query
            - 'retrieved_docs': List of retrieved document results
            - 'context': Formatted context string
            - 'answer': Generated answer from LLM
            - 'error': Error message if any
            - 'source': 'rag'
    """
    if not check_ollama_health():
        return {
            'query': query,
            'retrieved_docs': [],
            'context': '',
            'answer': '',
            'error': (
                f"Ollama not running. Start it with: ollama serve\n"
                f"Pull model: ollama pull {config.OLLAMA_MODEL}"
            ),
            'source': 'rag'
        }
    
    try:
        # Step 1: Retrieve context
        retrieved_docs, context = retrieve_context(query)
        
        if not context or context == "No relevant documents found.":
            return {
                'query': query,
                'retrieved_docs': [],
                'context': context,
                'answer': 'No relevant Doctor Who episodes found for this query.',
                'error': None,
                'source': 'rag'
            }
        
        # Step 2: Build prompt
        prompt = build_prompt(query, context)
        
        # Step 3: Generate answer
        answer = query_ollama(prompt)
        
        if not answer:
            return {
                'query': query,
                'retrieved_docs': retrieved_docs,
                'context': context,
                'answer': '',
                'error': 'Failed to generate answer from Ollama',
                'source': 'rag'
            }
        
        return {
            'query': query,
            'retrieved_docs': retrieved_docs,
            'context': context,
            'answer': answer,
            'error': None,
            'source': 'rag'
        }
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        return {
            'query': query,
            'retrieved_docs': [],
            'context': '',
            'answer': '',
            'error': str(e),
            'source': 'rag'
        }


def format_rag_output(rag_result: Dict) -> str:
    """Format RAG result for display."""
    output = []
    output.append(f"\n{'='*80}")
    output.append(f"RAG ANSWER")
    output.append(f"{'='*80}\n")
    
    if rag_result.get('error'):
        output.append(f"❌ Error: {rag_result['error']}\n")
        output.append("Setup instructions:")
        output.append("1. Install Ollama from https://ollama.ai")
        output.append("2. Run: ollama pull llama2")
        output.append("3. Run: ollama serve")
        return "\n".join(output)
    
    output.append(f"Question: {rag_result['query']}\n")
    output.append(f"Retrieved {len(rag_result['retrieved_docs'])} documents:\n")
    
    for i, doc in enumerate(rag_result['retrieved_docs'], 1):
        output.append(f"  {i}. {doc.get('title', 'Unknown')} (score: {doc.get('score', 0):.3f})")
    
    output.append(f"\n{'─'*80}")
    output.append("Generated Answer:")
    output.append(f"{'─'*80}\n")
    output.append(rag_result['answer'])
    output.append(f"\n{'='*80}\n")
    
    return "\n".join(output)
