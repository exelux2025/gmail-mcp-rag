# workflow.py

import os
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from pydantic import SecretStr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# State definition
class GraphState(TypedDict):
    query: str
    search_results: List[Dict[str, Any]]
    filtered_results: List[Dict[str, Any]]
    final_answer: str
    error: str

def filter_emails_node(state: GraphState) -> GraphState:
    """Filter and rank the search results based on relevance to the query"""
    try:
        query = state["query"]
        search_results = state["search_results"]
        
        if not search_results:
            state["filtered_results"] = []
            state["error"] = "No search results found"
            return state
        
        # Create a prompt to filter and rank emails
        filter_prompt = f"""
        Given the user query: "{query}"
        
        Below are search results from an email database. Please filter and rank them by relevance to the query.
        For each result, provide a relevance score (0-10) and a brief reason.
        
        Search Results:
        {chr(10).join([f"{i+1}. ID: {result['id']} | Score: {result['score']:.4f} | Content: {result['snippet'][:200]}..." for i, result in enumerate(search_results)])}
        
        Please return a JSON array with filtered results in this format:
        [
            {{
                "id": "email_id",
                "relevance_score": 8,
                "reason": "Brief explanation of why this email is relevant",
                "original_score": 0.5828,
                "snippet": "email snippet"
            }}
        ]
        
        Only include emails that are actually relevant to the query (relevance_score >= 5).
        """
        
        # Use OpenAI to filter results
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
            
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            api_key=SecretStr(api_key)
        )
        
        response = llm.invoke([HumanMessage(content=filter_prompt)])
        
        # Parse the response (simplified - in production you'd want better JSON parsing)
        try:
            import json
            content_str = str(response.content)
            filtered_data = json.loads(content_str)
            state["filtered_results"] = filtered_data
        except:
            # Fallback: keep top 3 results
            state["filtered_results"] = search_results[:3]
        
        return state
        
    except Exception as e:
        state["error"] = f"Error in filter_emails_node: {str(e)}"
        state["filtered_results"] = search_results[:3] if search_results else []
        return state

def formulate_answer_node(state: GraphState) -> GraphState:
    """Generate a comprehensive answer based on the filtered email results"""
    try:
        query = state["query"]
        filtered_results = state["filtered_results"]
        
        if not filtered_results:
            state["final_answer"] = "I couldn't find any relevant emails matching your query."
            return state
        
        # Create a prompt to generate a comprehensive answer
        answer_prompt = f"""
        User Query: "{query}"
        
        Based on the following relevant emails, provide a comprehensive and helpful answer to the user's query.
        If the emails contain specific information that answers the query, include that information.
        If the emails are related but don't directly answer the query, explain what information is available.
        
        Relevant Emails:
        {chr(10).join([f"Email {i+1} (Relevance: {result.get('relevance_score', 'N/A')}): {result.get('snippet', result.get('snippet', ''))}" for i, result in enumerate(filtered_results)])}
        
        Please provide a clear, well-structured answer that:
        1. Directly addresses the user's query
        2. References specific information from the emails when relevant
        3. Acknowledges if the information is limited or incomplete
        4. Is helpful and actionable
        
        Answer:
        """
        
        # Use OpenAI to generate the answer
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
            
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            api_key=SecretStr(api_key)
        )
        
        response = llm.invoke([HumanMessage(content=answer_prompt)])
        state["final_answer"] = str(response.content)
        
        return state
        
    except Exception as e:
        state["error"] = f"Error in formulate_answer_node: {str(e)}"
        state["final_answer"] = "Sorry, I encountered an error while processing your query."
        return state

def create_workflow():
    """Create the LangGraph workflow"""
    
    # Create the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("filter_emails", filter_emails_node)
    workflow.add_node("formulate_answer", formulate_answer_node)
    
    # Set the entry point
    workflow.set_entry_point("filter_emails")
    
    # Add edges
    workflow.add_edge("filter_emails", "formulate_answer")
    workflow.add_edge("formulate_answer", END)
    
    # Compile the graph
    return workflow.compile()

# Initialize the workflow
workflow = create_workflow() 