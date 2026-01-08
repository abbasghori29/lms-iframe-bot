"""
Simple test script for the chat API
"""
import requests
import json

API_URL = "http://localhost:8005/api/v1/chat/chat"

def test_chat(question: str, chat_history=None):
    """Test the chat endpoint"""
    payload = {
        "question": question,
        "k": 5
    }
    
    if chat_history:
        payload["chat_history"] = [
            {"role": role, "content": content}
            for role, content in chat_history
        ]
    
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        
        result = response.json()
        print("\n" + "="*80)
        print(f"Question: {question}")
        print("="*80)
        print(f"\nAnswer:\n{result['answer']}\n")
        print(f"Sources used: {result['context_used']}")
        print("\nSources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source['source']} (Page {source['page']})")
            print(f"     Preview: {source['content_preview'][:100]}...")
        print()
        
        return result
    except Exception as e:
        print(f"Error: {e}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response.text}")
        return None


if __name__ == "__main__":
    print("Testing Chat API")
    print("Make sure the FastAPI server is running: uvicorn app.main:app --reload")
    print()
    
    # Test simple question
    test_chat("What is CSI?")
    
    # Test with follow-up
    test_chat(
        "What courses do they offer?",
        chat_history=[
            ("human", "What is CSI?"),
            ("ai", "CSI is a leading provider of professional training...")
        ]
    )
    
    # Test specific question
    test_chat("What are the certification requirements for CSI?")

