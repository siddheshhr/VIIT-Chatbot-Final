import os
import torch
from fastapi import FastAPI, Request, HTTPException
from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Step 1: Initialize FastAPI
app = FastAPI()

# Step 2: HuggingFace Model and Tokenizer Setup
MODEL_NAME = "AnishaShende/tinyllama-unsloth-merged_1"
MONGODB_URI = "mongodb+srv://anisha22320184:YT0nSJqiOneznlFW@cluster0.mxq6c.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Secure environment variables
os.environ["HF_TOKEN"] = "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"

# Initialize model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# MongoDB Connection
try:
    client = MongoClient(MONGODB_URI)
    db = client["scrape_data"]
    collection = db["acade_calen"]
except Exception as e:
    logger.error(f"MongoDB connection error: {e}")
    raise

# Initialize Embeddings and Vector Store
embedding_model = HuggingFaceEmbeddings(model_name="baai/bge-large-en-v1.5")
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embedding_model,
    index_name="vector_index",
    embedding_key="embeddings",
    text_key="answer",
)

def preprocess_query(query: str) -> str:
    """
    Preprocess the query to improve matching accuracy.
    """
    # Convert to lowercase
    query = query.lower()
    
    # Remove common stop words and punctuation
    import string
    import re
    
    # Remove punctuation
    query = query.translate(str.maketrans('', '', string.punctuation))
    
    # Remove common stop words
    stop_words = set(['what', 'was', 'were', 'is', 'are', 'tell', 'me', 'about', 'can', 'you', 'explain'])
    query_words = [word for word in query.split() if word not in stop_words]
    
    return ' '.join(query_words)

def retrieve_relevant_context(query: str, top_k: int = 5) -> List[str]:
    """
    Advanced context retrieval with multiple matching strategies.
    """
    try:
        # Preprocess the query
        processed_query = preprocess_query(query)
        
        # Retrieve documents using similarity search
        relevant_docs = vector_store.similarity_search(processed_query, k=top_k)
        
        # Extract and process contexts
        contexts = [doc.page_content for doc in relevant_docs]
        
        # Advanced context scoring
        def context_score(context):
            # Calculate relevance based on keyword matching
            processed_context = preprocess_query(context)
            
            # Count keyword matches
            keyword_matches = sum(
                keyword in processed_context 
                for keyword in processed_query.split()
            )
            
            # Additional scoring criteria
            length_penalty = min(1, len(context) / 100)  # Favor more informative contexts
            
            return keyword_matches * length_penalty
        
        # Sort contexts by relevance score
        scored_contexts = sorted(
            contexts, 
            key=context_score, 
            reverse=True
        )
        
        return scored_contexts
    
    except Exception as e:
        logger.error(f"Context retrieval error: {e}")
        return []

def llm_inference(question: str, contexts: List[str], max_length: int = 300) -> str:
    """
    Advanced inference with multiple response generation strategies.
    """
    # Direct context matching first
    def find_direct_match(question, contexts):
        # Preprocess question
        processed_question = preprocess_query(question)
        
        # Find contexts with the highest keyword match
        matching_contexts = [
            context for context in contexts
            if all(
                keyword in preprocess_query(context) 
                for keyword in processed_question.split()
            )
        ]
        
        return matching_contexts[0] if matching_contexts else None
    
    # Try direct matching first
    direct_match = find_direct_match(question, contexts)
    if direct_match:
        return direct_match.strip()
    
    # Prepare context for generation
    combined_context = "\n\n".join(contexts) if contexts else "No specific context available."
    
    # Enhanced prompt with multiple reasoning strategies
    input_prompt = f"""<s>[INST]
    Instruction: 
    - Carefully analyze the given context
    - Extract the most relevant and precise information
    - If a direct answer exists, use it verbatim
    - Prioritize clarity, accuracy, and completeness
    - If no specific information is found, clearly state that

    Context:
    {combined_context}

    Question: {question}
    [/INST]
    """
    
    try:
        # Tokenize and generate response
        inputs = tokenizer(input_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_return_sequences=1,
                num_beams=4,
                repetition_penalty=1.3,
                temperature=0.3,
                top_k=50,
                top_p=0.9,
                do_sample=True
            )
        
        # Decode and clean the output
        raw_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract the response
        answer = raw_output.split("[/INST]")[-1].strip()
        
        # Fallback mechanisms
        if not answer or len(answer) < 10:
            # If generated response is too short, use the first context
            return contexts[0] if contexts else "No information found."
        
        return answer
    
    except Exception as e:
        logger.error(f"LLM inference error: {e}")
        return "Unable to generate a response based on the available context."

# Modify the chat endpoint to use these functions
@app.post("/chat")
async def get_response(request: Request):
    """
    Enhanced chat endpoint with robust context retrieval and response generation.
    """
    try:
        data = await request.json()
        user_input = data.get("input_text")
        
        if not user_input:
            raise HTTPException(status_code=400, detail="No input text provided")
        
        # Retrieve relevant contexts
        contexts = retrieve_relevant_context(user_input)
        
        # Generate response with context
        answer = llm_inference(user_input, contexts)
        
        return {
            "input": user_input, 
            "response": answer, 
            "contexts_used": contexts[:3]  # Limit returned contexts for clarity
        }
    
    except HTTPException as http_err:
        logger.error(f"HTTP Error: {http_err}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"message": "Chatbot is running and ready to assist!"}

# Optional: Add shutdown hook to close MongoDB connection
@app.on_event("shutdown")
def shutdown_event():
    client.close()
    logger.info("Application shutdown. MongoDB connection closed.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
