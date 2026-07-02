from langchain_nvidia_ai_endpoints import ChatNVIDIA , NVIDIAEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()

# nvidia/nemotron-3-nano-omni-30b-a3b-reasoning

# llm=ChatNVIDIA(model="meta/llama-3.1-8b-instruct") 
llm=ChatGroq(model="llama-3.3-70b-versatile")

embedding_model = NVIDIAEmbeddings(
    model="nvidia/nv-embed-v1",
)

