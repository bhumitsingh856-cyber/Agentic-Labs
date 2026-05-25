from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA 
from dotenv import load_dotenv
import os
load_dotenv()

# nvidia/nemotron-3-nano-omni-30b-a3b-reasoning

# llm=ChatNVIDIA(api_key=os.getenv("NVIDIA_API_KEY"),model="meta/llama-3.1-8b-instruct") 
llm=ChatGroq(model="llama-3.3-70b-versatile")
