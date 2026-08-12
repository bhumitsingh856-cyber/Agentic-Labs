from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
import os
import asyncio

load_dotenv()

llm = ChatGroq(model="qwen/qwen3.6-27b", temperature=0.2)

async def main():
    client = MultiServerMCPClient({
            "math": {
                "transport": "stdio",
                "command": "python",
                "args": [
                    "C:\\Users\\BHUMIT SINGH\\Documents\\LangGraph\\MCP\\mcp_server.py"
                ],
            },
        })

    tools = await client.get_tools()
    prompt = await client.get_prompt("math", "calculator_guide")
    
    agent = create_agent(model=llm, tools=tools, system_prompt=prompt[0].content)
    res = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is (10 * 32 + 20) * 4 + 3"}]}
    ) 
    print("-" * 30)
    print(res["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
