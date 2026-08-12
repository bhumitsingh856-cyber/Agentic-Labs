from fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

@mcp.prompt()
def calculator_guide() -> str:
    """Guide for using the calculator"""
    return """
    You are a calculator assistant. Follow these steps:
    1. Ask the user what operation they want
    2. Get two numbers
    3. Use the add tool to calculate
    4. Explain the result clearly
    """


if __name__ == "__main__":
    mcp.run(transport="stdio")