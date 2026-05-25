from langgraph.graph import StateGraph , START , END
from typing import TypedDict
from llm import llm
from langchain_core.prompts import PromptTemplate

class BlogState(TypedDict):
    topic:str
    outline:str
    blog:str

prompt1=PromptTemplate.from_template("You are a Outline writer for the topic {topic} , write in clear , precise way")
prompt2=PromptTemplate.from_template("You are a professional blog writer , write a blog for topic - {topic} , outline - {outline}")

def writeOutline(state:BlogState)->BlogState:
    topic=state['topic']
    res=llm.invoke(prompt1.format(topic=topic)).content
    state['outline'] = res
    return state    

def writeBlog(state:BlogState)->BlogState:
    topic=state['topic']
    outline=state['outline']
    res=llm.invoke(prompt2.format(topic=topic,outline=outline)).content
    state['blog'] = res
    return state    

graph=StateGraph(BlogState)

graph.add_node("outline",writeOutline)
graph.add_node("blog",writeBlog)

graph.add_edge(START,"outline")
graph.add_edge("outline","blog")
graph.add_edge("blog",END)

graph.compile()

res=workflow.invoke({'topic':"Minecraft"})
print(res['topic'])
print(res['outline'])
print(res['blog'])


