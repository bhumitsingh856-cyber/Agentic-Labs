from langgraph.graph import   START , END , StateGraph 
from langchain_core.messages.utils import trim_messages , count_tokens_approximately 
from langchain.messages import RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from llm import llm
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage,BaseMessage,AIMessage
from typing import TypedDict,Annotated 

class MessagesState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    summary:str

def chat(state:MessagesState):
    # -----Short term memory using trimming

    # stm=trim_messages(
    #     state['messages'],
    #     strategy="last",
    #     token_counter=count_tokens_approximately,
    #     max_tokens=200
    # )
    
    # -----Short term memory using Summarizing
    msg=[]
    if(state.get('summary')):
        msg.append(AIMessage(content=f"SUmmary of previus conversation : {state['summary']}"))
    msg.extend(state['messages'])

    print("-----------------\n",msg,"\n------------")
    res= llm.invoke(msg).content
    return {"messages":[AIMessage(content=res)]}

def summarize(state:MessagesState):
    if(state.get('summary')):
        prompt=f"Existing summary : {state['summary']} , extend the summary with new conversation"
    else :
        prompt="Summarize the above conversation"

    msg=state['messages'][:4] 
    res=llm.invoke(msg+[HumanMessage(content=prompt)]).content
    print("Summary---------------")
    print(res)
    return {"messages": [RemoveMessage(id=i.id) for i in msg],"summary": res}
def condition(state:MessagesState):
    return len(state['messages'])>6

graph=StateGraph(MessagesState)
graph.add_node("chat",chat)
graph.add_node("summary",summarize)

graph.add_edge(START,"chat")
graph.add_conditional_edges("chat",condition,{True:'summary',False:END})
graph.add_edge("chat",END)
checkpoint=MemorySaver()
workflow=graph.compile(checkpointer=checkpoint)

while True:
    user=input("Enter prompt = ")
    print("User - ",user)
    if(user=="1"):
        break
    res=workflow.invoke({'messages':[HumanMessage(content=user)]},config={"configurable":{"thread_id":"1"}})
    print("AI - ",res['messages'][-1].text)