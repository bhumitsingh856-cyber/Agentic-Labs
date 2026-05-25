from langgraph.graph import StateGraph,START,END
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langgraph.types import Send
from typing import TypedDict,Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode,tools_condition
from pydantic import BaseModel , Field
from image_generation_runnable import image_generation
from llm import llm
import operator

class TaskSchema(BaseModel):
    id:str
    title:str
    brief:str = Field(...,description="Brief description of task")
class PlannerSchema(BaseModel):
    blog_topic:str=Field(description="Topic of the blog")
    tasks: list[TaskSchema]
class BlogState(TypedDict):
    topic:str
    plan: PlannerSchema
    sections:Annotated[list[str],operator.add]
    blog:str

# @tool
# def generate_image(prompt: str):
#     """Generate images from a prompt"""
   


def planner(state:BlogState):
    print("Started planning...\n")
    plan=llm.with_structured_output(PlannerSchema).invoke([
        SystemMessage(content=(
            "You are an expert blog content planner and SEO strategist. Your job is to create a highly detailed, "
            "logically structured, and comprehensive blog plan for the provided topic. The plan must consist of "
            "atleast 40 distinct sections.\n\n"
            "For each section, provide:\n"
            "- A compelling, click-worthy title.\n"
            "- A detailed and comprehensive brief explaining exactly what sub-topics, key terms, concepts, "
            "and insights need to be covered in that section. Ensure the brief is clear, structured, and deep enough "
            "so that a writer can produce high-quality, comprehensive content without overlap or redundancy with other sections.\n\n"
            "The sections must progress logically, starting with a powerful introduction, moving through core concepts, "
            "methodologies/technologies, applications, challenges, future outlook, and ending with a strong conclusion."
        )),
        HumanMessage(content=f"create a detailed blog plan for this topic {state['topic']}")
    ])
    return {"plan":plan}

def task_assigner(state:BlogState):
    print("Assigning tasks to workers...\n")
    return [Send("worker",{"task":task,"topic":state["topic"]}) for task in state['plan'].tasks]

def worker(work:dict):
    topic=work['topic']
    task=work["task"]
    print(f"Writing section: {task.title}\n")
    res=llm.invoke([
        SystemMessage(content=(
            "You are a world-class, professional technical writer and subject matter expert. Your objective is to write "
            "a highly comprehensive, publication-grade blog section based on the provided main topic, section title, and "
            "detailed section brief.\n\n"
            "Follow these strict Writing Guidelines:\n\n"
            "1. STRUCTURAL DEPTH & DETAIL:\n"
            "   - Dive deep into technical details, methodologies, and architectural concepts. Avoid superficial overviews, "
            "fluff, or generic advice.\n"
            "   - Provide concrete, real-world examples, precise analogies, and step-by-step breakdowns.\n"
            "   - Write in a highly informative, educational, and authoritative voice.\n\n"
            "2. FORMATTING WITH MARKDOWN:\n"
            "   - Use clean, well-structured Markdown. Do NOT start the output with a single top-level heading (#). Use nested "
            "headings such as '###' or '####' for sub-sections.\n"
            "   - Incorporate lists, tables, bold text for key terms, and blockquotes for critical callouts to make the content "
            "highly readable and scannable.\n\n"
            "3. HIGH-QUALITY CODE SNIPPETS (IF APPLICABLE):\n"
            "   - If the section refers to programmatic concepts, write complete, correct, and robust code snippets with proper "
            "syntax highlighting (e.g., ```python, ```javascript).\n"
            "   - Include inline comments explaining complex logic or crucial configuration parameters.\n\n"
            "4. NARRATIVE FLOW & CONTEXT:\n"
            "   - Keep in mind that this is a single, focused section of a larger collaborative blog post. Do NOT write introductory "
            "remarks (e.g., 'In this section, we will discuss...') or concluding wraps for the whole post.\n"
            "   - Focus solely on the scope of the brief provided, ensuring smooth transitions and absolute professional polish."
        )),
        HumanMessage(content=(
            f"Main Blog Topic: {topic}\n"
            f"Assigned Section Title: {task.title}\n"
            f"Assigned Section Brief: {task.brief}\n\n"
            "Please draft this section now, adhering strictly to the system guidelines and covering all aspects of the brief."
        ))
    ]).content
    print(f"Writing section {task.title} completed...\n")
    return {"sections":[res]}

def aggregator(state:BlogState):
    print("Aggregating sections...\n")
    blog = f"# {state['plan'].blog_topic}\n\n" + "\n\n".join(state['sections'])
    print("Aggregation completed...\n")
    return {"blog":blog}

graph=StateGraph(BlogState)
graph.add_node("planner",planner)
graph.add_node("task_assigner",task_assigner)
graph.add_node("worker",worker)
graph.add_node("aggregator",aggregator)

graph.add_edge(START,"planner")
graph.add_conditional_edges("planner",task_assigner,['worker'])
graph.add_edge("worker","aggregator")
graph.add_edge("aggregator",END)

workflow=graph.compile()
res=workflow.invoke({'topic':"Self attention"})
print("Blog created")

a=open('blog.md','w',encoding="utf-8")
a.write(res['blog'])
a.close()


