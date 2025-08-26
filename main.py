from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from todoist_api_python.api import TodoistAPI


# Load environment variables from a .env file
load_dotenv()

# Retrieve API keys from environment variables
todoist_api_key = os.getenv("TODOIST_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize Todoist API client with the Todoist API key
todoist = TodoistAPI(todoist_api_key)

# Define a tool to add a new task to the user's Todoist task list
@tool
def add_task(task, description=None):
    """ Add a new task to the users task list. Use this when the user want to add or create a task. """
    todoist.add_task(content=task, description=description)
    return f"Task '{task}' added successfully."

# Define a tool to retrieve and display the user's tasks
@tool
def show_tasks():
    """ Use this tool when the user wants to see their tasks. show the tasks in a bullet list"""
    results_paginator = todoist.get_tasks()
    tasks = []
    for task_list in results_paginator:
        for task in task_list:
            tasks.append(task.content)
    return tasks

# Define a tool to delete a task from Todoist by task content
@tool
def delete_task(task_content):
    """Delete a task from the user's task list by matching its content. Use this when the user wants to remove a task."""
    results_paginator = todoist.get_tasks()
    for task_list in results_paginator:
        for task in task_list:
            if task.content.lower() == task_content.lower():
                todoist.delete_task(task_id=task.id)
                return {"status": "success", "task": task_content}
    return {"status": "not_found", "task": task_content}


# Create a list of available tools for the agent
tools = [add_task, show_tasks, delete_task]

# Initialize the Gemini language model with specified API key and temperature
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key=gemini_api_key,
    temperature=0.3
)

# Define the system prompt to guide the assistant's behavior
system_prompt = (
    "You are a helpful, polite assistant. You can add tasks, show tasks in a bullet list, and delete tasks. "
    "Ensure tasks are shown only in a bullet list format (e.g., '- Task 1\n- Task 2'). "
    "For adding a task, respond with 'I added the task [task name] for you.' "
    "For deleting a task, respond with 'I deleted the task [task name] for you.' if successful, "
    "or 'I couldn’t find the task [task name].' if not found. "
    "For other queries, respond concisely and appropriately based on the tool output or user input."
)

# Create a chat prompt template with system prompt, conversation history, user input, and agent scratchpad
prompt = ChatPromptTemplate([
    ("system", system_prompt),
    MessagesPlaceholder("history"),
    ("user", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),

])

#chain = prompt | llm | StrOutputParser()
# Create an agent that uses the language model, tools, and prompt
agent = create_openai_tools_agent(llm, tools, prompt)
# Initialize the agent executor to handle tool execution and responses
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

#response = chain.invoke({"input": user_input})

# Initialize an empty list to store conversation history
history = []

# Start an infinite loop to continuously accept user input
while True:
    user_input = input("You: ")
    response = agent_executor.invoke({"input": user_input, "history": history})
    print(response['output'])
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response['output']))


