import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler


# Project description
# This project is a Streamlit chatbot app that uses Groq + LangChain
# to answer math, reasoning, and simple knowledge questions.
# It takes a user’s question, sends it to an AI agent,
# and the agent can use tools like a calculator,
# reasoning chain, and Wikipedia search to generate an answer.

# Step-by-step code algorithm

# 1. Import required libraries
# Load Streamlit, LangChain, Groq model, tools, chains, prompts, and callback handler.

# 2. Create the Streamlit app UI
# Set page title, icon, and heading.

# 3. Take Groq API key from sidebar
# Ask the user to enter the API key securely.

# 4. Check API key
# If the key is missing, show a message and stop the app.

# 5. Initialize the language model
# Create a ChatGroq model using llama-3.1-8b-instant.

# 6. Create Wikipedia tool
# Build a tool that searches Wikipedia for general information.

# 7. Create calculator tool
# Build a math chain so the app can solve mathematical expressions.

# 8. Create reasoning prompt and chain
# Define a custom prompt for logical and step-by-step answers,
# then wrap it in an LLMChain.

# 9. Create reasoning tool
# Convert the reasoning chain into a tool for the agent.

# 10. Initialize the agent
# Combine Wikipedia, calculator, and reasoning tools into one LangChain agent.

# 11. Store chat history
# Use st.session_state to save assistant and user messages.

# 12. Display previous messages
# Show the stored chat conversation on the screen.

# 13. Get user question
# Provide a text area where the user enters a problem.

# 14. Handle button click
# When the user clicks the button, start processing the input.

# 15. Save and display user message
# Add the question to chat history and show it in chat.

# 16. Run the agent
# Send the question to the agent so it can choose the right tool and generate an answer.

# 17. Save and show response
# Store the AI reply in session state and display it on the page.

# 18. Show warning if input is empty
# If no question is entered, display a warning.


# Create the Streamlit app UI
st.set_page_config(page_title="Text to Math Problem Solver", page_icon="🧮")
st.title("Text to Math Problem Solver")


# Take Groq API key from sidebar
groq_api_key = st.sidebar.text_input(label="Groq API key", type="password")


# Check API key
if not groq_api_key:
    st.info("Please add your API key")
    st.stop()


# Initialize the language model
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.1-8b-instant"
)


# Create Wikipedia tool
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Internet to find various information on the topics mentioned"
)


# Create calculator tool
def basic_calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"
    
calculator = Tool(
    name="Calculator",
    func=basic_calculator,
    description="Use this only for direct math expressions like 25*4, 100/5, (3+2)*8."
)


# Create reasoning prompt and chain
prompt = """
You are an agent tasked with solving users' mathematical questions.
Logically arrive at the solution and provide a detailed explanation.
Display it point-wise for the question below.

Question: {question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

chain = LLMChain(llm=llm, prompt=prompt_template)


# Create reasoning tool
reasoning_tool = Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)


# initialize_agent(...)
# This function creates an AI agent that can use tools.

# tools=[wikipedia_tool, calculator, reasoning_tool]
# These are the tools given to the agent:
# wikipedia_tool -> for searching information
# calculator -> for math calculations
# reasoning_tool -> for logic and explanation-based answers

# llm=llm
# This tells the agent which language model to use.
# In your code, llm is the ChatGroq model.

# agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
# This chooses the agent type.
# It means:
# Zero-shot -> no training examples are given
# React -> the agent thinks, chooses a tool, gets the result, and then answers
# Description -> it decides which tool to use based on the tool descriptions

# verbose=False
# This hides the internal thinking steps in the console/output.
# If set to True, you can see what the agent is doing step by step.

# handle_parsing_errors=True
# If the agent output is not in the expected format,
# it will try to handle the error instead of crashing.
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)


# Store chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I'm a Math chatbot who can answer all your maths questions"
        }
    ]


# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# Get user question
question = st.text_area(
    "Enter your question:",
    "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. "
    "Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries "
    "contains 25 berries. How many total pieces of fruit do I have at the end?"
)


# Handle button click
if st.button("Find my answer"):
    if question:
        with st.spinner("Generating response..."):
            # Save and display user message
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(
                st.container(),
                expand_new_thoughts=False
            )

            # Run the agent
            response = assistant_agent.run(
                question,
                callbacks=[st_cb]
            )

            # Save and show response
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
            st.write("### Response:")
            st.success(response)
    else:
        # Show warning if input is empty
        st.warning("Please enter the question")