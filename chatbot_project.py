import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import torch
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Your AI Assistant")
st.title("Your AI assistant")

model_class = "hf_hub" # @param ["hf_hub", "openai", "ollama"]

def model_hf_hub(model="meta-llama/Meta-llama-3-8B-Instruct", temperature=0.1):
    llm = HuggingFaceEndpoint(
        repo_id=model,
        temperature=temperature,
        max_new_tokens=512,
        return_full_text=False,
        task="text-generation",
    )

    return llm

def model_openai(model="gpt-4o-mini", temperature=0.1):
   llm = ChatOpenAI(model=model, temperature=temperature)
   return llm


def model_ollama(model="phi3", temperature=0.1):
   llm = ChatOllama(model=model, temperature=temperature)
   return llm


def model_response(user_query, chat_history, model_class):
    if model_class == "hf_hub":
        llm = model_hf_hub()
    elif model_class == "openai":
        llm = model_openai()
    elif model_class == "ollama":
        llm = model_ollama()
    
    # prompt
    system_prompt = """
        You are a helpful assistant answering the general questions. Please respond in {language}.
    """

    language = "the same language the user is using to chat"

    if model_class.startswith("hf"):
        user_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{input}<|start_header_id|>assistant<|end_header_id|>\n"
    else:
        user_prompt = "{input}"


    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", user_prompt)
    ])

    # chain
    chain = prompt_template | llm | StrOutputParser()

    # Response
    return chain.stream({
        "chat_history": chat_history,
        "input": user_query,
        "language": language
    })
        

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content = "Hi, I'm your virtual assistant! How can i help you?")
    ]

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)


user_query = st.chat_input("Enter your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)
    
    with st.chat_message("AI"):
        placeholder = st.empty()
        final_response = ""
        for chunk in model_response(
            user_query=user_query,
            chat_history=st.session_state.chat_history,
            model_class="hf_hub"  # or "openai", "ollama"
        ):
            cleaned_chunk = chunk.replace("<|eot_id|>", "")
            final_response += cleaned_chunk
            placeholder.markdown(final_response)
            # st.write(cleaned_chunk, unsafe_allow_html=True)
        
        st.session_state.chat_history.append(AIMessage(content=final_response))
        # resp = st.write_stream(model_response(user_query=user_query, chat_history=st.session_state.chat_history, model_class=model_class))
    
    
    # st.session_state.chat_history.append(AIMessage(content=resp))
    
    

