#This code sets up a basic Streamlit application that loads a language model chain,
# handles user input and audio transcription, 
#saves chat history to a JSON file, with the goal of creating a conversational AI interface.
import streamlit as st
from llm_chains import load_normal_chain
from langchain.memory import StreamlitChatMessageHistory
from streamlit_mic_recorder import mic_recorder
from utils import save_chat_history_json, get_timestamp, load_chat_history_json
from audio_handler import transcribe_audio
import yaml
import os

#This code sets up a Streamlit application for a multimodal AI assistant.
#It initializes the page, manages session state, and sets up UI elements for text input,
#voice recording, and audio file uploads.
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

chat_history_path = config.get("chat_history_path", "chat_sessions")
os.makedirs(chat_history_path, exist_ok=True)  # Ensure directory exists

def load_chain(chat_history):
    return load_normal_chain(chat_history)

def clear_input_field():
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""

def set_send_input():
    st.session_state.send_input = True
    clear_input_field()

def track_index():
    st.session_state.session_index_tracker = st.session_state.session_key

def save_chat_history():
    if st.session_state.history:
        session_filename = st.session_state.session_key if st.session_state.session_key != "new_session" else f"{get_timestamp()}.json"
        st.session_state.new_session_key = session_filename  # Save new session key if needed
        save_chat_history_json(st.session_state.history, os.path.join(chat_history_path, session_filename))

def main():
    st.set_page_config(page_title="💬 Multimodal Local AI Powered by Hugging Face", layout="wide")
    st.title("Multimodal Local AI ⚡Powered by Hugging Face🤗⚡")
    st.sidebar.title("Chat Sessions")
    chat_sessions = ["new_session"] + os.listdir(config["chat_history_path"])

    if "send_input" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.send_input = False
        st.session_state.user_question = ""
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"
    if st.session_state.session_key == "new_session" and st.session_state.new_session_key != None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    
    index = chat_sessions.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox("Select Chat Session", chat_sessions, key="session_key", index=index, on_change=track_index)
    
    if st.session_state.session_key != "new_session":
        st.session_state.history = load_chat_history_json(os.path.join(chat_history_path, st.session_state.session_key))
    else:
        st.session_state.history = []
    
    chat_history = StreamlitChatMessageHistory(key="history")
    llm_chain = load_chain(chat_history)
    chat_container = st.container()
    
    col1, col2, col3 = st.columns([6, 1, 1])  # Added a third column for voice recording

    with col1:
        st.text_input("Type your message here", key="user_input", on_change=set_send_input, label_visibility="collapsed")

    with col2:
        send_button = st.button("Send", key="send_button", on_click=clear_input_field)

    with col3:
        voice_recording=mic_recorder(start_prompt="Start recording",stop_prompt="Stop recording", just_once=True)
    
    
    uploaded_audio = st.sidebar.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
    
    
    rendered_chat = False
    if uploaded_audio:
        transcribed_audio = transcribe_audio(uploaded_audio.getvalue())
        print(transcribed_audio)
        if transcribed_audio:
            response = llm_chain.run("Summarize this text: " + transcribed_audio)
            save_chat_history()
            with chat_container:
                for message in chat_history.messages:
                    st.chat_message(message.type).write(message.content)
                    rendered_chat = True  

            

    rendered_chat = False
    if voice_recording:
        transcribed_audio = transcribe_audio(voice_recording["bytes"])
        print(transcribed_audio)
        if transcribed_audio:
            chat_history.add_user_message(transcribed_audio)
            response = llm_chain.run(transcribed_audio)
            chat_history.add_ai_message(response)
            save_chat_history()
            with chat_container:
                for message in chat_history.messages:
                    st.chat_message(message.type).write(message.content)
                    rendered_chat = True  

        
    # Process user input
    if send_button or st.session_state.send_input:
        if st.session_state.user_question:
                llm_response = llm_chain.run(st.session_state.user_question)
                st.session_state.user_question = ""
        
        if chat_history.messages:
            with chat_container:
                st.write("Chat History:")
                for message in chat_history.messages:
                    st.chat_message(message.type).write(message.content)
        
        save_chat_history()

if __name__ == "__main__":
    main()
