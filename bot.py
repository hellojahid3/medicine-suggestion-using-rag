import os
import streamlit as st
from RAG.query_engine import ask_question_with_rag

if __name__ == "__main__":
    st.title("Ask Chatbot!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    try:
        prompt = st.chat_input("Pass your prompt here")

        if prompt:
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            answer = ask_question_with_rag(prompt)

            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
    except Exception as e:
        st.error(f"Error: {str(e)}")
