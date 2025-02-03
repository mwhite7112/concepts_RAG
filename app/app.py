import streamlit as st
from pathlib import Path
from utils.processor import process_chapters
from utils.qa_chain import create_qa_chain

def initialize_qa_system():
    chapters_dir = Path("data")
    vectorstore = process_chapters(chapters_dir)
    qa_chain = create_qa_chain(vectorstore)
    return qa_chain

def main():
    st.title("Math Textbook Q&A")
    
    if "qa_chain" not in st.session_state:
        with st.spinner("Processing chapters..."):
            st.session_state.qa_chain = initialize_qa_system()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt := st.chat_input("Ask about the textbook"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            response = st.session_state.qa_chain.run(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()