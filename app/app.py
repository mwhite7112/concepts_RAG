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
        
    # Initialize chat_history if not present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt := st.chat_input("Ask about the textbook"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            # Use the chain with chat history
            response = st.session_state.qa_chain.invoke({
                "input": prompt,
                "chat_history": st.session_state.chat_history
            })
            
            answer = response['answer']
            st.markdown(answer)
            
            # Update messages and chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.chat_history.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer}
            ])
            
            if 'source_documents' in response:
                with st.expander("Sources"):
                    for doc in response['source_documents']:
                        st.write(f"From: {doc.metadata['source']}")
                        st.write(doc.page_content)

if __name__ == "__main__":
    main()