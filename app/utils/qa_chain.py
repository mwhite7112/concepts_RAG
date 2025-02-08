from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

def create_qa_chain(vectorstore):
    # Initialize the base retriever
    retriever = vectorstore.as_retriever()
    
    # Initialize LLM
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    
    # Create a history-aware retriever
    contextualize_q_system_prompt = """
    Given a chat history and the latest user question which might reference context in the chat history, 
    formulate a standalone question which can be understood without the chat history. 
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
    """
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, 
        retriever, 
        contextualize_q_prompt
    )
    
    # Create the response prompt
    qa_system_prompt = """
    Use the following conversation history and retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    
    Keep your responses focused and relevant to the question asked.
    """
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("system", "Here's relevant context: {context}"),
        ]
    )
    
    # Create the document chain
    document_chain = create_stuff_documents_chain(
        llm,
        qa_prompt
    )
    
    # Create the final retrieval chain
    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        document_chain
    )
    
    return retrieval_chain