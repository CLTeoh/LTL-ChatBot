import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from streamlit_chat import message
from streamlit_extras.add_vertical_space import add_vertical_space
from dotenv import load_dotenv

load_dotenv()

# Set consistent styling
st.set_page_config(page_title='LTL Chat App', page_icon='your_icon.png', layout='wide')
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #f4f4f4;
        }
        .main .block-container {
            max-width: 1200px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar contents
with st.sidebar:
    st.image('cb.png', width=100)
    st.title('LTL Chat App 💬')
    st.markdown('''
    ## About 
    This app allows you to chat with PDF documents using our services.
    - Q&A
    - Summarizing
    - Error Checking
    ''')
    add_vertical_space(2)
    st.write('Made with ❤️ by us')

def main():
    st.title('LTL Chat App 💬')
    st.write("Chat with PDF")

    # Insert PDF file here
    pdf_get = st.file_uploader("Upload your PDF file here:", accept_multiple_files=False)

    if pdf_get is not None:
        # Check if the uploaded file is a PDF
        try:
            pdf_got = PdfReader(pdf_get)
            st.success('PDF successfully uploaded! Ready to chat.')

            text = ""
            for page in pdf_got.pages:
                text += page.extract_text()

            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=600,
                length_function=len
            )

            chunks = text_splitter.split_text(text=text)

            # Load Embbeding Model
            modelPath = "llmrails/ember-v1"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False}
            embeddings = HuggingFaceEmbeddings(
                model_name=modelPath,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )

            #st.write(chunks)
                # Create and Load VectorStore
            db = FAISS.from_texts(chunks, embeddings)

            # Create a prompt template
            template = """You are an AI assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Try to give an answer in maximum 50 words that is descriptive, complete, accurate and concise.
            Make sure the answer is related to the data stored in the VectorStore, db.
            Context: {context} 
            Question: {question} 
            Answer:
            """
            prompt = ChatPromptTemplate.from_template(template)

            # Import LLM Model
            llm = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={"temperature": 0.7, "max_length": -1})
            memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

            conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 4}), memory=memory)

            st.session_state.conversation = conversation_chain

            # create a RAG pipeline
            rag_chain = (
                {"context": conversation_chain, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            # Handle user input
            user_question = st.text_input("Ask Away:")
            if user_question:
                handle_userinput(user_question, chunks, embeddings, db, rag_chain)

        except Exception as e:
            st.error(f"Error: {e}. Please upload a valid PDF file.")


def handle_userinput(user_question, chunks, embeddings, db, rag_chain):

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    reply = rag_chain.invoke(user_question)

   # Layout of input/response containers
    response_container = st.container()

    with response_container:
        # Display user's question
        message(user_question, is_user=True, key="user_question")

        # Break chatbot response into chunks for better visibility
        chatbot_responses = reply.split()
        chunk_size = 1000  # Adjust the chunk size as needed

        # Display chatbot response in chunks
        for i in range(0, len(chatbot_responses), chunk_size):
            chunk = " ".join(chatbot_responses[i:i+chunk_size])
            message(chunk, is_user=False, key=f"chatbot_response_chunk_{i}")

if __name__ == '__main__':
    main()

# Footer section
st.markdown('---')
st.text('Version 1.0.0')
st.markdown('For support, contact us at ltlsupport@gmail.com')
st.text('View source code on [GitHub](https://github.com/taruc/ltlChatApp)')
