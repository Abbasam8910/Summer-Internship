# Chat With Multiple PDF Documents With Langchain And Google Gemini Pro or using chromadb
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("Google_API_Key")
genai.configure(api_key=os.getenv("Google_API_Key"))






def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




import streamlit as st
from PIL import Image

# Modern page configuration
st.set_page_config(page_title="📄 Ask My PDF", layout="wide")

def main():
    # Header section with icon
    st.markdown("""
        <style>
            .main-title {
                font-size: 2.5rem;
                font-weight: bold;
                color: #4A90E2;
            }
        </style>
        <div class="main-title">💁 Ask My PDF</div>
    """, unsafe_allow_html=True)

    # Layout split: input on left, result on right
    col1, col2 = st.columns([2, 1])

    with col1:
        user_question = st.text_input("🔍 Ask a question about your uploaded PDF(s):", placeholder="E.g., Summarize chapter 2...")

        if user_question:
            st.markdown("#### Thinking ....")
            user_input(user_question)  # You need to define this function

    # Sidebar menu with styling
    with st.sidebar:
        st.image("https://brandeps.com/icon-download/U/User-icon-vector-01.svg", width=150)
        st.markdown("## 📂 Upload PDFs")
        pdf_docs = st.file_uploader("Upload your PDF files below", accept_multiple_files=True, type=['pdf'])

        if pdf_docs:
            st.markdown("### ✅ Ready to process your files.")
        else:
            st.info("Please upload one or more PDF files to get started.")

        if st.button("🚀 Submit & Process"):
            if pdf_docs:
                with st.spinner("⏳ Processing your PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)  # You need to define this
                    text_chunks = get_text_chunks(raw_text)  # You need to define this
                    get_vector_store(text_chunks)  # You need to define this
                    st.success("✅ PDFs processed successfully!")
            else:
                st.warning("⚠️ Please upload at least one PDF before processing.")

    # Footer / Help
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        - Upload one or more PDF documents using the sidebar.
        - Ask a question related to the uploaded content.
        - Wait for Gemini to respond using the processed knowledge.
        """)

if __name__ == "__main__":
    main()
