import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import os

# --- HARDCODE YOUR API KEY HERE ---
# Replace "YOUR_GOOGLE_API_KEY" with your actual Google API key
GOOGLE_API_KEY = "AIzaSyD1hCJHjlqZfirXG_JUVnGVu2T9W7xwIg4"
# ----------------------------------

def get_pdf_text(pdf_file):
    """Extracts text from the uploaded PDF file."""
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def get_text_chunks(text):
    """Splits the text into manageable chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Creates a FAISS vector store from text chunks."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(text_chunks, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def get_gemini_response(prompt):
    """Gets a response from the Gemini API."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred with the Gemini API: {e}"

def main():
    st.set_page_config(page_title="Studymate Q&A", page_icon="📚", layout="centered")

    # Custom CSS for chat bubbles
    st.markdown("""
        <style>
            .chat-message {
                padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;
            }
            .chat-message.user {
                background-color: #2b313e;
            }
            .chat-message.bot {
                background-color: #475063;
            }
            .chat-message .avatar {
                width: 20%;
            }
            .chat-message .avatar img {
                max-width: 78px; max-height: 78px; border-radius: 50%; object-fit: cover;
            }
            .chat-message .message {
                width: 80%; padding: 0 1.5rem; color: #fff;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📚 Studymate: AI-Powered PDF Q&A Assistant")

    with st.sidebar:
        st.header("Your Document")
        uploaded_file = st.file_uploader("Upload a PDF and start asking questions", type="pdf")

    if uploaded_file:
        # Process the document only once
        if "vector_store" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
            with st.spinner("Processing your document..."):
                raw_text = get_pdf_text(uploaded_file)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = get_vector_store(text_chunks)
                    st.session_state.file_name = uploaded_file.name
                    st.session_state.messages = [] # Initialize chat history
                    st.sidebar.success("Document processed!")
                else:
                    st.sidebar.error("Could not extract text. Please try another PDF.")
        
        # Display chat messages from history
        if "messages" in st.session_state:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Chat input for user questions
        if prompt := st.chat_input("Ask a question about your document..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Thinking..."):
                    try:
                        # Configure Gemini API
                        if GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY":
                             st.error("Please add your Google API Key to the code.")
                             return
                        genai.configure(api_key=GOOGLE_API_KEY)

                        # Perform similarity search
                        docs = st.session_state.vector_store.similarity_search(prompt, k=3)
                        context = "\n".join([doc.page_content for doc in docs])
                        
                        # Construct the final prompt for the model
                        full_prompt = f"""
                        Answer the following question based only on the provided context. 
                        If the answer is not in the context, say "I don't know, the answer is not in the document."

                        Context:
                        {context}

                        Question:
                        {prompt}
                        """
                        
                        response = get_gemini_response(full_prompt)
                        message_placeholder.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                    except Exception as e:
                        error_message = f"An error occurred: {e}"
                        message_placeholder.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})

    else:
        st.info("Please upload a PDF document to begin.")

if __name__ == '__main__':
    main()
