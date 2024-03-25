


# def main():
#     load_dotenv()
#     st.set_page_config(page_title="Ask your PDF")
#     st.header("Ask your PDF 💬")
    
#     # upload file
#     pdf = st.file_uploader("Upload your PDF", type="pdf")
    
#     # extract the text
#     if pdf is not None:
#       pdf_reader = PdfReader(pdf)
#       text = ""
#       for page in pdf_reader.pages:
#         text += page.extract_text()
        
#       # split into chunks
#       text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#       )
#       chunks = text_splitter.split_text(text)
      
#       # create embeddings
#       embeddings = OpenAIEmbeddings()
#       knowledge_base = FAISS.from_texts(chunks, embeddings)
      
#       # show user input
#       user_question = st.text_input("Ask a question about your PDF:")
#       if user_question:
#         docs = knowledge_base.similarity_search(user_question)
        
#         llm = OpenAI()
#         chain = load_qa_chain(llm, chain_type="stuff")
#         with get_openai_callback() as cb:
#           response = chain.run(input_documents=docs, question=user_question)
#           print(cb)
           
#         st.write(response)
    

# if __name__ == '__main__':
#     main()



from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your Uploaded PDF 💬")

    pdf = st.file_uploader("Upload your PDF", type="pdf")
    knowledge_base = None  # Initialize knowledge_base as None

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ""
        
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

    user_question = st.text_input("Ask a question about your PDF:")

    if user_question:
        if user_question.lower().strip() == "call":
            name = st.text_input("Enter your name:")
            email = st.text_input("Enter your email:")
            number = st.text_input("Enter your phone number:")

            # Add a button to submit the information
            if st.button("Submit"):
                # Here, you can process the collected information
                st.success(f"Thank you, {name}. We will contact you soon.")
                # Example: st.write(f"Name: {name}, Email: {email}, Number: {number}")
        elif knowledge_base:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
            
            st.write(response)
        else:
            st.warning("Please upload a PDF file to ask questions about its content.")
    
if __name__ == '__main__':
    main()