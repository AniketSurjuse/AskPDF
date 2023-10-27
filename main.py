import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
import os
from langchain.chains.question_answering import load_qa_chain
import pickle

# os.environ['OPENAI_API_KEY'] = "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"


st.header(":red[AskPDF]📝")
def main():
    st.subheader("Chat with PDF")
    pdf = st.file_uploader("upload your PDF",type='pdf')

    if pdf:
        pdfreader = PdfReader(pdf)
        # st.write(pdfreader)
        text = ""
        for page in pdfreader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        file_name = pdf.name[:-4]
        if os.path.exists(f"{file_name}.pkl"):
            with open(f"{file_name}.pkl",'rb') as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks,embedding=embeddings)
            with open(f"{file_name}.pkl",'wb') as f:
                pickle.dump(VectorStore,f)

        # st.write(chunks)
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

        # query = st.chat_input("Whats up?")
        # chain = load_qa_chain(OpenAI(model_name='gpt-3.5-turbo'), chain_type="stuff")
        # docs = VectorStore.similarity_search(query)
        # response = chain.run(input_documents=docs, question=query)

        query = st.chat_input("Whats up?")
        if query:
            with st.chat_message('user'):
                st.markdown(query)

            st.session_state.messages.append({'role': 'user', 'content': query})

            chain = load_qa_chain(OpenAI(model_name='gpt-3.5-turbo'), chain_type="stuff")
            docs = VectorStore.similarity_search(query)
            response = chain.run(input_documents=docs, question=query)
            with st.chat_message('assistant'):
                st.markdown(response)

            st.session_state.messages.append({'role': 'assistant', 'content': response})
        # st.write(query)


# st.footer("Made with ❤️ by Aniket")
if __name__ == '__main__':
    main()
