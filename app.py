import streamlit as st
from dotenv import load_dotenv
from utils import *
import uuid

#Creating session variables
if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] =''

def main():
    load_dotenv()

    st.set_page_config(page_title="Screening Assistance")

    job_description = st.text_area("Please paste the 'JOB DESCRIPTION' here...",key="1")
    document_count = st.text_input("No.of 'documents' to return",key="2")
    
    pdf = st.file_uploader("Upload here, only PDF files allowed", type=["pdf"],accept_multiple_files=True)

    submit=st.button("Help me with the analysis")

    if submit:
        with st.spinner('Wait for it...'):

            #Creating a unique ID, so that we can use to query and get only the user uploaded documents from PINECONE vector store
            st.session_state['unique_id']=uuid.uuid4().hex

            #Create a documents list out of all the user uploaded pdf files
            final_docs_list=create_docs(pdf,st.session_state['unique_id'])

        
            st.write("*uploaded* :"+str(len(final_docs_list)))
            embeddings=create_embeddings_load_data()
            push_to_pinecone("71adf081-aace-4ee4-be84-0a9076ad361e","gcp-starter","test",embeddings,final_docs_list)

            relavant_docs=similar_docs(job_description,document_count,"71adf081-aace-4ee4-be84-0a9076ad361e","gcp-starter","test",embeddings,st.session_state['unique_id'])


            #Introducing a line separator
            st.write(":heavy_minus_sign:" * 30)

            #For each item in relavant docs - we are displaying some info of it on the UI
            for item in range(len(relavant_docs)):
                
                st.subheader("👉 "+str(item+1))

                #Displaying Filepath
                st.write("**File** : "+relavant_docs[item][0].metadata['name'])

                #Introducing Expander feature
                with st.expander('Show me 👀'): 
                    st.info("**Match Score** : "+str(relavant_docs[item][1]))
                    
                    #Gets the summary of the current item using 'get_summary' function that we have created which uses LLM & Langchain chain
                    summary = get_summary(relavant_docs[item][0])
                    st.write("**Summary** : "+summary)

        st.success("end")


#Invoking main function
if __name__ == '__main__':
    main()
