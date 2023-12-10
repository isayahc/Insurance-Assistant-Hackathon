import gradio as gr

from langchain.document_loaders import OnlinePDFLoader

from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub

# from langhchain.llms import openai
from langchain.llms import OpenAI

from langchain.vectorstores import Chroma

from langchain.chains import RetrievalQA

from langchain.document_loaders import PyPDFLoader
from langchain.memory import VectorStoreRetrieverMemory

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import CohereEmbeddings


from langchain.embeddings import HuggingFaceHubEmbeddings, OpenAIEmbeddings

import dotenv

import os

from prompt_template import template

dotenv.load_dotenv()



text_splitter = CharacterTextSplitter(chunk_size=350, chunk_overlap=0)

# flan_ul2 = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={"temperature":0.1, "max_new_tokens":300})
# flan_ul2 = OpenAI()
from langchain.chat_models import ChatOpenAI

flan_ul2 = chat = ChatOpenAI(
    model_name='gpt-3.5-turbo-16k',
    # temperature = self.config.llm.temperature,
    # openai_api_key = self.config.llm.openai_api_key,         
    # max_tokens=self.config.llm.max_tokens
)

global qa

# embeddings = HuggingFaceHubEmbeddings()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=COHERE_API_KEY
)
           




def loading_pdf():
    return "Loading..."
def pdf_changes(pdf_doc):
    # embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceHubEmbeddings()

    embeddings = CohereEmbeddings(
    model="embed-english-light-v3.0",
    # cohere_api_key=COHERE_API_KEY
)
    
    loader = PyPDFLoader(pdf_doc.name)
    documents = loader.load()
    texts = text_splitter.split_documents(documents)
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    # memory = VectorStoreRetrieverMemory(retriever=retriever)
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
    
    # prompt_template = """You have been given a pdf or pdfs. You must search these pdfs. 
    # If you don't know the answer, just say that you don't know, don't try to make up an answer.
    # Only answer the question.
    
    
    # Question: {query}
    # Answer:"""
    # PROMPT = PromptTemplate(
    #     template=prompt_template, input_variables=["context", "question"]
    # )

    prompt = PromptTemplate(input_variables=["chat_history", "human_input", "context"], template=template)
    chain_type_kwargs = {"prompt": prompt}
    global qa 
    # qa = RetrievalQA.from_chain_type(
    #     llm=flan_ul2, 
    #     memory=memory,
    #     chain_type="stuff", 
    #     retriever=retriever, 
    #     return_source_documents=True,
    #     chain_type_kwargs=chain_type_kwargs,
    # )

    prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)
    memory = ConversationBufferMemory(memory_key="history", input_key="question")

    qa = RetrievalQAWithSourcesChain.from_chain_type(llm=flan_ul2, retriever=retriever, return_source_documents=True, verbose=True, chain_type_kwargs={
        "verbose": True,
        "memory": memory,
        "prompt": prompt,
        "document_variable_name": "context"
    }
        )

    return "Ready"

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    response = infer(history[-1][0],"")
    history[-1][1] = response['answer']
    return history

# def bot(history):
#     response = infer(history[-1][0], history)
#     sources = [doc.metadata.get("source") for doc in response['source_documents']]
#     src_list = '\n'.join(sources)
#     print_this = response['answer'] + "\n\n\n Sources: \n\n\n" + src_list
#     return print_this

def infer(question, history) -> dict:
    
    query = question
    # result = qa({"query": query, "context":""})
    # result = qa({"query": query, })
    result = qa({"query": query, "history": history, "question": question})

    # result = result['answer']
    return result

css="""
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
"""

title = """
<div style="text-align: center;max-width: 700px;">
    <h1>Chat with PDF</h1>
    <p style="text-align: center;">Upload a .PDF from your computer, click the "Load PDF to LangChain" button, <br />
    when everything is ready, you can start asking questions about the pdf ;)</p>
</div>
"""


with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)
        # with gr.Blocks() as demo:
        
        with gr.Column():
            pdf_doc = gr.File()
            # pdf_doc = gr.File(label="Load a pdf", file_types=['.pdf'], type="filepath") #try filepath for type if binary does not work
            with gr.Row():
                langchain_status = gr.Textbox(label="Status", placeholder="", interactive=False)
                load_pdf = gr.Button("Load pdf to langchain")
        
        chatbot = gr.Chatbot([], elem_id="chatbot") #.style(height=350)
        with gr.Row():
            question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
    load_pdf.click(loading_pdf, None, langchain_status, queue=False)    
    load_pdf.click(pdf_changes, pdf_doc, langchain_status, queue=False)
    question.submit(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot
    )

demo.launch()
