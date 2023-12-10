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

from prompt.prompt_template import template

dotenv.load_dotenv()



text_splitter = CharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=0
    )

# llm= HuggingFaceHub(
#     repo_id="HuggingFaceH4/zephyr-7b-beta", 
#     model_kwargs={
#         "temperature":0.1, 
#         "max_new_tokens":300
#         }
#         )

# llm= OpenAI()
from langchain.chat_models import ChatOpenAI

llm= chat = ChatOpenAI(
    model_name='gpt-3.5-turbo-16k',
    # temperature = self.config.llm.temperature,
    # openai_api_key = self.config.llm.openai_api_key,         
    # max_tokens=self.config.llm.max_tokens
)

global qa

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=COHERE_API_KEY
)
           


def loading_pdf():
    return "Loading..."
def pdf_changes(pdf_doc):

    embeddings = CohereEmbeddings(
    model="embed-english-light-v3.0",

)
    
    loader = PyPDFLoader(pdf_doc.name)
    documents = loader.load()
    texts = text_splitter.split_documents(documents)
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="human_input"
        )


    prompt = PromptTemplate(
        input_variables=[
            "chat_history",
            "human_input", 
            "context"
            ], 
            template=template
            )


    global qa 


    prompt = PromptTemplate(
    input_variables=[
        "history", 
        "context", 
        "question"
        ],
    template=template,
)
    memory = ConversationBufferMemory(
        memory_key="history", 
        input_key="question"
        )

    qa = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, 
        retriever=retriever, 
        return_source_documents=True, 
        verbose=True, 
        chain_type_kwargs={
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



def infer(question, history) -> dict:
    
    query = question

    result = qa({"query": query, "history": history, "question": question})

    return result

css="""
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
"""

title = """
<div style="text-align: center;max-width: 700px;">
    <h1>Insurance Assistant 💼</h1>
    <p style="text-align: center;">Upload a .PDF from your computer, click the "Load PDF to LangChain" button, <br />
    when everything is ready, you can start asking questions about the pdf ;)</p>
</div>
"""


with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)

        
        with gr.Column():
            pdf_doc = gr.File()

            with gr.Row():
                langchain_status = gr.Textbox(
                    label="Status", 
                    placeholder="", 
                    interactive=False
                    )
                load_pdf = gr.Button("Load pdf to langchain")
        
        chatbot = gr.Chatbot(
            [], 
            elem_id="chatbot"
            ) #.style(height=350)
        
        with gr.Row():
            question = gr.Textbox(
                label="Question", 
                placeholder="Type your question and hit Enter "
                )

    load_pdf.click(
        loading_pdf, 
        None, 
        langchain_status, 
        queue=False
        )
    
    load_pdf.click(
        pdf_changes, 
        pdf_doc, 
        langchain_status, 
        queue=False
        )
    
    question.submit(
        add_text, 
     [
        chatbot, 
        question
      ], 
      [
          chatbot, 
          question
          ]
          ).then(
        bot, 
        chatbot, 
        chatbot
    )

demo.launch()
