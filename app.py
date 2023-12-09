import gradio as gr

from langchain.document_loaders import OnlinePDFLoader

from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate

text_splitter = CharacterTextSplitter(chunk_size=350, chunk_overlap=0)

from langchain.llms import HuggingFaceHub
flan_ul2 = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={"temperature":0.1, "max_new_tokens":300})

from langchain.embeddings import HuggingFaceHubEmbeddings
embeddings = HuggingFaceHubEmbeddings()

from langchain.vectorstores import Chroma

from langchain.chains import RetrievalQA
def loading_pdf():
    return "Loading..."
def pdf_changes(pdf_doc):
    loader = OnlinePDFLoader(pdf_doc.name)
    documents = loader.load()
    texts = text_splitter.split_documents(documents)
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    
    prompt_template = """You have been given a pdf or pdfs. You must search these pdfs. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Only answer the question.
    
    {context}
    
    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    global qa 
    qa = RetrievalQA.from_chain_type(
        llm=flan_ul2, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )
    return "Ready"

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    response = infer(history[-1][0])
    history[-1][1] = response['result']
    return history

def infer(question):
    
    query = question
    result = qa({"query": query})

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
        
        with gr.Column():
            pdf_doc = gr.File(label="Load a pdf", file_types=['.pdf'], type="file")
            with gr.Row():
                langchain_status = gr.Textbox(label="Status", placeholder="", interactive=False)
                load_pdf = gr.Button("Load pdf to langchain")
        
        chatbot = gr.Chatbot([], elem_id="chatbot").style(height=350)
        with gr.Row():
            question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
    load_pdf.click(loading_pdf, None, langchain_status, queue=False)    
    load_pdf.click(pdf_changes, pdf_doc, langchain_status, queue=False)
    question.submit(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot
    )

demo.launch()