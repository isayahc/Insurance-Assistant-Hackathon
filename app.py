import gradio as gr


from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import CohereEmbeddings
from langchain.schema.embeddings import Embeddings

from utils.qa_utils import create_embeddings_from_pdf

# from utils import create_embeddings_from_pdf

from config import COHERE_API_KEY


global qa
from qa import qa



def start_gradio(qa,embedding:Embeddings):

    def loading_pdf():
        return "Loading..."
    def pdf_changes(pdf_doc):


        # global qa
        # embeddings = CohereEmbeddings(
        #     model="embed-english-v3.0",
        #     cohere_api_key=COHERE_API_KEY
        # )
        # db = Chroma.from_documents(texts, embeddings)
        # db = Chroma()

        db = create_embeddings_from_pdf(
            pdf_doc,
            embedding,
            Chroma(),


        )

        # text_splitter = CharacterTextSplitter(
        #     chunk_size=350,
        #     chunk_overlap=0
        # )

        # loader = PyPDFLoader(pdf_doc.name)
        # documents = loader.load()
        # texts = text_splitter.split_documents(documents)
        # db = Chroma.from_documents(texts, embeddings)


        retriever = db.as_retriever()

        qa.retriever = retriever



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
        <h1>Chat with PDF</h1>
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

    return demo



if __name__ == '__main__':
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=COHERE_API_KEY
    )
    demo = start_gradio(qa)
    demo.launch()
