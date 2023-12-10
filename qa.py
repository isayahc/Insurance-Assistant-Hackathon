from langchain.document_loaders import OnlinePDFLoader


from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub

# from langhchain.llms import openai
from langchain.llms import OpenAI



from langchain.chains import RetrievalQA


from langchain.memory import VectorStoreRetrieverMemory

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import CohereEmbeddings


from langchain.embeddings import HuggingFaceHubEmbeddings, OpenAIEmbeddings


import os

from prompt.prompt_template import template

from config import COHERE_API_KEY




from langchain.chat_models import ChatOpenAI



llm= chat = ChatOpenAI(
    model_name='gpt-3.5-turbo-16k',

)


embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=COHERE_API_KEY
)




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


# prompt = PromptTemplate(
# input_variables=[
#     "history", 
#     "context", 
#     "question"
#     ],
# template=template,
# )


qa = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm, 
    # retriever=retriever(), 
    return_source_documents=True, 
    verbose=True, 
    chain_type_kwargs={
        "verbose": True,
        "memory": memory,
        "prompt": prompt,
        "document_variable_name": "context"
        }
    )