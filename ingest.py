from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.text_splitter import SentenceSplitter

documents = SimpleDirectoryReader("./data").load_data()

text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
service_context = ServiceContext.from_defaults(text_splitter=text_splitter)

index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

from llama_index.query import QueryBuilder

# Define the query text
query_text = "How does the weather affect crop growth?"

# Preprocess the query text
query_builder = QueryBuilder(service_context)
query = query_builder.build_query(query_text)

# Search for similar documents or retrieve relevant information
results = index.search(query)

# Process the search results
for result in results:
    document_id = result.document_id
    score = result.score
    document = documents[document_id]
    # Process the retrieved document or display the relevant information
    print(f"Document ID: {document_id}, Score: {score}")
    print(f"Document Text: {document.text}")