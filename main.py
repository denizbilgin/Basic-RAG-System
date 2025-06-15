from rag import Rag
from retriever import Retriever

if __name__ == '__main__':
    query_1 = 'What are the main types of renewable energy?'
    query_2 = "How is solar power generated and what are its recent advancements?"
    query_3 = "Why is battery technology important for renewable energy?"
    query_4 = "What is the capital of France?"

    retriever = Retriever()
    documents = retriever.get_documents()

    rag = Rag(documents)
    rag.run_rag_system(query_1)
