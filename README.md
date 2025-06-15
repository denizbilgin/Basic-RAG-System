# Retrieval-Augmented Generation (RAG) with GPT-2

This project implements a lightweight **Retrieval-Augmented Generation (RAG)** system built around the GPT-2 language model. The goal is to enhance GPT-2's ability to generate accurate and context-aware responses by integrating external knowledge retrieved from static text files.

While simple by design, this system demonstrates how combining vector-based retrieval with generation can simulate up-to-date knowledge in a modular and extendable architecture.

---

## How It Works

1. A user submits a query and a value `k`, specifying how many similar texts to retrieve.
2. Cosine similarity is used to compare the query against a collection of pre-loaded knowledge texts.
3. The top `k` most relevant texts are selected and used as context for GPT-2.
4. GPT-2 then generates a response as if it had up-to-date knowledge about the topic.

> The entire system is built using clean code principles and is highly modular — each component can be swapped or extended independently.

---

## Dependencies

- `transformers`
- `torch`
- `scikit-learn`
- `sentence-transformers`
- `numpy`
- `textwrap`

## Example Usage

```python
    query_1 = 'What are the main types of renewable energy?'
    query_2 = "How is solar power generated and what are its recent advancements?"
    query_3 = "Why is battery technology important for renewable energy?"
    query_4 = "What is the capital of France?"

    retriever = Retriever()
    documents = retriever.get_documents()

    rag = Rag(documents)
    rag.run_rag_system(query_1)
```

Output:
```
    Embedding model 'all-MiniLM-L6-v2' loaded.
    
    --- Running RAG for Query: 'What are the main types of renewable energy?' ---
    Retrieving relevant documents...
    Generated embeddings for 8 documents.
    Each embedding has a dimension of: 384
    Retrieved 3 documents:
      Doc 1 (Score: 0.7328): Renewable energy sources include solar, wind, hydro, and geothermal power. They are naturally...
      Doc 2 (Score: 0.4867): Wind energy harnesses the kinetic energy of wind using large turbines. Offshore wind farms are a...
      Doc 3 (Score: 0.4794): Fossil fuels like coal, oil, and natural gas are non-renewable and contribute significantly to...

    Generating response with LLM...
    Generated embeddings for 8 documents.
    Each embedding has a dimension of: 384
    
    CUDA not available, model running on CPU.
    LLM model 'gpt2' and tokenizer loaded.
    
    --- Generated Response ---
    Yes, you can use this if you are writing a query that requires a few different
    queries. For example, if your question requires data from a site where certain
    information is reported to the search engines, then you could use the answer
    that you provided. In the event the query does not answer, use a different
    query. This is because the data for the site could be different from the actual
    data. To provide the correct answer for your query, write the response that the
    user needs in the form of a question. So, the first step is to select the most
    appropriate query for you. Next, specify the type of the database you want to
    query and the page that will be displayed in your application. If you have only
    a single page for all
```
