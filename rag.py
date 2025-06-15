import textwrap
from typing import List, AnyStr, Tuple
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from embedder import Embedder
from transformers import AutoTokenizer, AutoModelForCausalLM


class Rag:
    def __init__(self, documents: List[AnyStr], embedding_model_name: AnyStr = 'all-MiniLM-L6-v2', generator_model_name: AnyStr = 'gpt2', prompt_path: AnyStr = 'prompt.txt'):
        self.documents = documents
        self.embedder = Embedder(embedding_model_name)
        self.generator_model_name = generator_model_name
        self.prompt_path = prompt_path

        self.tokenizer = AutoTokenizer.from_pretrained(self.generator_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

    def __retrieve_k_relevant_documents(self, query: AnyStr, top_k: int = 3) -> List[Tuple[AnyStr, List[float]]]:
        query_embedding = self.embedder.embedding_model.encode(query, convert_to_tensor=True)

        # Getting document embeddings
        document_embeddings = self.embedder.embed_documents(self.documents)

        # Calculating cosine similarity between query and document embeddings
        similarities: List[float] = cosine_similarity(query_embedding.unsqueeze(0).cpu(), document_embeddings.cpu())[0]

        top_k_indices = np.argsort(similarities)[::-1][:top_k]

        retrieved_info: List[Tuple[AnyStr, List[float]]] = []
        for index in top_k_indices:
            retrieved_info.append((self.documents[index], similarities[index]))
        return retrieved_info

    def __get_cuda_info(self, generator_model: AutoModelForCausalLM) -> AnyStr:
        if torch.cuda.is_available():
            generator_model.to('cuda')
            return "Model moved to CUDA (GPU)."
        else:
            return "CUDA not available, model running on CPU."

    def __generator(self):
        try:
            generator_model = AutoModelForCausalLM.from_pretrained(self.generator_model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
            print(self.__get_cuda_info(generator_model))
        except Exception as e:
            print(f"Error loading model with float16 or 8bit: {e}. Trying without specific dtype/quantization.")
            generator_model = AutoModelForCausalLM.from_pretrained(self.generator_model_name, low_cpu_mem_usage=True)
            print(self.__get_cuda_info(generator_model))

        # Setting model to evaluation mode
        generator_model.eval()
        print(f"LLM model '{self.generator_model_name}' and tokenizer loaded.")
        return generator_model

    def __generate_response_with_llm(self, query: AnyStr, top_k: int = 3) -> AnyStr:
        # Combining top k documents
        retrieved_documents_info = self.__retrieve_k_relevant_documents(query, top_k)
        context = "\n".join([doc_text for doc_text, _ in retrieved_documents_info])

        # Building the prompt
        with open(self.prompt_path, "r", encoding="utf-8") as file:
            prompt = file.read()
            prompt.replace("{context}", context)
            prompt.replace("{query}", query)

        # Tokenizing the prompt
        inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=1024, truncation=True)
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')

        # Generating response
        generator = self.__generator()
        output_sequences = generator.generate(
            inputs,
            max_length=inputs.shape[1] + 150,  # Generating up to 150 new tokens
            temperature=0.7,
            top_k=50,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            no_repeat_ngram_size=2  # Avoiding repeating n-grams of size 2
        )

        generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)

        # Finding the start of the answer part
        answer_prefix = "Answer:"
        if answer_prefix in generated_text:
            response = generated_text.split(answer_prefix, 1)[1].strip()
        else:
            response = generated_text.strip()

        # Removing the original prompt from the generated text if it's still there
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        return response

    def run_rag_system(self, query, top_k: int = 3) -> AnyStr:
        print(f"\n--- Running RAG for Query: '{query}' ---")

        # Retrieving relevant document
        print("Retrieving relevant documents...")
        retrieved_docs_info = self.__retrieve_k_relevant_documents(query, top_k)

        print(f"Retrieved {len(retrieved_docs_info)} documents:")
        for i, (doc_text, score) in enumerate(retrieved_docs_info):
            print(f"  Doc {i + 1} (Score: {score:.4f}): {textwrap.shorten(doc_text, width=100, placeholder='...')}")

        # Generating response using LLM with retrieved context
        print("Generating response with LLM...")
        generated_response = self.__generate_response_with_llm(query, top_k)

        print("\n--- Generated Response ---")
        print(textwrap.fill(generated_response, width=80))

        return generated_response
