from typing import List, AnyStr


class Retriever:
    def get_documents(self, documents_path: str = 'documents.txt') -> List[AnyStr]:
        documents: List[AnyStr] = []
        with open(documents_path, 'r') as file:
            lines = file.readlines()
            for doc in lines:
                doc = doc.strip().replace('\n', '')
                documents.append(doc)
        return documents
