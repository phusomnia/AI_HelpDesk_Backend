from typing import Any, List
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.Features.LangChainAPI.LangChainDTO import Callback, ChunkResponse
from src.SharedKernel.utils.yamlenv import load_env_yaml

config = load_env_yaml()

law_separators=[
    r"\nĐiều\s+\d+",
    r"\n\d+\.\s",
    r"\n[a-z]\)\s",
    "\n\n",
    "\n"
]
class Process:
    def __init__(self,         
        provider: BaseChatModel, 
        callbacks: Callback
    ) -> None:
        self.provider = provider
        self.callbacks = callbacks or {}
    ...

    def _split_docs(self, text: str):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.splitter.chunk_size,
            chunk_overlap=config.splitter.chunk_overlap,
            separators=law_separators,
            is_separator_regex=True,
            strip_whitespace=True  
        )

        docs = text_splitter.create_documents([text])

        chunks = [
            ChunkResponse(
                index=i, 
                content=doc.page_content, 
                length=len(doc.page_content)
            )
            for i, doc in enumerate(docs)
        ]

        return chunks
        ...
    def split_by_page(self, docs: List[Document]):
        for doc in docs:
            page_chunks = self._split_docs(doc.page_content)
            print(f" -> Created {len(page_chunks)} chunks")

            for chunk in page_chunks:
                ...
        ...
    def _split_docs_PaC(self, docs_PaC: dict[str, Any]):
        ...
    def split_PaC(self, docs: List[Document]):
        parent_chunks = []
        child_chunks = []

        full_text = ""
        page_map = [] 

        cursor = 0

        for doc in docs:
            text = doc.page_content
            start = cursor
            end = cursor + len(text)

            page_map.append({
                "start": start,
                "end": end,
                "page_number": doc.metadata["page_number"]
            })

            full_text += text
            cursor = end

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=410,
            separators=law_separators,
            is_separator_regex=True,
            add_start_index=True
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=102,
            separators=law_separators,
            is_separator_regex=True
        )
        
        parent_docs = parent_splitter.create_documents([full_text])
        parent_chunks = []
        child_chunks = []
        for idx, parent in enumerate(parent_docs):
            start_index = parent.metadata["start_index"]

            page_number = 1

            for p in page_map:
                if p["start"] <= start_index < p["end"]:
                    page_number = p["page_number"]
                    break
                
            parent_id = f"parent_docs:{docs[0].metadata["source"]}:page_{page_number}:{idx}"

            parent.metadata = {
                "parent_id": parent_id,
                "page_number": page_number,
                "source": docs[0].metadata["source"]
            }

            parent_chunks.append(parent)

            children = child_splitter.split_documents([parent])

            for child in children:
                child.metadata = {
                    "parent_id": parent_id,
                    "page_number": page_number,
                    "source": parent.metadata.get("source", "")
                }

                child_chunks.append(child)

        return {
            "parent"   : parent_chunks,
            "children" : child_chunks
        }