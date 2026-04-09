from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.Features.LangChainAPI.LangChainDTO import ChunkResponse
from src.SharedKernel.utils.yamlenv import load_env_yaml
from src.SharedKernel.base.Metrics import Metrics

config = load_env_yaml()

law_separators=[
    r"\nĐiều\s+\d+",
    r"\n\d+\.\s",
    r"\n[a-z]\)\s",
    "\n\n",
    "\n"
]

class Process:
    def __init__(self) -> None:
        ...

    def _split_docs(self, text: str):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.llm.splitter.PaC.parent_chunk_size,
            chunk_overlap=config.llm.splitter.PaC.parent_chunk_overlap,
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

    def split_PaC(self, docs: List[Document]):
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

            full_text += text + "\n"
            cursor = end + 1

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.llm.splitter.PaC.parent_chunk_size,
            chunk_overlap=config.llm.splitter.PaC.parent_chunk_overlap,
            separators=law_separators,
            add_start_index=True
        )

        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.llm.splitter.PaC.child_chunk_overlap,
            chunk_overlap=config.llm.splitter.PaC.child_chunk_overlap,
            separators=law_separators,
            add_start_index=True
        )

        parent_docs = parent_splitter.create_documents([full_text])

        parent_chunks = []
        child_chunks = []

        for idx, parent in enumerate(parent_docs):

            start_index = parent.metadata["start_index"]
            end_index = start_index + len(parent.page_content)

            pages = set()

            for p in page_map:
                if not (end_index < p["start"] or start_index > p["end"]):
                    pages.add(p["page_number"])

            pages = sorted(list(pages)) if pages else [1]
            pages_str = [str(p) for p in pages]

            parent_id = f"parent_docs:{docs[0].metadata['source']}:{idx}"

            parent.metadata = {
                "parent_id": parent_id,
                "pages": pages_str,
                "page_span": f"{pages[0]}-{pages[-1]}",
                "source": docs[0].metadata["source"]
            }

            parent_chunks.append(parent)

            children = child_splitter.split_documents([parent])

            for i, child in enumerate(children):
                child.metadata = {
                    "parent_id": parent_id,
                    "pages": pages_str,
                    "page_span": parent.metadata["page_span"],
                    "source": parent.metadata["source"],
                    "chunk_index": i,
                    "total_chunks": len(children),
                    "content_length": len(child.page_content)
                }

                child_chunks.append(child)

        return {
            "parent": parent_chunks,
            "children": child_chunks
        }
        ...