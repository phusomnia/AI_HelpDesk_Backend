from collections import defaultdict
import json
import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional
from Features.LangChainAPI.RedisVectorRepo import RedisVectorRepo
from Features.LangChainAPI.LangChainDTO import Callback, ChunkResponse, SplitRequest
from SharedKernel.ai.AIConfig import AIConfigFactory
from SharedKernel.persistence.Decorators import Service
from SharedKernel.utils.yamlenv import load_env_yaml
from langchain_community.document_loaders import PlaywrightURLLoader, UnstructuredPDFLoader
from fastapi import UploadFile
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

config = load_env_yaml()

# @Service
class Synthesis:
    def __init__(self, ai_factory: AIConfigFactory, provider, callbacks) -> None:
        self.provider = provider
        self.callbacks = callbacks or {}
        self.ai_factory = ai_factory
        self.loader = Loader(self.provider, self.callbacks)
        self.process = Process(self.provider, self.callbacks)
        self.vs_repo = RedisVectorRepo(self.ai_factory)
    ...

    async def call_rag(self, query: str):
        result = await self.syn.vs_repo.manual_rag(
            query=query, 
            provider=self.provider
        )
        return result

    #
    # INGEST
    #
    async def ingest_pdf(self, file: UploadFile):
        await self.vs_repo.delete_documents_by_metadata(
            {
                "source": file.filename
            }
        )

        docs = self.loader.load_pdf(file)
        if not docs:
            print("No documents loaded")
            return
        print(f"Loaded {len(docs)} elements")
        all_chunks = self.process.split_by_page(docs)
        # await self.batch_ingest_with_metadata(file, all_chunks)
    ...

    async def batch_ingest_with_metadata(self, file: UploadFile, chunks: List[Any]):
        documents = []
        current_time = int(time.time())

        for i, chunk in enumerate(chunks):
            metadata = chunk.metadata if chunk.metadata else {}

            metadata.update({
                "source": file.filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "timestamp": current_time,
                "content_length": len(chunk.content),
                "type": file.content_type
            })
        
            documents.append(Document(
                page_content=chunk.content,
                metadata=metadata
            ))

        print(f"Saving {len(documents)} documents to vector store")
    
        if documents:
            print(f"Sample metadata: {documents[0].metadata}")
        
        await self.abatch_add_documents(documents)
    ...

###
class Loader:
    def __init__(self, provider, callbacks) -> None:
        self.provider = provider
        self.callbacks = callbacks or {}
    ...
    
    def load_webpage(self, target_url: str):
        loader = PlaywrightURLLoader(
            urls=[target_url], 
            headless=True
        )

        documents = loader.load()
        
        if not documents:
            print(f"Cảnh báo: Không lấy được nội dung tại {target_url}")
            return []
            
        return documents[0].page_content
    ...

    def load_pdf(self, file: UploadFile):
        if file.filename and not file.filename.lower().endswith('.pdf'):
            raise ValueError(f"File {file.filename} is not a PDF file")

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        loader = UnstructuredPDFLoader(
            temp_path, 
            mode="elements"
        )
        documents = loader.load()
        
        os.unlink(temp_path)

        return documents
    ...
...

###
class Process:
    def __init__(self,         
        provider: BaseChatModel, 
        callbacks: Callback
    ) -> None:
        self.provider = provider
        self.callbacks = callbacks or {}
    ...

    def _split_docs(self, split_req: SplitRequest):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=split_req.chunk_size,
            chunk_overlap=split_req.chunk_overlap,
            separators=split_req.separators,
            length_function=len,
            is_separator_regex=False
        )

        docs = text_splitter.create_documents([split_req.text])

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

    def split_by_page(self, docs: List[Any]):
        page_groups = {}
        for doc in docs:
            page_num = doc.metadata.get('page_number', 0)
            page_groups.setdefault(page_num, []).append(doc)
        
        print(f"Grouped into {len(page_groups)} pages")

        all_chunks = []
        for page_num, page_docs in sorted(page_groups.items()):
            print(f"Processing page {page_num} with {len(page_docs)} elements")
            
            page_text = "\n\n".join(doc.page_content for doc in page_docs)
            
            req_split = SplitRequest()
            req_split.text = page_text
            req_split.chunk_size = config.splitter.chunk_size
            req_split.chunk_overlap = config.splitter.chunk_overlap
            req_split.separators = config.splitter.separators
            
            page_chunks = self._split_docs(req_split)
            
            print(f" -> Created {len(page_chunks)} chunks")
            
            for chunk in page_chunks:
                if not chunk.metadata:
                    chunk.metadata = {}

                chunk.metadata.update({
                    'page_number': page_num,
                    'total_elements': len(page_docs)
                })

                print(f"  Chunk {chunk.index}: {len(chunk.content)} chars")
                print(f"   Page: {chunk.metadata['page_number']}")
            all_chunks.extend(page_chunks)
        print(f"Total chunks created: {len(all_chunks)}")

        return all_chunks
        ...
###  