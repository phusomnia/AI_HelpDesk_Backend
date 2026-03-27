import os
import shutil
import tempfile
from fastapi import UploadFile
from langchain_community.document_loaders import PlaywrightURLLoader, UnstructuredPDFLoader
from langchain_core.documents import Document

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
            mode="paged"
        )
        documents = loader.load()
        
        formatted_docs = []
        for doc in documents:
            metadata = {
                "page_number": doc.metadata.get("page_number", 1),
                "source": file.filename,
                "language": doc.metadata.get("languages", ["unknown"])[0],
                "content_type": file.content_type
            }
            
            formatted_doc = Document(
                page_content=doc.page_content,
                metadata=metadata
            )
            formatted_docs.append(formatted_doc)
        
        os.unlink(temp_path)
        return formatted_docs
        ...