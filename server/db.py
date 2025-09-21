import os
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class QdrantPDFUploader:
    def __init__(self, api_key: str, cloud_url: str, collection_name: str = "pdf_documents"):
        """
        Initialize the PDF uploader with Qdrant Cloud connection
        
        Args:
            api_key: Qdrant Cloud API key
            cloud_url: Qdrant Cloud cluster URL
            collection_name: Name of the collection to store PDFs
        """
        self.client = QdrantClient(
            url=cloud_url,
            api_key=api_key,
        )
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dimensional embeddings
        
    def create_collection(self, recreate_if_exists: bool = False):
        """Create or recreate the collection in Qdrant"""
        if recreate_if_exists:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # all-MiniLM-L6-v2 embedding size
                    distance=models.Distance.COSINE
                )
            )
            print(f"Collection '{self.collection_name}' recreated successfully!")
            return
        
        try:
            self.client.get_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' already exists.")
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Collection '{self.collection_name}' created successfully!")
    
    def extract_text_from_pdf(self, file_path: str) -> List[Dict[str, str]]:
        """Extract text from PDF with page numbers"""
        reader = PdfReader(file_path)
        documents = []
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text.strip():  # Only add non-empty pages
                documents.append({
                    "text": text,
                    "page": page_num,
                    "source": os.path.basename(file_path)
                })
        
        return documents
    
    def process_pdf_folder(self, folder_path: str, batch_size: int = 32):
        """Process all PDFs in a folder and upload to Qdrant"""
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder '{folder_path}' does not exist")
        
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
        if not pdf_files:
            print(f"No PDF files found in '{folder_path}'")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        all_documents = []
        for pdf_file in tqdm(pdf_files, desc="Extracting PDF text"):
            file_path = os.path.join(folder_path, pdf_file)
            all_documents.extend(self.extract_text_from_pdf(file_path))
        
        # Process in batches
        for i in tqdm(range(0, len(all_documents), batch_size), desc="Uploading to Qdrant"):
            batch = all_documents[i:i + batch_size]
            texts = [doc["text"] for doc in batch]
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            
            # Prepare payloads
            payloads = [
                {
                    "text": doc["text"],
                    "page": doc["page"],
                    "source": doc["source"],
                    "full_text_length": len(doc["text"])
                }
                for doc in batch
            ]
            
            # Upload to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=models.Batch(
                    ids=list(range(i, i + len(batch))),
                    vectors=embeddings.tolist(),
                    payloads=payloads
                )
            )
        
        print(f"Successfully uploaded {len(all_documents)} document chunks from {len(pdf_files)} PDFs")

if __name__ == "__main__":
    # Configuration - replace with your details
    QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzUxMTQzNjgwfQ.s54bLE4-tQn0C4t-nP-tMggZl5RJsk3MlmfCjdbLfow"
    QDRANT_CLOUD_URL = "https://b46a2c8f-2e88-4883-adc7-50ea89e775e3.us-west-1-0.aws.cloud.qdrant.io:6333"
    PDF_FOLDER_PATH = "./pdfs"  # Folder containing your PDFs
    
    # Initialize uploader
    uploader = QdrantPDFUploader(
        api_key=QDRANT_API_KEY,
        cloud_url=QDRANT_CLOUD_URL,
        collection_name="pdf_documents"
    )
    
    # Create collection (set recreate_if_exists=True if you want to start fresh)
    uploader.create_collection(recreate_if_exists=True)
    
    # Process and upload PDFs
    uploader.process_pdf_folder(PDF_FOLDER_PATH)



'''
for each document: {
        file name
        ---- text --- thousands of words
        page number
        text length,
        "document title" -- generate_title()
    } -> [00,3430034] (vectors)

LLMs Multi Modal -- Audio/images (accept)
query X document (30 words) -> vector


Web2:
** frontend

Frontend Basics, 
(longer long time, generalizable)
Frontend --> SWE / Backend (streaming, async, sync) -->  

---> AI/ML
--> Pipelines/Engineering Side
    - AI Model 
    - Add Software
    - **Kubernetes** attach -- **infrastructure** is very very important
    - Scaleups
    - Workflow

--> Core Research
    - heavy research --- youtube videos pytorch, openai ex-researcher YT videos

(naukri chahiye, paise chahiye)
If someone says Blockchain Company ----
--- Software (web2)
--- Full Stack Dev (web2)
--- **Contracts --- Solidity, Rust, Node JS (backend)**
'''
