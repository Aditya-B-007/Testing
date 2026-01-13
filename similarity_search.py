import os
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
CHROMA_PATH = "C://Users//adity//Projects_of_Aditya//Working//UBQ//chroma_db"
COLLECTION_NAME = "multimodal_docs"
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL_NAME = "clip-ViT-B-32"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_text_model = None
_clip_model = None

def get_text_model():
    global _text_model
    if _text_model is None:
        logger.info(f"Loading Local Text Model: {TEXT_MODEL_NAME}...")
        _text_model = SentenceTransformer(TEXT_MODEL_NAME, device="cpu")
    return _text_model

def get_clip_model():
    global _clip_model
    if _clip_model is None:
        logger.info(f"Loading Local CLIP Model: {CLIP_MODEL_NAME}...")
        _clip_model = SentenceTransformer(CLIP_MODEL_NAME, device="cpu")
    return _clip_model

def search_collection(query_text: str, mode: str, n_results: int = 3) -> Optional[Dict[str, Any]]:
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        target_name = f"{COLLECTION_NAME}_{mode}"
        try:
            collection = client.get_collection(name=target_name)
        except Exception:
            logger.warning(f"Collection '{target_name}' not found. Have you run ingestion?")
            return None

        if mode == "text":
            model = get_text_model()
            query_embedding = model.encode(query_text, normalize_embeddings=True).tolist()
        else:
            model = get_clip_model()
            query_embedding = model.encode(query_text, normalize_embeddings=True).tolist()

        logger.info(f"Searching {mode.upper()} collection for: '{query_text}'")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        return results

    except Exception as e:
        logger.error(f"Error during {mode} search: {e}")
        return None

def display_results(results: Dict[str, Any], mode: str):
    if not results or not results.get('ids') or len(results['ids'][0]) == 0:
        print(f"\n[{mode.upper()}] No relevant matches found.")
        return

    print(f"\n--- {mode.upper()} MATCHES ---")
    
    ids = results['ids'][0]
    docs = results.get('documents', [[]])[0] 
    metas = results['metadatas'][0]
    distances = results['distances'][0]

    for i in range(len(ids)):
        dist = distances[i]
        meta = metas[i]
        filename = meta.get('filename', 'Unknown')
        patient_id = meta.get('patient_id', 'Unknown')
        
        print(f"Result #{i+1} (Distance: {dist:.4f})")
        print(f"File: {filename} | Patient: {patient_id}")
        if mode == "text":
            if docs and docs[i]:
                snippet = docs[i][:150].replace("\n", " ") + "..."
                print(f"Snippet: {snippet}")
            else:
                print("Snippet: [Content not available in DB]")
        elif mode == "image":
            print(f"Path: {meta.get('path', 'Unknown')}")
            
        print("-" * 50)

def unified_search(query_text: str):
    print("\n" + "="*60)
    print(f" UNIFIED SEARCH REPORT: '{query_text}'")
    print("="*60)
    
    text_results = search_collection(query_text, mode="text")
    display_results(text_results, mode="text")

    image_results = search_collection(query_text, mode="image")
    display_results(image_results, mode="image")

if __name__ == "__main__":
    while True:
        query = input("\nEnter query (or 'q' to quit): ").strip()
        if query.lower() == 'q':
            break
        if query:
            unified_search(query)