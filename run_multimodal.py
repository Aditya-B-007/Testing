import os
import uuid
import logging
import mimetypes
from typing import Optional, Any, List, Dict
import chromadb
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingFunction
from sentence_transformers import SentenceTransformer
from PIL import Image
from dotenv import load_dotenv

# --- Configuration ---
# UPDATE THIS to your actual folder path
DATA_FOLDER = "C://Users//adity//Projects_of_Aditya//Working//UBQ//temp_data"
CHROMA_PATH = "C://Users//adity//Projects_of_Aditya//Working//UBQ//chroma_db"
COLLECTION_NAME = "multimodal_docs"
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL_NAME = "clip-ViT-B-32"

load_dotenv()
# Check for API Key (Optional for local models, but good practice)
if not os.getenv("HUGGINGFACE_API_KEY"):
    os.environ["HUGGINGFACE_API_KEY"] = "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"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --- Global Model Instances (Lazy Loading) ---
_chroma_client: Optional[Any] = None
_text_model: Optional[Any] = None
_clip_model: Optional[Any] = None

def get_chroma_client() -> Optional[Any]:
    global _chroma_client
    if _chroma_client is None:
        try:
            _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
            logger.info(f"Connected to ChromaDB at {CHROMA_PATH}")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            return None
    return _chroma_client

def get_text_model():
    global _text_model
    if _text_model is None:
        logger.info(f"Loading Text Model: {TEXT_MODEL_NAME}")
        _text_model = SentenceTransformer(TEXT_MODEL_NAME, device="cpu")
    return _text_model

def get_clip_model():
    global _clip_model
    if _clip_model is None:
        _clip_model = SentenceTransformer(CLIP_MODEL_NAME, device="cpu")
    return _clip_model

def init_collection():
    """Ensures collections exist."""
    client = get_chroma_client()
    if client:
        client.get_or_create_collection(name=f"{COLLECTION_NAME}_text")
        client.get_or_create_collection(name=f"{COLLECTION_NAME}_image")

# --- CORE ENGINE: The Batch Processor ---
def process_batch(file_data: List[Dict[str, Any]], batch_size: int = 16):
    """
    Processes a list of files in batches.
    """
    client = get_chroma_client()
    if not client:
        return

    text_batch = []
    image_batch = []

    def flush_batch(batch_data, model_getter, collection_suffix, is_image_data=False):
        if not batch_data: return
        
        try:
            model = model_getter()
            inputs = [item['input'] for item in batch_data]
            
            # Generate Embeddings (Added normalize_embeddings=True for better distance scores)
            embeddings = model.encode(inputs, batch_size=len(batch_data), normalize_embeddings=True).tolist()
            
            ids = [item['id'] for item in batch_data]
            metadatas = [item['metadata'] for item in batch_data]
            
            sub_collection_name = f"{COLLECTION_NAME}_{collection_suffix}"
            collection = client.get_or_create_collection(name=sub_collection_name)
            
            if is_image_data:
                # Images: We don't save the PIL Image object as text
                collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
            else:
                # Text: CRITICAL FIX - We MUST save the 'documents' (inputs) to retrieve them later
                collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=inputs)
                
            logger.info(f"Upserted batch of {len(batch_data)} {collection_suffix} documents.")
        except Exception as e:
            logger.error(f"Error flushing {collection_suffix} batch: {e}")

    for item in file_data:
        file_path = item.get("path")
        if not file_path or not os.path.exists(file_path):
            continue

        filename = os.path.basename(file_path)
        
        # Determine Type
        mime_type, _ = mimetypes.guess_type(file_path)
        is_image = mime_type and mime_type.startswith("image")
        is_text = mime_type and mime_type.startswith("text")
        
        if not is_image and not is_text:
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']: is_image = True
            elif ext in ['.txt', '.md', '.csv', '.log']: is_text = True

        # Generate Deterministic ID
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, os.path.abspath(file_path)))
        
        payload = {
            "filename": filename, 
            "path": os.path.abspath(file_path),
            "patient_id": item.get("patient_id", "Unknown"),
            "hospital_id": item.get("hospital_id", "Unknown"),
            "type": "image" if is_image else "text"
        }

        try:
            if is_text:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text_content = f.read()
                # We save a preview in metadata, but full text goes to 'input' for vectorization
                payload["preview"] = text_content[:200]
                text_batch.append({'id': point_id, 'metadata': payload, 'input': text_content})

            elif is_image:
                image = Image.open(file_path)
                image_batch.append({'id': point_id, 'metadata': payload, 'input': image})
        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}")

        # Flush if batch full
        if len(text_batch) >= batch_size:
            flush_batch(text_batch, get_text_model, "text", False)
            text_batch = []
        if len(image_batch) >= batch_size:
            flush_batch(image_batch, get_clip_model, "image", True)
            image_batch = []

    # Final Flush
    flush_batch(text_batch, get_text_model, "text", False)
    flush_batch(image_batch, get_clip_model, "image", True)

# --- SMART RUNNER LOGIC ---
def get_existing_ids():
    """Fetches all IDs currently stored in the database."""
    client = get_chroma_client()
    if not client: return set()
    
    existing_ids = set()
    for suffix in ["text", "image"]:
        try:
            col = client.get_collection(f"{COLLECTION_NAME}_{suffix}")
            results = col.get(include=[]) 
            if results and 'ids' in results:
                existing_ids.update(results['ids'])
        except Exception:
            pass 
    return existing_ids

def run_smart_ingest():
    if not os.path.exists(DATA_FOLDER):
        logger.error(f"Data folder not found: {DATA_FOLDER}")
        return

    init_collection()
    
    logger.info("checking existing database records...")
    existing_ids = get_existing_ids()
    logger.info(f"Database currently holds {len(existing_ids)} records.")

    logger.info(f"Scanning folder: {DATA_FOLDER}")
    files_to_process = []
    skipped_count = 0
    
    for filename in os.listdir(DATA_FOLDER):
        if filename.lower().endswith(('.txt', '.png', '.jpg', '.jpeg')):
            full_path = os.path.join(DATA_FOLDER, filename)
            
            generated_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, os.path.abspath(full_path)))
            
            if generated_id in existing_ids:
                skipped_count += 1
                continue
            
            files_to_process.append({
                "path": full_path,
                "patient_id": "Unknown", 
                "hospital_id": "Unknown"
            })

    if not files_to_process:
        logger.info(f"No new files found. (Skipped {skipped_count} existing files).")
        return

    logger.info(f"Found {len(files_to_process)} NEW files. (Skipped {skipped_count} existing). Starting Ingestion...")
    process_batch(files_to_process)

if __name__ == "__main__":
    run_smart_ingest()