import os
from dotenv import load_dotenv
import uuid
import logging
import mimetypes
from typing import Optional, Dict, Any
import chromadb
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from PIL import Image

# --- Configuration ---
CHROMA_PATH = "C://Users//adity//Projects_of_Aditya//Working//UBQ//chroma_db"
COLLECTION_NAME = "multimodal_docs"
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL_NAME = "clip-ViT-B-32"
load_dotenv()
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
    """Returns a connected Chroma client."""
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
    """ChromaDB collections are created lazily or on demand."""
    pass

def process_and_store(file_path: str, patient_id: str = "unknown", hospital_id: str = "unknown"):
    """
    Detects file type, generates embedding, and upserts to ChromaDB.
    """
    client = get_chroma_client()
    if not client:
        logger.error("Skipping file processing: No DB connection.")
        return

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return

    filename = os.path.basename(file_path)
    mime_type, _ = mimetypes.guess_type(file_path)
    
    # Determine file type
    is_image = mime_type and mime_type.startswith("image")
    is_text = mime_type and mime_type.startswith("text")
    
    # Fallback extension check
    if not is_image and not is_text:
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            is_image = True
        elif ext in ['.txt', '.md', '.csv', '.log']:
            is_text = True

    # Generate a deterministic ID based on the file path to prevent duplicates
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, os.path.abspath(file_path)))
    embedding_vector = []
    payload = {
        "filename": filename, 
        "path": os.path.abspath(file_path),
        "patient_id": patient_id,
        "hospital_id": hospital_id
    }

    try:
        if is_text:
            payload["type"] = "text"
            model = get_text_model()
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text_content = f.read()
            embedding = model.encode(text_content).tolist()
            embedding_vector = embedding
            payload["preview"] = text_content[:200]

        elif is_image:
            payload["type"] = "image"
            model = get_clip_model()
            image = Image.open(file_path)
            embedding = model.encode(image).tolist()
            embedding_vector = embedding

        else:
            logger.warning(f"Unsupported file type for: {filename}")
            return
        sub_collection_name = f"{COLLECTION_NAME}_{payload['type']}"
        collection = client.get_or_create_collection(name=sub_collection_name)
        
        collection.upsert(
            ids=[point_id],
            embeddings=[embedding_vector],
            metadatas=[payload]
        )
        logger.info(f"Successfully stored {filename} ({payload['type']}).")

    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")

def process_batch(file_data: list[Any], batch_size: int = 16):
    """
    Processes a list of files in batches for efficient embedding generation on CPU.
    """
    client = get_chroma_client()
    if not client:
        logger.error("Skipping batch processing: No DB connection.")
        return

    text_batch = []
    image_batch = []

    def flush_batch(batch_data, model_getter, collection_suffix, is_image_data=False):
        if not batch_data: return
        
        try:
            model = model_getter()
            # Extract content for embedding (text strings or PIL Images)
            inputs = [item['input'] for item in batch_data]
            
            # Batch encode
            embeddings = model.encode(inputs, batch_size=len(batch_data)).tolist()
            
            ids = [item['id'] for item in batch_data]
            metadatas = [item['metadata'] for item in batch_data]
            
            sub_collection_name = f"{COLLECTION_NAME}_{collection_suffix}"
            collection = client.get_or_create_collection(name=sub_collection_name)
            
            collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
            logger.info(f"Upserted batch of {len(batch_data)} {collection_suffix} documents.")
        except Exception as e:
            logger.error(f"Error flushing {collection_suffix} batch: {e}")

    for item in file_data:
        if isinstance(item, str):
            file_path = item
            patient_id = "unknown"
            hospital_id = "unknown"
        elif isinstance(item, dict):
            file_path = item.get("path")
            patient_id = item.get("patient_id", "unknown")
            hospital_id = item.get("hospital_id", "unknown")
        else:
            continue

        if not file_path or not os.path.exists(file_path):
            continue

        filename = os.path.basename(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        is_image = mime_type and mime_type.startswith("image")
        is_text = mime_type and mime_type.startswith("text")

        if not is_image and not is_text:
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                is_image = True
            elif ext in ['.txt', '.md', '.csv', '.log']:
                is_text = True

        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, os.path.abspath(file_path)))
        payload = {
            "filename": filename, 
            "path": os.path.abspath(file_path),
            "patient_id": patient_id,
            "hospital_id": hospital_id
        }

        try:
            if is_text:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text_content = f.read()
                payload["type"] = "text"
                payload["preview"] = text_content[:200]
                text_batch.append({'id': point_id, 'metadata': payload, 'input': text_content})

            elif is_image:
                image = Image.open(file_path)
                payload["type"] = "image"
                image_batch.append({'id': point_id, 'metadata': payload, 'input': image})
        except Exception as e:
            logger.error(f"Error preparing file {filename}: {e}")
        if len(text_batch) >= batch_size:
            flush_batch(text_batch, get_text_model, "text", is_image_data=False)
            text_batch = []
        if len(image_batch) >= batch_size:
            flush_batch(image_batch, get_clip_model, "image", is_image_data=True)
            image_batch = []
    flush_batch(text_batch, get_text_model, "text", is_image_data=False)
    flush_batch(image_batch, get_clip_model, "image", is_image_data=True)

if __name__ == "__main__":
    # Initialize DB
    init_collection()
    process_and_store("C://Users//adity//Projects_of_Aditya//Working//UBQ//temp_data//patient_note.txt")
    process_and_store("C://Users//adity//Projects_of_Aditya//Working//UBQ//temp_data//xray_sample.png")