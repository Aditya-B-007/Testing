import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingFunction
from sentence_transformers import SentenceTransformer
import logging
from typing import Dict, Any, Optional
CHROMA_PATH = "C://Users//adity//Projects_of_Aditya//Working//UBQ//chroma_db"
COLLECTION_NAME = "patient_data"
TEXT_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL_NAME = "clip-ViT-B-32"
IMAGE_COLLECTION_NAME = "multimodal_docs_image"

# Load environment variables
load_dotenv()
MANUAL_API_KEY = "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"  
if MANUAL_API_KEY:
    os.environ["HUGGINGFACE_API_KEY"] = MANUAL_API_KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def retrieve_relevant_context(user_query: str, collection_name: Optional[str] = None, n_results: int = 3) -> Optional[Dict[str, Any]]:
    """
    Searches the ChromaDB collection for the most similar documents to the user query.
    """
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        hf_ef = HuggingFaceEmbeddingFunction(
            api_key=os.getenv("HUGGINGFACE_API_KEY"),
            model_name=TEXT_MODEL_PATH
        )

        if collection_name:
            target_collections = [client.get_collection(name=collection_name, embedding_function=hf_ef)]
        else:
            collections = client.list_collections()
            if not collections:
                logger.warning("No collections found in the database.")
                return None
            target_collections = [client.get_collection(name=c.name, embedding_function=hf_ef) for c in collections]

        aggregated_results = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        for collection in target_collections:
            if collection.count() == 0:
                continue
            
            try:
                logger.info(f"Searching collection: '{collection.name}'")
                results = collection.query(
                    query_texts=[user_query],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"]
                )
                aggregated_results["ids"][0].extend(results["ids"][0])
                aggregated_results["documents"][0].extend(results["documents"][0])
                aggregated_results["metadatas"][0].extend(results["metadatas"][0])
                aggregated_results["distances"][0].extend(results["distances"][0])
            except Exception as e:
                logger.error(f"Could not search collection '{collection.name}': {e}")
        if aggregated_results["ids"][0]:
            combined = list(zip(
                aggregated_results["ids"][0],
                aggregated_results["documents"][0],
                aggregated_results["metadatas"][0],
                aggregated_results["distances"][0]
            ))
            combined.sort(key=lambda x: x[3])
            top_combined = combined[:n_results]
            return {
                "ids": [[x[0] for x in top_combined]],
                "documents": [[x[1] for x in top_combined]],
                "metadatas": [[x[2] for x in top_combined]],
                "distances": [[x[3] for x in top_combined]]
            }
        
        return None

    except Exception as e:
        logger.error(f"An error occurred while accessing ChromaDB: {e}")
        return None

def retrieve_relevant_images(query_text: str, n_results: int = 3) -> Optional[Dict[str, Any]]:
    """
    Searches the ChromaDB image collection using the local CLIP model.
    """
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        logger.info(f"Loading CLIP model: {CLIP_MODEL_NAME}")
        model = SentenceTransformer(CLIP_MODEL_NAME)

        try:
            collection = client.get_collection(name=IMAGE_COLLECTION_NAME)
        except Exception:
            logger.warning(f"Collection '{IMAGE_COLLECTION_NAME}' not found.")
            return None

        logger.info(f"Searching collection: '{IMAGE_COLLECTION_NAME}'")
        query_embedding = model.encode(query_text).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas", "distances"]
        )
        return results

    except Exception as e:
        logger.error(f"Error searching images: {e}")
        return None

def data_assembly_step(results: Dict[str, Any]):
    """
    Loops through the results and prints a formatted summary for each match.
    """
    if not results or not results.get('ids') or len(results['ids'][0]) == 0:
        print("\n[Data Assembly] No relevant context could be resolved.")
        return

    print("\n" + "="*70)
    print(" DATA ASSEMBLY: SIMILARITY SEARCH SUMMARY")
    print("="*70)

    # ChromaDB returns results as lists of lists (supporting multiple queries)
    # We extract the first (and only) query result set.
    ids = results['ids'][0]
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    distances = results['distances'][0]

    for i in range(len(ids)):
        patient_id = metas[i].get("patient_id", "Unknown ID")
        # Create a snippet of the original medical note (first 150 chars)
        snippet = docs[i][:150].replace("\n", " ") + "..." if len(docs[i]) > 150 else docs[i]
        
        print(f"Match #{i+1} | Similarity Distance: {distances[i]:.4f}")
        print(f"Patient ID : {patient_id}")
        print(f"Note Snippet: {snippet}")
        print("-" * 70)

def display_image_results(results: Dict[str, Any]):
    if not results or not results.get('ids') or len(results['ids'][0]) == 0:
        print("\n[Image Search] No relevant images found.")
        return

    print("\n" + "="*70)
    print(" IMAGE SEARCH SUMMARY")
    print("="*70)

    ids = results['ids'][0]
    metas = results['metadatas'][0]
    distances = results['distances'][0]

    for i in range(len(ids)):
        filename = metas[i].get("filename", "Unknown")
        path = metas[i].get("path", "Unknown")
        print(f"Match #{i+1} | Similarity Distance: {distances[i]:.4f}")
        print(f"Filename: {filename}")
        print(f"Path    : {path}")
        print("-" * 70)

if __name__ == "__main__":
    # Check for API key before running
    if not os.getenv("HUGGINGFACE_API_KEY"):
        api_key = input("HUGGINGFACE_API_KEY not found. Please enter it now: ").strip()
        if api_key:
            os.environ["HUGGINGFACE_API_KEY"] = api_key
        else:
            logger.warning("WARNING: HUGGINGFACE_API_KEY is not set. API calls will likely fail.")

    # Example Query for testing
    query_text = input("Please enter your query....")
    
    search_results = retrieve_relevant_context(query_text)
    data_assembly_step(search_results)

    image_results = retrieve_relevant_images(query_text)
    display_image_results(image_results)