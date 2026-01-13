import os
import logging
from PIL import Image
from dotenv import load_dotenv
from multimodal_ingest import process_and_store, CLIP_MODEL_NAME

# Load environment variables from .env file
load_dotenv()
MANUAL_API_KEY = "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"

if MANUAL_API_KEY:
    os.environ["HUGGINGFACE_API_KEY"] = MANUAL_API_KEY
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Creates sample text and image files for testing."""
    # Create a sample text file
    text_file = "C://Users//adity//Projects_of_Aditya//Working//UBQ//temp_data//sample_test_note.txt"
    with open(text_file, "w") as f:
        f.write("This is a test medical note. The patient shows signs of rapid recovery after the procedure.")
    
    # Create a sample image file (a simple red square)
    image_file = "C://Users//adity//Projects_of_Aditya//Working//UBQ//temp_data//sample_test_image.png"
    img = Image.new('RGB', (100, 100), color='red')
    img.save(image_file)
    
    return os.path.abspath(text_file), os.path.abspath(image_file)

def main():
    if not os.getenv("HUGGINGFACE_API_KEY"):
        api_key = input("HUGGINGFACE_API_KEY not found. Please enter it now: ").strip()
        if api_key:
            os.environ["HUGGINGFACE_API_KEY"] = api_key
        else:
            logger.warning("WARNING: HUGGINGFACE_API_KEY is not set. API calls will likely fail.")

    logger.info("Creating sample data...")
    text_path, image_path = create_sample_data()

    try:
        logger.info(f"Testing ingestion for text: {text_path}")
        process_and_store(text_path)
        
        logger.info(f"Testing ingestion for image: {image_path} using {CLIP_MODEL_NAME}")
        process_and_store(image_path)
        
        logger.info("Ingestion test completed successfully.")
        
    except Exception as e:
        logger.error(f"Ingestion test failed: {e}")
    finally:
        # Cleanup
        if os.path.exists(text_path):
            os.remove(text_path)
        if os.path.exists(image_path):
            os.remove(image_path)
        logger.info("Cleaned up sample files.")

if __name__ == "__main__":
    main()