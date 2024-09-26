import os

import logging

import pathway as pw

from pathway.xpacks.llm.document_store import DocumentStore
from pathway.xpacks.llm.servers import QASummaryRestServer

from fastapi import FastAPI, HTTPException

from dotenv import load_dotenv

from pydantic import BaseModel

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from typing import List

from sentence_transformers import SentenceTransformer, util
import numpy as np

pw.set_license_key('239239-349AF7-177B05-8B461E-8D449B-V3')

# Loading environment variables
load_dotenv()

# Retrieving API key and model name from environment variables
API_KEY = os.getenv('API_KEY')
MODEL_NAME = os.getenv('MODEL_NAME')

# Checking if required variables are set
if API_KEY is None:
    raise ValueError("API_KEY must be set in the environment variables.")
if MODEL_NAME is None:
    raise ValueError("MODEL_NAME must be set in the environment variables.")

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = FastAPI()

class DocumentIndexingApp(BaseModel):
    document_store: DocumentStore
    host: str = "0.0.0.0"
    port: int = 8000
    with_cache: bool = True
    terminate_on_error: bool = False
    model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')

    def run(self) -> None:
        logging.info("App is ready to accept requests.")

    def detect_pii(self, text: str) -> bool:
        analyzer = AnalyzerEngine()
        results: List[RecognizerResult] = analyzer.analyze(text=text, language='en')
        return any(result.score > 0.5 for result in results)

    def handle_user_input(self, user_input: str, max_results: int = 5):
        if self.detect_pii(user_input):
            logging.warning("PII detected in input. Input could not be processed.")
            return "I'm sorry, but I can't process sensitive information."
        
        try:
            # Get all document texts from the document store
            document_texts = [doc.text for doc in self.document_store.get_all_documents()]
            if not document_texts:
                return "I'm sorry, couldn't find relevant information!"

            # Creating embeddings for user input and documents
            user_embedding = self.model.encode(user_input, convert_to_tensor=True)
            document_embeddings = self.model.encode(document_texts, convert_to_tensor=True)

            # Calculating cosine similarity
            cosine_scores = util.pytorch_cos_sim(user_embedding, document_embeddings)[0]
            top_results_idx = np.argsort(cosine_scores.numpy())[-max_results:][::-1]  # Getting top five indices

            # preparing the response
            responses = []
            for idx in top_results_idx:
                score = cosine_scores[idx].item()  # Get similarity score
                doc = self.document_store.get_document_by_index(idx)  # Retrieve the document
                responses.append(f"Source: {doc.source}\nContent: {doc.text}")

            return "\n\n".join(responses)
        
        except Exception as e:
            logging.error(f"Error querying document store: {e}")
            return "An error occurred while searching for information."

    
# Instantiating the app
document_indexing_app = DocumentIndexingApp(document_store=DocumentStore())

@app.post("/feedback")
async def feedback(user_id: str, input_text: str, rating: int):
    # Store or process feedback here
    logging.info(f"User {user_id} rated input '{input_text}' with a score of {rating}.")
    return {"message": "Thank you for your feedback!"}

@app.post("/query")
async def query(input_text: str):
    try:
        response = document_indexing_app.handle_user_input(input_text)
        return {"response": response}
    except Exception as e:
        logging.error(f"Error processing input: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    document_indexing_app.run()