import os
import openai
import psycopg # Use the newer psycopg library
from psycopg.rows import dict_row # To get results as dictionaries
from dotenv import load_dotenv
import numpy as np # Used for embedding calculations
# Removed socket import as detailed connection debugging is removed

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# OpenAI Models Configuration
EMBEDDING_MODEL = "text-embedding-ada-002"
# Choose a chat model (e.g., "gpt-3.5-turbo", "gpt-4")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")

# --- Local PostreSQL Connection Details ---
# Reads connection details from .env file or uses defaults
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "embedding_db")
DB_USER = os.getenv("DB_USER", os.getenv("USER")) # Defaults to system user
DB_PASSWORD = os.getenv("DB_PASSWORD", "") # Default to empty password

# Construct the connection parameters dictionary
conn_params = {
    "host": DB_HOST,
    "port": DB_PORT,
    "dbname": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "row_factory": dict_row # Get results as dictionaries
}

# Initialize OpenAI Client
try:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
except TypeError:
    # Fallback for older openai versions
    openai.api_key = OPENAI_API_KEY
    client = openai
    
# --- Sample Data ---
# This is the text we want to store and search within
document_text = """
The Solar System is the gravitionally bound system of the Sun and the objects that orbit it.
It formed 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud.
The vast majority of the system;s mass is in the Sun, with most of the remaining mass contained in the planet Jupiter.
The four inner system planets-Mercury, Venus, Earth and Mars-are terrestial planets, being composed primarily of rock and metal.
The four outer system planets--Jupiter, Saturn, Uranus and Neptune-are giant planets, being substantially more massive than the terrestrials.
Jupiter and Saturn are gas giants, being composed mainly of hydrogen and helium; Uranus and Neptune are ice giants, being composed mostly of substrances with relatively high melting points.
All eight planets have almost circular orbits that lie within a nearly flat disc called the ecliptic.
"""
# Simple splitting by paragraph
text_chunks = [chunk.strip() for chunk in document_text.strip().split('\n') if chunk.strip()]

# This is the question we want to ask.
user_question = "Which planets are made mostly of gas?"

# --- Core Functions ---

def get_db.connection():
    """Establishes a connection to the local PostreSQL database."""
    print(f"\nAttempting to connecto to database '{conn_params['dbname']}' on {conn_params['host']}:{conn_params['port']}...")
    try:
        # Connect using connection parameters dictionary
        conn = psycopg.connect(**conn_params)
        print("Database connection established successfully.")
        return conn
    except Exception as e: # Catch generic exception for connection errors
        print(f"\n--- DATABASE CONNECTION FAILED ---")
        print(f"Error: {e}")
        print("Please ensure PostreSQL is running and connection details are correct.")
        return None


def get_openai_embedding(text_to_embed: str, model=EMBEDDING_MODEL):
    """
    Generates an embedding for the given text using OpenAI's API.
    
    Args:
        text_to_embed: The string of text to embed.
        model: The OpenAI embedding model to use.

    Returns:
        A list of floats representing the embedding vector, or None if an error occurs.
    """
    print(f"    Generating embedding for: '{text_to_embed[:60]}...'") #Slightly shortened print
    try:
        if hasattr(client, 'embeddings'):
            response = client.embeddings.create(input=[text_to_embed], model=model)
            embedding = response.data[0].embedding
        else:
            response = openai.Embedding.create(input=[text_to_embed], model=model)
            embedding = response['data'][0]['embedding']
        # print(f"  Embedding generated (dimension: {len(embedding)})") # Optional: Can uncomment if needed
        # Return as a list
        return list(embedding)
    except Exception as e:
        print(f"    Error generating OpenAI embedding: {e}")
        return None

def clear_documents_table(conn):
    """Clears all entries from the documents table."""
    print("\nClearing existing documents table...")
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents;")
            conn.commit() # Commit the transaction
            print("Table cleared.")
            return True
        except Exception as e:
            print(f"Error clearing table: {e}")
            conn.rollback() # Rollback on error
            return False


def store_chunks_in_db(conn, chunks: list[str]):
    """
    Generates embeddings for text chunks and stores them in the local DB.

    Args:
        conn: Active database connection.
        chunks: A list of text strings (paragraphs, sentences, etc.).
    
    Returns:
        True if all chunks were stored successfully, False otherwise.
    """
    print("\n--- Storing Document Chunks ---")
    success = True
    for i, chunk in enumerate(chunks):
        print(f"\nProcessing chunk {i+1}/{len(chunks)}:")
        embedding_list = get_openai_embedding(chunk)
        if embedding_list: # Check if embedding was successful
            try:
                #Use INSERT ... ON CONFLICT (Upsert)
                sql = """
                    INSERT INTO documents (content, embedding)
                    VALUES (%s, %s)
                    ON CONFLICT (content) DO UPDATE
                    SET embedding = EXCLUDED.embedding;
                """
                with conn.cursor() as cur:
                    # Pass the Python list directly
                    cur.execute(sql, (chunk, embedding_list))
                conn.commit() # Commit after each successful insert/upsert
                print(f"    Stored chunk {i+1} in DB.")
            except Exception as e:
                print(f"    Error storing chunk {i+1} in DB: {e}")
                conn.rollback() # Rollback the failed transaction
                success = False # Mark as failed
                break # Stop processing if one chunk fails
        else:
            print(f"    Skipping chunk {i+1} due to embedding error.")
            success = False # Mark as failed if embedding fails

        if success:
            print("\n--- Finished storing chunks ---")
        else:
            print("\n--- Storing chunks finished with errors ---")
        return success


def find_similar_chunks(conn, query: str, match_count=3, match_threshold=0.75):
    """
    Finds text chunks in the local DB semantically similar to the query.

    Args:
        conn: Active database connection.
        query: The user's question or search term.
        match_count: The maximum number of similar chunks to return.
        match_threshold: The minimum similarity score for a chunk to be considered a match.

    Returns:
        A list of dictionaries, each containing the 'content' and 'similarity'
        of a matching chunk, or None if an error occurs during the process.
    """
    print("\n--- Finding Similar Chunks (Retrieval Step) ---")
    print(f"Original Query: '{query}'")

    # 1. Generate embedding for the user's query
    query_embedding_list = get_openai_embedding(query)
    if not query_embedding_list:
        print("Could not generate embedding for the query.")
        return None
    
    # 2. Query Database using pgvector operator and type cast
    print(f"\n    Querying DB for top {match_count} matches (similarity > {match_threshold})...")
    try:
        # Use the cosine distance operator '<=>' from pgvector
        # Similarity = 1 - distance
        # Add explicit ::vector cast to the parameter placeholder
        sql = """
            SELECT id, content, 1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            WHERE 1 - (embedding <=> %s::vector) > %s
            ORDER BY similarity DESC
            LIMIT %s;
        """
        with conn.cursor() as cur:
            # Pass the Python list directly for the parameter
            cur.execute(sql, (query_embedding_list, query_embedding_list, match_threshold, match_count))
            results = cur.fetchall() # Fetch all matching rows

        if results:
            print(f"    Found {len(results)} relevant chunks.")
            return results # Returns a list of dictionaries
        else:
            print("    No sufficiently similar chunks found.")
            return [] # Return empty list if no matches found
    except Exception as e:
        print(f"    Error querying database: {e}")
        return None # Return None to indicate an error occured

def generate_augmented_answer(query: str, context_chunks: list[dict], model=CHAT_MODEL)
    """
    Generates an answer using an LLM, augmented by the retrieved context chunks.

    Args:
        query: The original user question.
        context_chunks: A list of dictionaries, where each dict contains 'content' and 'similarity'.
        model: The OpenAI chat model to use.

    Returns:
        The LLM-generated answer as a string, or None if an error occurs.
    """
    print("\n--- Generating Answer using LLM (Generation Step) ---")

    # Combine the content of the chunks into a single context string
    context_str = "\n\n".join([chunk['content'] for chunk in context_chunks])

    # Create the prompt for the LLM
    system_prompt = "You are a helpfule assistant. Answer the user's question based *only* on the provided context. If the context doesn't contain the answer, say you don't know"
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"

    print(f"    Using model: {model}")
    # print(f"    System Prompt: {system_prompt}") # Optional: print propts for debugging
    # print(f"    User Prompt: {user_prompt[:200]}...") # Optional: print propts for debugging

    try:
        # Use the chat completions endpoint
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "context": system_prompt},
                {"role": "user", "context": user_prompt}
            ],
            temperature=0.2 # Lower temperature for more factual answers based on context
        )
        # Extract the answer from the respons
        answer = response.choices[0].message.content.strip()
        print("    LLM generated answer successfully.")
        return answer
    except Exception as e:
        print(f"    Error calling OpenAI ChatCompletion API: {e}")
        return None

#--- Main Execution ---
if __name__ == "__main__":
    """
    TODO:
    --- Retrieved Context Chunks (similar_chunks variable) --- section, return JSON list
    add generate_augmented answer to main execution
    """
    print("Starting Embeddings Demo (RAG with Local PostgreSQL)...")

    # Establish database connection
    db_conn = get_db_connection()

    # Proceed only if the database connection was successful
    if db_conn:
        storage_successful = True # Assume success initially
        clearing_successful = True

        try: # Wrap database operations in a try block
            # Ask user if they want to clear and re-index
            confirmation = input("Do you want to clear existing documents and re-index? (yes/no): ").lower()
            if confirmation == 'yes':
                clearing_successful = clear_documents_table(db_conn)
                # Only store if clearing was successful
                if clearing_successful:
                    storage_successful = store_chunks_in_db(db_conn, text_chunks)
                else:
                    print("Skipping storing chunks due to table clearing error.")
                    storage_successful = False # Mark storage as failed if clearing failed
            else:
                print("\nSkipping indexing. Assuming documents are already in DB.")

            # Proceed to search only if storage was successful (or skipped)
            if storage_successful:
                similar_chunks = find_similar_chunks(db_conn, user_question)

                print("\n--- Search Results ---")
                # Check if the search function returned results (not None)
                if similar_chunks:
                    for i, chunk_info in enumerate(similar_chunks):
                        print(f"\nMatch {i+1} (Similarity: {chunk_info['similarity']:.4f}):")
                        print(f"  Content: {chunk_info['content']}")
                    else: # Search was successful, but no matches found
                        print("No relevant information found based on the similarity threshold.")
                else: # An error occurred during the search function itself
                    print("An error occurred during the search process.")
            else:
                # Explain why search is skipped if storage/clearing failed
                print("\nSkipping search becasue data preparation failed.")

        except Exception as main_e:
            # Catch any unexpected errors during the main execution flow
            print(f"\nAn unexpected error occurred in the main execution: {main_e}")
        finally:
            # Ensure the database connection is alsways closed if it was opened
            print("\nClosing database connection.")
            db_conn.close()
    else:
        print("\nExiting script due to database connection failure.")

    print("\nDemo finished.") # embedding_rag_demo.py finishes at line 282, some comments were removed from embedding_demo.py      
