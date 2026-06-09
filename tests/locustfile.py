import random
from locust import HttpUser, task, between, events

# Standard queries for load testing the RAG pipeline
TEST_QUERIES = [
    "how to implement CORS?",
    "give me an example of Dependency Injection",
    "what are Multiple Models in FastAPI?",
    "how to use pydantic's .model_dump() method",
    "give me an example of application structure",
    "how to convert a data type to JSON?",
    "How to create a middleware?",
    "What happens when a client tries to access a non-existent resource?",
    "How to override the httpexception error handling?"
]

# Standard search modes supported by our FastAPI/Qdrant retriever
SEARCH_MODES = ["dense", "sparse", "hybrid", "hybrid_rerank"]

class RAGUser(HttpUser):
    """
    Simulated user executing concurrent queries against the RAG FastAPI application.
    """
    # Wait time between tasks simulating user think time (1 to 3 seconds)
    wait_time = between(1.0, 3.0)

    def on_start(self):
        """
        Executed when a simulated user starts.
        """
        print(f"Starting simulated user session: {self}")

    def on_stop(self):
        """
        Executed when a simulated user exits.
        """
        print(f"Ending simulated user session: {self}")

    @task(4)
    def post_chat_query(self):
        """
        Simulate a user posting a question regarding FastAPI to the RAG chat service.
        Randomly picks a query and a search mode to exercise the complete retrieval pipeline.
        """
        query = random.choice(TEST_QUERIES)
        mode = random.choice(SEARCH_MODES)
        
        payload = {
            "message": query,
            "mode": mode
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Capture performance metrics and assert status code
        with self.client.post("/chat", json=payload, headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    res_json = response.json()
                    # Basic validation of response schema
                    if "response" in res_json and "retrieved_documents" in res_json:
                        response.success()
                    else:
                        response.failure("Response missing required schema fields.")
                except Exception as e:
                    response.failure(f"Failed to parse response JSON: {e}")
            else:
                response.failure(f"Chat request failed with status code {response.status_code}: {response.text}")

    @task(1)
    def check_health(self):
        """
        Simulate health check request.
        """
        with self.client.get("/health", catch_response=True) as response:
            # Note: Both 200 (healthy) and 503 (unhealthy database connection)
            # are expected operational status returns and shouldn't log network errors
            if response.status_code == 200:
                response.success()
            elif response.status_code == 503:
                # 503 is returned dynamically on database failures, which is an expected state indicator
                try:
                    res_json = response.json()
                    if res_json.get("qdrant_connected") is False:
                        response.success() # Gracefully register expected status
                    else:
                        response.failure("Health check returned 503 but Qdrant connected is not False.")
                except Exception:
                    response.failure("Health check returned 503 but JSON payload is invalid.")
            else:
                response.failure(f"Health check failed with unexpected status code: {response.status_code}")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print(f"--- Locust Load Test Initiating against target: {environment.host} ---")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("--- Locust Load Test Completed ---")
