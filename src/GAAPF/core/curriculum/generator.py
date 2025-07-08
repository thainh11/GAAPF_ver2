
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Ensure API keys are available
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

import chromadb
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# Attempt to set Google credentials if the file exists
# This helps in environments where the credentials are not already set
try:
    credential_path = Path(__file__).parent.parent.parent.parent.parent / "google-credentials.json"
    if credential_path.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credential_path)
except Exception as e:
    print(f"Could not set Google credentials: {e}")

CURRICULUM_TEMPLATE = """
You are an expert curriculum designer. Your task is to create a personalized learning plan for a user based on their profile and a specific framework they want to learn.
You will be given a context containing relevant information about the framework, extracted from a knowledge base.

User Profile:
{user_profile}

Framework Context:
{context}

Based on the user's profile and the provided context, generate a structured and comprehensive curriculum. The curriculum should be in JSON format and follow this structure:
{{
  "framework": "Name of the Framework",
  "user_level": "Beginner/Intermediate/Advanced",
  "estimated_duration_hours": <integer>,
  "modules": [
    {{
      "title": "Module 1: Title",
      "description": "Brief overview of the module.",
      "duration_hours": <integer>,
      "topics": [
        {{
          "topic": "Topic 1.1",
          "description": "Detailed explanation of the topic.",
          "resources": [
            "URL or reference to a specific document section from the context"
          ]
        }}
      ]
    }}
  ]
}}

Generate the curriculum for the framework: {framework_name}
"""

QA_TEMPLATE = """
You are an expert assistant.  Use ONLY the information in the
context to answer the user’s question.  If the context does not
contain the answer, simply say you don’t know.

Context:
{context}

Question: {question}
Answer (concise):
"""

class CurriculumGenerator:
    """
    Generates dynamic, personalized curricula using a Vertex AI LLM and a vector database.
    """
    def __init__(self, db_client: chromadb.Client, collection_name: str = "chroma_ltm", project: str = None, location: str = None):
        """
        Initializes the generator with a ChromaDB client.
        """
        self.embedding_function = VertexAIEmbeddings(
            model_name="text-embedding-large-exp-03-07", # Use a stable, 768-dimension model
            project=project,
            location=location
        )
        self._client = db_client
        self._collection_name = collection_name
        
        self._collection = self._client.get_or_create_collection(name=self._collection_name)

        self.vectordb = Chroma(
            client=self._client,
            collection_name=self._collection_name,
            embedding_function=self.embedding_function
        )
        self.llm = ChatVertexAI(
            model_name="gemini-2.5-flash", 
            temperature=0.3,
            project=project,
            location=location
        )
        self.prompt = ChatPromptTemplate.from_template(CURRICULUM_TEMPLATE)

    def generate(self, framework_name: str, user_profile: Dict) -> Dict:
        """
        Generates a curriculum for a given framework and user profile.
        """
        print(f"Generating curriculum for '{framework_name}'...")

        retriever = self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 20, "filter": {"framework": framework_name}}
        )
        
        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(f"Source: {doc.metadata.get('source', 'N/A')}\nContent: {doc.page_content}" for doc in docs)

        chain = (
            {
                "context": itemgetter("framework_name") | retriever | format_docs,
                "user_profile": itemgetter("user_profile"),
                "framework_name": itemgetter("framework_name"),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        profile_str = json.dumps(user_profile, indent=2)
        
        result = chain.invoke({"user_profile": profile_str, "framework_name": framework_name})

        try:
            # Clean the output by removing markdown and stripping whitespace
            if result.strip().startswith("```json"):
                result = result.strip()[7:-3].strip()
            curriculum = json.loads(result)
            print("Successfully generated and parsed curriculum.")
            return curriculum
        except json.JSONDecodeError:
            print("Error: Failed to parse LLM output as JSON.")
            print("Raw output:", result)
            return {"error": "Failed to generate valid curriculum JSON.", "raw_output": result}

    def add_documents(self, documents: List[Document], framework_name: str):
        """
        Adds documents to the vector store with framework-specific metadata.
        """
        for doc in documents:
            doc.metadata["framework"] = framework_name
            self.vectordb.add_documents([doc])
        print(f"Added {len(documents)} documents for framework '{framework_name}'.")

    def answer_question(
        self,
        question: str,
        framework_name: str | None = None,
        k: int = 8,
    ) -> str:
        """
        Free-form Q&A over the vector-store.
        If `framework_name` is given we restrict retrieval to that framework.
        """
        # Build a retriever with or without a framework filter
        filter_ = {"framework": framework_name} if framework_name else None
        retriever = self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "filter": filter_},
        )

        # Fetch top-k relevant chunks
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join(
            f"Source: {d.metadata.get('source','N/A')}\n{d.page_content}"
            for d in docs
        )

        # Fill prompt
        prompt = ChatPromptTemplate.from_template(QA_TEMPLATE).format(
            context=context, question=question
        )

        # Ask Gemini-2.5-Flash
        answer_msg = self.llm.invoke([HumanMessage(content=prompt)])
        return answer_msg.content.strip()

if __name__ == '__main__':
    # Example usage:
    # 1. Initialize the ChromaDB client directly
    db_path = str(Path(__file__).parent.parent.parent.parent.parent / "data" / "framework_cache" / "chroma_db")
    client = chromadb.PersistentClient(path=db_path)

    # Clean up old collection to avoid dimension errors during development
    try:
        print("Attempting to delete old 'chroma_ltm' collection to ensure compatibility...")
        client.delete_collection(name="chroma_ltm")
        print("Successfully deleted existing 'chroma_ltm' collection.")
    except Exception as e:
        print(f"Info: 'chroma_ltm' collection not found or could not be deleted. A new one will be created. Details: {e}")

    # 2. Pass the client to the generator
    generator = CurriculumGenerator(
        db_client=client,
        collection_name="chroma_ltm", # Explicitly pass collection name
        project="gen-lang-client-0305686287",
        location="us-central1"
    )

    # Add sample documents for LangChain
    langchain_docs = [
        Document(
            page_content="A vector store is responsible for storing and managing the vector representations of documents (embeddings). It provides the underlying infrastructure for efficient storage and retrieval of these vectors.",
            metadata={"source": "langchain_docs"}
        ),
        Document(
            page_content="A retriever is a component that fetches the most relevant documents from a vector store based on a user's query. It acts as an intermediary between the user's question and the stored data, implementing the logic for semantic search.",
            metadata={"source": "langchain_docs"}
        ),
        Document(
            page_content="In LangChain, a retriever takes a query as input and returns a list of relevant documents. Vector stores can be used as the backbone of a retriever.",
            metadata={"source": "langchain_docs"}
        )
    ]
    generator.add_documents(langchain_docs, framework_name="langchain")

    # Load an example user profile
    profile_path = Path(__file__).parent.parent.parent.parent.parent / "user_profiles" / "beginner_user_001.json"
    with open(profile_path, 'r') as f:
        test_user_profile = json.load(f)

    # Generate a curriculum
    framework = "langchain"
    generated_curriculum = generator.generate(framework, test_user_profile)

    # Save the output
    if "error" not in generated_curriculum:
        output_dir = Path(__file__).parent.parent.parent.parent.parent / "data" / "curriculums"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"dynamic_curriculum_{framework}.json"
        with open(output_path, 'w') as f:
            json.dump(generated_curriculum, f, indent=2)
        print(f"Saved generated curriculum to {output_path}")

    # New: Test the answer_question method
    print("\n--- Testing Q&A ---")
    question = "What is the difference between retrievers and vector stores?"
    answer = generator.answer_question(question, framework_name="langchain")
    print(f"Question: {question}")
    print(f"Answer: {answer}") 