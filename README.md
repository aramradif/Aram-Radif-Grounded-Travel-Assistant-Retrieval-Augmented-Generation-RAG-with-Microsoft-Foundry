# Aram Radif Grounded-Travel-Assistant-Retrieval-Augmented-Generation-RAG-with-Microsoft-Foundry

Develop a RAG-based solution with your own data using Microsoft Foundry

Overview
This project demonstrates how to design and implement a production-ready Retrieval Augmented Generation (RAG) system using:
•	Microsoft Foundry
•	Azure OpenAI (GPT-4o)
•	Azure OpenAI Embeddings (text-embedding-ada-002)
•	Azure AI Search (Vector + Hybrid Search)
•	Python (Azure OpenAI SDK)
The solution builds a grounded travel assistant that answers user questions using proprietary brochure data instead of relying only on pretrained LLM knowledge.
________________________________________
 Problem Statement
Language models can generate fluent answers, but without grounding they:
•	Hallucinate
•	Invent products or services
•	Provide outdated or generic responses
We solved this by implementing:
 Retrieval Augmented Generation (RAG)
 Vector-based Hybrid Search
 Context Injection into Prompt
 Multi-turn Chat Support
________________________________________
🏗️ Architecture
User Query
    ↓
Azure OpenAI (Query Vectorization)
    ↓
Azure AI Search (Hybrid: Vector + Keyword)
    ↓
Top-N Retrieved Documents
    ↓
Prompt Augmentation
    ↓
GPT-4o Completion
    ↓
Grounded Response + Citations
________________________________________
 Repository Structure
rag-grounded-travel-assistant/
│
├── README.md
├── requirements.txt
├── .env.example
├── rag-app.py
└── brochures/ (PDF data source)
________________________________________
 Environment Setup
python -m venv labenv
./labenv/bin/Activate.ps1
pip install -r requirements.txt openai
________________________________________
 Configuration (.env)
AZURE_OPENAI_ENDPOINT=your_openai_endpoint
AZURE_OPENAI_KEY=your_openai_api_key
CHAT_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-ada-002

AZURE_SEARCH_ENDPOINT=your_search_endpoint
AZURE_SEARCH_KEY=your_search_api_key
AZURE_SEARCH_INDEX=brochures-index
________________________________________
 Core Implementation – RAG Client App
 Keyword-Based RAG
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

chat_client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY")
)

prompt = [
    {"role": "system", "content": "You are a helpful travel assistant that answers only using provided sources."}
]

while True:
    user_input = input("Ask a travel question (or type quit): ")
    if user_input.lower() == "quit":
        break

    prompt.append({"role": "user", "content": user_input})

    rag_params = {
        "data_sources": [
            {
                "type": "azure_search",
                "parameters": {
                    "endpoint": os.getenv("AZURE_SEARCH_ENDPOINT"),
                    "index_name": os.getenv("AZURE_SEARCH_INDEX"),
                    "authentication": {
                        "type": "api_key",
                        "key": os.getenv("AZURE_SEARCH_KEY"),
                    }
                }
            }
        ]
    }

    response = chat_client.chat.completions.create(
        model=os.getenv("CHAT_MODEL"),
        messages=prompt,
        extra_body=rag_params
    )

    answer = response.choices[0].message.content
    print("\nResponse:\n", answer)

    prompt.append({"role": "assistant", "content": answer})
________________________________________
 Vector-Based RAG (Semantic Search)
rag_params = {
    "data_sources": [
        {
            "type": "azure_search",
            "parameters": {
                "endpoint": os.getenv("AZURE_SEARCH_ENDPOINT"),
                "index_name": os.getenv("AZURE_SEARCH_INDEX"),
                "authentication": {
                    "type": "api_key",
                    "key": os.getenv("AZURE_SEARCH_KEY"),
                },
                "query_type": "vector",
                "embedding_dependency": {
                    "type": "deployment_name",
                    "deployment_name": os.getenv("EMBEDDING_MODEL"),
                },
            }
        }
    ],
}
________________________________________
Outputs
 Query:
Where can I stay in New York?
❌ Before RAG (No Index)
Generic hotel suggestions with no brochure references.
________________________________________
✅ After RAG (Hybrid Search Enabled)
You can stay at The Manhattan Grand Hotel, located in Midtown near Central Park.
It offers modern suites, rooftop dining, and is within walking distance of major architectural landmarks.
Source: NYC_Brochure.pdf
________________________________________
 Follow-up:
Where can I stay there?
 Multi-turn Contextual Response
Based on your interest in architecture in New York, The Manhattan Grand Hotel is recommended...
________________________________________
 Index Creation Details
Feature	Configuration
Index Type	Vector + Keyword (Hybrid)
Embedding Model	text-embedding-ada-002
Chunking	Automatic (crack + chunk + embed)
Search Tier	Azure AI Search Basic
Retrieval Strategy	Top-N Hybrid Search
________________________________________
 RAG Flow in Prompt Flow (Microsoft Foundry)
Implemented using:
1.	Append chat history node
2.	Index Lookup tool
3.	Python node for context aggregation
4.	Prompt variant node (system message grounding enforcement)
5.	LLM completion node (GPT-4o)
________________________________________
 Performance Impact
Metric	Before RAG	After RAG
Hallucination Risk	High	Low
Domain Accuracy	Generic	Domain-specific
Context Awareness	None	Multi-turn
Semantic Matching	Keyword Only	Vector + Hybrid
________________________________________
 AI Engineer Highlights
 Technical Skills Demonstrated
•	Retrieval Augmented Generation (RAG)
•	Vector Embeddings
•	Hybrid Search Implementation
•	Azure AI Search Integration
•	Azure OpenAI SDK
•	Prompt Engineering
•	Multi-turn Chat Context Handling
•	Secure API Configuration
•	Cloud Resource Provisioning
________________________________________
 Resource Cleanup
Delete Azure Resource Group to avoid unnecessary costs.
________________________________________
 Summary
In this project, we:
✔ Built a grounded AI assistant
✔ Indexed proprietary data using Azure AI Search
✔ Implemented RAG in both SDK and Prompt Flow
✔ Used vector embeddings for semantic retrieval
✔ Delivered domain-specific, accurate responses
This project demonstrates real-world AI engineering skills in building enterprise-grade, grounded generative AI systems.

--

Aram Radif

