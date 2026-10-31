# Project: MediFind AI — Prescription Medicine Finder

## 1. Project Overview

I am building a portfolio-level AI application called MediFind AI.

The purpose of this application is to help users:
1. Upload an image of a doctor's prescription.
2. Extract possible medicine names, strengths and dosage forms using OCR and AI.
3. Allow users to review and correct extracted medicine details.
4. Search for matching medicine products.
5. Compare available prices from online retailers and provide direct product links.
6. Answer questions using a curated, verified medical knowledge base through Retrieval-Augmented Generation (RAG).
7. Return a clear "Information not found in verified database" response when reliable information is unavailable.

Users may optionally provide disease information or context. This must be treated as user-provided information, not a verified diagnosis. The application must not diagnose diseases or recommend changing prescribed medicines.

## 2. My Experience and Learning Requirements

I am a beginner in Python application development and AI engineering.

I have some programming experience in C++ and Java and basic exposure to Python, but I want to understand every component while building this project.

Act as my programming mentor.

Follow these rules:
- Explain concepts from the basics.
- Explain why a particular technology or library is needed.
- Do not assume I already know advanced Python, RAG, embeddings, OCR or Streamlit.
- Introduce concepts when they become relevant.
- Prefer simple implementations over unnecessary abstractions.
- Do not generate the entire project at once.
- Break the project into small, achievable milestones.
- Explain every important line of code.
- When I encounter an error, help me understand its root cause rather than just providing a replacement.
- Encourage me to attempt implementations before providing complete solutions.
- Do not introduce unnecessary frameworks or design patterns.

My goal is to learn while building a genuine, deployable project that adds value to my Data Engineering / AI Engineering resume.

## 3. Technology Stack

Use the following initial stack:

Frontend and application UI:
- Python
- Streamlit

AI and image processing:
- OCR or a suitable vision-capable LLM
- OpenAI API or another suitable LLM provider
- Python image-processing libraries

RAG:
- LangChain
- Embedding models
- Chroma vector database

Data storage:
- Start with structured local datasets.
- Introduce PostgreSQL when required.

Development:
- VS Code
- Git and GitHub
- Python virtual environment
- pytest
- Docker at a later stage

Backend:
- Initially use Streamlit with modular Python functions.
- Introduce FastAPI later if separating the API/backend becomes beneficial.
- Do not introduce React or other frontend frameworks initially.

Do not install all dependencies immediately. Add libraries only when needed.

## 4. Planned Project Architecture

Keep the application modular.

Proposed structure:

medifind-ai/
    app.py
    config.py

    services/
        ocr_service.py
        medicine_matching.py
        rag_service.py
        product_search.py

    database/
        vector_store.py
        medicine_repository.py

    models/
        medicine.py

    data/
        sample_medicines/
        knowledge_base/

    tests/

    .env.example
    .gitignore
    requirements.txt
    README.md

You may suggest improvements to this structure when justified. Do not create all these files immediately.

## 5. Core Application Workflow

The expected workflow is:

Step 1: User uploads a prescription image through Streamlit.

Step 2: Validate the image and process it using OCR or a vision model.

Step 3: Extract candidate medicine information:
- Medicine name
- Strength
- Dosage form
- Available dosage instructions, if legible

Step 4: Display extracted information for user confirmation.

Step 5: Search a structured medicine catalog and match exact medicine identity, strength and dosage form.

Step 6: Search approved external product sources for current product listings.

Step 7: Compare genuinely comparable pack sizes and prices.

Step 8: Display retailer names, prices, source timestamps and product links.

Step 9: Allow users to ask questions answered using relevant verified knowledge-base documents through RAG.

Step 10: If reliable information is unavailable, abstain instead of generating unsupported medical information.

## 6. RAG Requirements

I want to understand RAG rather than just use a framework.

Explain these concepts progressively:
- Documents
- Text extraction
- Chunking
- Embeddings
- Vector databases
- Similarity search
- Retrieval
- Prompt construction
- LLM generation
- Retrieval evaluation

Use a curated collection of trusted medical reference documents.

Store useful metadata:
- Source
- Document type
- Medicine identifier
- Therapeutic category
- Verification status
- Document date

Use disease categories as metadata where appropriate, not as the sole basis for medicine matching.

Implement retrieval thresholds and abstention carefully. Similarity scores must not be treated as proof of medical correctness.

## 7. Medical Safety Requirements

This application is an information and product-search assistant, not a diagnostic or prescribing system.

Mandatory safeguards:
- Never guess illegible handwritten medicine names.
- Never automatically substitute a prescribed medicine.
- Never recommend changing dosage or treatment.
- Require confirmation of uncertain extracted medicine information.
- Do not treat user-reported diseases as verified diagnoses.
- Only use approved, traceable medical information sources.
- Do not hallucinate prices, products, availability or retailer links.
- If a product is not found, clearly report that it is unavailable in the searched sources.
- Direct users to qualified healthcare professionals or pharmacists when necessary.
- Handle prescription images as sensitive health information.
- Never expose API keys or upload private prescription data to GitHub.

## 8. Development Methodology

Develop the application incrementally.

Milestone 1: Environment setup and Python fundamentals.

Milestone 2: Basic Streamlit interface and prescription image upload.

Milestone 3: OCR integration and structured extraction.

Milestone 4: Medicine confirmation and matching.

Milestone 5: Product search and price comparison.

Milestone 6: RAG implementation using LangChain and Chroma.

Milestone 7: Integration of all components.

Milestone 8: Testing, Docker, deployment and documentation.

For each milestone:
1. Explain the objective.
2. Explain the concepts and technologies required.
3. Give me one small implementation task.
4. Explain the code and commands.
5. Help me test the result.
6. Wait for my confirmation before proceeding to the next milestone.

Do not skip ahead.

## 9. Coding Standards

- Write readable, modular Python code.
- Use meaningful variable and function names.
- Add type hints where appropriate.
- Use environment variables for secrets.
- Handle exceptions properly.
- Validate external API responses.
- Avoid unnecessary dependencies.
- Write unit tests for important business logic.
- Document important functions.
- Keep README.md updated as the project evolves.

## 10. Important Instruction

I do not want you to build the entire application for me in one response.

I want to develop it myself with your guidance.

Whenever I ask for implementation help:
- First explain the approach.
- Give me the smallest useful implementation step.
- Explain why it works.
- Give me a test to perform.
- Review my code and suggest improvements.

Start by helping me set up the project environment and build the first basic Streamlit application.

Remember: teaching me to build the project is as important as completing the project.