## Introduction

In the ever-evolving landscape of artificial intelligence, the intersection of AI and enterprise applications has seen significant advancements. However, deploying AI systems that are not only intelligent but also adaptive remains a challenge, especially in sectors like finance and law where precision and up-to-date insights are paramount. Traditional generative AI models often fall short in these environments due to their static nature, outdated knowledge, and lack of real-time decision-making capabilities.

Here is where Retrieval-Augmented Generation (RAG) systems come into play. These systems have emerged as a promising solution, but they too have their limitations. In this blog, we’ll explore how we’ve developed a **Dynamic Agentic RAG System** specifically designed for long, intricate legal and financial documents. This system not only addresses the shortcomings of traditional RAG systems but also introduces novel approaches to retrieval, reasoning, and memory management.

## Why RAG? The Need for Dynamic Retrieval and Reasoning

### The Limitations of Traditional AI Models

Imagine asking ChatGPT about a niche financial law that was recently passed. The model wouldn’t know about it because it was trained before the law existed. Pretraining or finetuning the model is an expensive option. This is where **RAG** comes into play. Instead of relying solely on pre-trained data, RAG systems retrieve relevant information from external databases or documents and use Large Language Models (LLMs) to generate contextually accurate responses.

![Traditional RAG System](images/RAG_Image.png)

### Why RAG is Essential

1. **AI Models Can’t Store Everything**: The sheer volume of data in legal and financial domains makes it impossible for AI models to store all relevant information in their memory.
2. **Constant Data Creation**: New data is continuously being generated, and RAG ensures that the AI can access the most up-to-date information.
3. **Factual and Grounded Responses**: By retrieving information from external sources, RAG systems provide responses that are more factual and grounded in reality.

### The Problem with Traditional RAG Systems

Traditional RAG systems often retrieve information indiscriminately, failing to adapt dynamically based on query complexity. This inefficiency is particularly problematic in financial and legal contexts, where data relevancy is critical. Moreover, these systems treat retrieval and reasoning as separate entities, first performing context retrieval and then providing the context as a prompt to the LLM for reasoning. This approach falls short when dealing with complex multi-hop queries that require interleaved retrieval and reasoning.

## Introducing the Dynamic Agentic RAG System

### A Multi-Agent Approach

To address these challenges, we adopted a **multi-agent approach**. In this system, different AI agents specialize in distinct tasks, working together to achieve a common goal. This approach introduces specialization, allowing each agent to focus on a specific aspect of the retrieval and reasoning process.

#### What Are Agents?

Agents are autonomous systems that analyze and act based on their environment to achieve specific goals. They can retrieve, process, and synthesize information, making decisions dynamically rather than following rigid rules. In our system, we have multiple agents, each specializing in tasks like retrieval, reasoning, and tool handling.

### The Role of Tools

In addition to retrieval and reasoning, our system incorporates **tools**—modular add-ons that enhance the system’s capabilities. Tools can include calculators, web search modules, chart generators, and more. These tools not only increase the accuracy of responses but also reduce human intervention, allowing the AI to surpass its inherent limitations.

### Pathway: The Backbone of Our System

For such an intelligent system to operate at scale, it requires an underlying infrastructure capable of handling massive data flows and real-time computations. **Pathway** serves as the backbone of our system, offering:

- **High-speed data processing** to ensure minimal latency.
- **Real-time retrieval** for dynamic knowledge updates.
- **Multi-modal data handling** to process various types of data.
- **Simplified deployment** with Docker and Kubernetes.

With Pathway, our Dynamic RAG system can operate seamlessly across vast datasets, continuously learning and adapting to new information without compromising speed or accuracy.

## System Architecture and Workflow

![System Architecture](images/system_architecture.png)

### Overview of the Workflow

The workflow of our Dynamic Agentic RAG System begins with the user providing a query (Q), a set of documents (D), and a set of tools (T). The system then follows these steps:

1. **Supervisor Agent Activation**: The Supervisor Agent activates the Code & Reasoning (C&R) Agent, which can interact with tools and the RAG Agent.
2. **Document Indexing**: The RAG Agent builds a document index for D using Pathway’s VectorStore Server.
3. **Page-Level Retrieval**: The system uses Jina Embeddings to perform page-level retrieval, extracting the top-k most relevant pages for Q.
4. **Hierarchical Indexing**: The retrieved pages are chunked and indexed using RAPTOR, forming a hierarchical structure over the summary of the chunks.
5. **Interleaved Retrieval and Reasoning**: The RAG Agent uses an interleaving approach to iterate between reasoning and retrieval, performing multi-hop contextual reasoning.
6. **Tool-Specific Tasks**: The C&R Agent utilizes tools for any tool-specific tasks based on the RAG Agent’s response and user query.
7. **Response Consolidation**: The Supervisor Agent consolidates the outputs and returns the final response to the user.

### Two-Stage Retrieval Pipeline

![Two-Stage Retrieval Pipeline](images/retriever.png)

Retrieving information from large documents, such as financial and legal reports, is challenging due to their inherent hierarchies and diverse entities like tables, charts, and images. To address this, we designed a custom **two-stage retrieval pipeline**:

#### 1. Page-Level Retrieval using Jina Embeddings

We use **Jina Embeddings-v3**, which is specifically trained for embedding generation in long-context document retrieval. Given an initial query Q and a long document D with N pages, we generate query embeddings and page-level embeddings using each page’s text content. We then construct a FAISS index and retrieve the top-k pages relevant to the query.

- **Integration with Pathway**: Pathway’s VectorStoreServer indexes documents by storing page-level content. For a long document D with N pages, a JSONL file is created, where each entry contains page text and its corresponding page number as metadata. The top-k pages are retrieved based on the embedding similarity between the query and page text.

#### 2. Page-Level Preprocessing using Unstructured

Documents often contain tables and images that hold critical information. To capture this data, we use the **Unstructured library** to parse the retrieved pages into images, plain text, and tables, each with associated metadata (page number). The extracted data is then summarized by an LLM, recombined page-wise, and indexed.

#### 3. RAPTOR Index for Context Retrieval

**RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval) is a bottom-up indexing approach that segments a document into text chunks, embeds them, clusters the embeddings, and summarizes each cluster using an LLM, forming a hierarchical tree structure. This method significantly reduces indexing time and enables structured retrieval at multiple levels.  

Unlike traditional chunking methods, which divide text into fixed, independent segments without capturing semantic links, RAPTOR ensures that related chunks are meaningfully connected. Since information in documents is inherently interconnected, treating chunks as isolated units can lead to inefficient retrieval. RAPTOR addresses this by clustering semantically related chunks and structuring them hierarchically, allowing for context-aware retrieval.  

- **Integration with Pathway**: Integration with Pathway requires metadata-level filtering to enable structured retrieval across hierarchical clusters.  

## **Fine-Tuning LLMs for Domain-Specific Tasks**  

![Each document chunk is summarized and added to the Pathway vector store](images/summary_module.png)  

During our experiments, we observed that a significant amount of API compute was being consumed in generating summaries for the RAPTOR module. This made the retrieval process costly and dependent on external APIs, creating scalability issues. To address this, we explored fine-tuning a smaller model to perform high-quality summarization locally.  

While large models like **LLaMA 2-70B** or **405B** are powerful, they are often resource-intensive and impractical for cost-efficient inference. Instead, we fine-tuned **LLaMA-7B** using **Parameter Efficient Fine-Tuning (PEFT)** with **LoRA adapters** on an **Nvidia A100**. The objective was to generate high-quality summaries tailored to the **CUAD dataset** (Contract Understanding Dataset), which focuses on legal document understanding.  

By fine-tuning a smaller model, we achieved **on-par performance with larger models** while significantly reducing inference costs. In some cases, our locally deployed summarizer even **outperformed** API-based solutions, making it a more scalable and efficient alternative.  

![summarizer_results](images/summarizer_results.png)

## **Interleaved Reasoning: Finding the Balance Between Retrieval and Synthesis**  

![Interleaving approach iterating between retrieval and reasoning](images/interleaving.png)

### The Need for Interleaving

Why do we need specialized reasoning techniques when RAG already exists? The answer lies in its limitations—traditional RAG lacks **deduction and synthesis capabilities**, which are crucial for handling complex legal and financial queries.  

Multi-hop queries require **multi-step retrieval and reasoning over intermediate retrieval steps**—a challenge that basic retrieval-augmented generation (RAG) struggles to handle. Several reasoning paradigms have attempted to bridge this gap:  

- **Chain of Thought (CoT)**: Encourages step-by-step reasoning by breaking problems into logical steps. However, it follows a linear path, making it inefficient for multi-hop queries that require branching logic.  
- **Tree of Thought (ToT)**: Extends CoT by exploring multiple reasoning paths, similar to a decision tree. While more flexible, ToT introduces redundancy by retrieving unnecessary information and increases token usage, especially when a reasoning path leads to a dead end.  
- **Graph-Based Reasoning**: We also explored graph-based reasoning with ROG (Reasoning on Graphs). While effective for structured knowledge graphs (KGs), it struggled with sparsity issues. Many LLM-driven KG generation techniques fail to capture implicit logical dependencies in financial and legal documents, limiting their reliability.  

#### Interleaving RAG Reasoning

Traditional RAG systems separate retrieval and reasoning into distinct steps, leading to inefficiencies in complex multi-hop queries. Our system introduces a novel **interleaving RAG reasoning approach**, allowing LLMs to dynamically decide when to retrieve and when to reason. By integrating retrieval within the reasoning process, our approach eliminates redundant lookups, efficiently resolving multi-hop contextual queries.

### How Interleaving Works

1. **Query Input**: The process starts with a user query.
2. **LLM Generates a Thought**: The LLM produces an initial reasoning step.
3. **Retrieval Step**: The model retrieves relevant documents from an index based on the reasoning step.
4. **LLM Refines the Thought**: The retrieved information is processed, and the LLM generates further reasoning.
5. **Interleaving Process**: This cycle of retrieval and reasoning continues iteratively, refining the knowledge step by step.
6. **Final Answer**: After sufficient iterations, the LLM produces a final, well-informed answer.

### Why Interleaving is Useful

- **Dynamic Refinement**: It allows the model to dynamically refine its reasoning based on retrieved information.
- **Reduced Hallucination**: By grounding responses in real-time knowledge retrieval, interleaving reduces the likelihood of the model generating incorrect or hallucinated responses.
- **Improved Performance**: Interleaving significantly improves performance in multi-step reasoning tasks, especially for complex queries.

## Benchmarking and Results

To validate our approach, we benchmarked different retrieval techniques:

- **Vanilla RAG and its variants** performed poorly in both Mean Reciprocal Rank (MRR) and time.
- **RAPTOR + Jina Embeddings** drastically outperformed traditional chunking, delivering high-precision retrieval without compromising speed.

![rag_results](images/rag_results.png)

We also experimented with various reasoning methods:

- **Knowledge Graphs (KGs)**: While effective when fully structured, KGs struggle with sparse data scenarios.
- **Interleaving RAG**: This approach bridges the gap by dynamically balancing retrieval and reasoning, outperforming traditional methods like Chain of Thought (CoT) and Tree of Thought (ToT).

!interleaving_results[](images/interleaving_results.png)


## Scaling Retrieval Efficiency with HNSW

![Retrieval Memory Cache using Utility HNSW Graph](images/cache.png)

### Dynamic Memory Cache Module

To enhance retrieval efficiency in long-document RAG, we built a **Dynamic Memory Cache Module** using **HNSW (Hierarchical Navigable Small World)** for fast approximate nearest neighbor search.

For each retrieval query, we extract the top-k most relevant chunks and generate utility queries based on their content. These queries are cross-referenced with the existing query bank to eliminate redundancy. We then construct a dynamic memory cache using **nmslib**, building an HNSW graph over the utility queries.

This enables efficient retrieval by checking the query bank for similar queries and directly accessing relevant chunks if a match is found. HNSW’s multi-layered graph structure supports real-time updates, making it well-suited for dynamic RAG.

### Why HNSW?

- **Efficiency in High Dimensions**: Unlike tree-based methods that suffer in high dimensions, HNSW maintains efficiency.
- **Real-Time Indexing**: HNSW supports real-time creation and modification of graph indexes, ideal for dynamic systems.
- **Memory Efficiency**: It avoids brute-force comparisons and stores embeddings in a compressed form, reducing storage overhead.

### How We Use HNSW in Dynamic Memory

- **Metadata Tagging**: QA pairs, related queries, and retrieved chunks enrich the knowledge base.
- **User-Adaptive Learning**: The system adapts to query history over time.
- **Lightning-Fast Follow-Ups**: Stored query embeddings speed up contextualized retrieval.

![cache_results](images/cache_results.png)

By combining HNSW with interleaving RAG, we achieve ultra-fast, context-aware retrieval in follow-up queries, pushing long-document retrieval into the future.

## Code & Reasoning Agent
### **Chain of Function Call**  
The Code & Reasoning Agent is based on a Chain of Function Call tool reasoning paradigm, where at each step, a single python function call is performed, based on the previous history of python function calls and responses. Each tool call is executed using an interpreter to generate the function tool response.  While this approach shares similarities with ReAct\[4\], the Evaluations below clearly show that it is more token and cost-efficient due to its reliance on code-driven planning.

## **Error Handling and Reflexion in Code & Reasoning Agent**

In real-world scenarios, **Tool can fail** for various reasons. To ensure system robustness, we have developed mechanisms to handle Tool failures effectively. These errors are broadly classified into two types.


### **1. ***Loud*** Tool Failure**
A Loud Tool Failure occurs when executing a tool call generates a Python error or exception. These errors are immediately visible in the form of execution failures, which makes them easier to detect. We have classified loud tool failure in two types.

#### **A. Incorrect Python Syntax** 
   When the LLM generates a function call, it might incorrectly assign argument types, leading to a syntax error.

##### Example
```python
def process_data(name: str, age: int):
    return f"{name} is {age} years old."
```
##### Incorrect function call generated by LLM
```python
process_data("Alice", "25")  
# ❌ Error: 'age' should be an integer
```

- In order to solve such type of error we have developed ``Code Reflextion Agent``              

#### **B. Internal Tool Failure**
  If there is some internal failure in tool due to server issue , wrong api key or any other internal reason . 
##### Example
```python
def process_data():
    resp = requests.get("https://xyz.com/hello")  # API Call
    return resp.json()  # Return API Response

process_data()  
```
##### Tool Failure due to API Key
```python
"Api key is not provided"
```

- In case of internal error, we can not modify tool, as tools are provided by the user. So only option left is to remove the tool.
- For handling such error `API REFLEXTION AGENT` is called after removing the tool.


### **2. ***Silent*** Tool Failure** : 
A Silent Tool Failure occurs when executing a tool call do not generates a Python error but still there is a error. These errors are not immediately visible in the form of execution failures, which makes them hatder to detect. We have classified loud tool failure in two types. Further it can divided into two parts

#### **A. Incorrect Input Argument** 
If the argument pass by LLM are correct by syntax but their value does not align with what was asked in the question
##### Example
```python
"Task :- What is the capital Of Indiana"
```
##### Silent Tool Failure Detected
```python
#Tool Call
"web_Search('What is the capital of India')"
# ❌ Error: 'india' should be an indiana
```

- For handling these error , we have implemented `critic agent` , which will analyze the argument passed on the basis of the task and tool description
- If `critic agent` find any error then `silen error reflexion agent` is activated


#### **B. Incorrect Function Tool Response** 
If the output generated by the tool is incorrect, unrelated, or inconsistent with the expected task, it results in a Silent Tool Failure

##### Example
```python
# Task: Compute 5 * 6
calc('5*6')  
```
##### Silent Tool Failure Detected
```python
# Incorrect Tool Response
ans = 45  # ❌ Error: Expected 30, but received 45
```

- For handling such kind of error `critic agent` is utilized
- When `critic agent` find irrelevant answer it will activate the `api reflextion agent`

## Dynamic Tool Set Enhancement
While the proposed reflexion policies effectively manage tool failures, there are scenarios where the available toolset may not contain the necessary tools for answering a user’s query, or where all relevant tools are corrupt. We propose two methods to handle such cases: 

1. **Human-In-The-Loop** : The supervisor can prompt the user to provide function tools along with proper descriptions in order to address the query.   
2. **Dynamic Tool Generator Agent :**  If no relevant tool is available and the user does not provide one, the supervisor employs a dynamic tool generator to create real-time agentic tools tailored to the user's needs. It uses a use case driven prompt-refinement algorithm to dynamically generate the appropriate agent to mimic tool response. For a formal explanation, refer to **Algo1** in **Appendix A.**

## Conversational Module
In the real world, it can be expected that the user would have multiple queries related to a given document. It is important to develop a component that supports conversations with the user and maintains track of the history of interactions with the user. We have developed the following modules to support conversations : 

1. **Conversational Module** : This is a module that supports human-in-the-loop in order to answer any follow-up queries that the user may have.  The follow up query is requested from the user and we incorporate the most relevant answers that have already been generated by the Supervisor with the help of the Supervisor Memory Module in the follow-up query.  
2. **Supervisor Memory Module** :  This memory module is implemented as a vector index over the previous history of generated supervisor answers. Once a follow up query is received, we perform retrieval over the memory module and provide the top-k most similar answers along with the follow-up query to aid in rapid-answering of the same.

# Evaluation and Experimentation
## Rag Agent

1) **Context Retrieval** : We use our self-curated CUAD dataset (**Appendix C**) to evaluate the retrieval performance of various retrieval techniques. Table \[1\] shows that while the RAPTOR module has the best retrieval performance, it lags behind others in time considerations.

**Tab 1**: Comparison of retrieval methods based           **Tab 2**: Performance Comparison of various Doc  
      on MRR and Time Efficiency on CUAD                        level retrieval methods on FinanceBench

2) **Page Level Retrieval :** Raptor’s exponential time scaling problem with token count prompts us to explore various Doc-level retrieval techniques in order to pre-select pages for RAPTOR processing. As shown in Table 2\. Jina Embeddings-v3 gives not only the highest MRR scores but also best time efficiency showcasing its effectiveness in handling long document retrieval tasks.   
3) **Reasoning Technique Comparison** : Table 3 compares three key LLM reasoning techniques: Chain of Thought (COT), which breaks complex queries into sequential sub-queries; Tree of Clarifications (TOC), which recursively branches into sub-questions until resolution; and the Interleaving approach, which dynamically alternates between retrieval and reasoning to reach the final solution.  
4) **CRAG and RAGAS Performance :** To test the generalizability of our retrieval pipeline, we tested the performance of our Interleaving approach using Jina \+ RAPTOR retrieval on standard RAG Benchmarks (Tab 4 and Tab 5). CRAG is a lightweight retrieval evaluator designed to assess the overall quality of retrieved documents for  a query, while Ragas is a specialized evaluation framework designed to assess the performance of **Retrieval Augmented Generation** (RAG) systems using LLMs as judges.

   
       **Tab 3**: Performance Comparison of Various                       **Tab 4**: CRAG score of our retrieval module  
           Reasoning Techniques on FinanceBench                                vs Vanilla RAG based retrieval

      **Tab 5:** RAGAS score of our retrieval module   
                    vs Vanilla RAG based retrieval

We also conducted an experiment to the effectiveness of our memory module in follow up question answering tasks. The following table gives a comparison between the average time taken to answer the first query versus the average time taken to answer the next follow up questions.  

               **Tab 6:** Table showing the Retrieval Time Data of the RAG pipeline with the Memory Module  
We evaluated custom multi-hop queries on tool reasoning in the Legal and Finance domains using curated datasets: CUAD for legal texts and Finance-10k for financial reports. MetaTool (ICLR 2024), a benchmark for tool usage and selection, was used to test tasks 1 (single-tool reasoning) and 4 (multi-tool reasoning). As a baseline we considered ReAct. Clearly, CoFC is considerably more efficient and also highly accurate. The results are illustrated in the following figures (Fig 7 , 8 , 9\)

**Fig 7, Fig 8, Fig 9 :** Plots of average inference time, accuracy and token count for ReAct and CoFC  on different dataset and settings.

     

**Tab 7:** Bert Score comparison between fine tuned 7B and other LlamA models  for legal chunk summarization  
      
While LLaMA 2-7B lacks the response quality of larger models like LLaMA 2-70B or 405B, it can be effectively fine-tuned for specific tasks. We fine-tuned the LLaMA 2–7B model using Parameter Efficient Fine-Tuning with LoRA adapters on Nvidia A100 to generate high-quality summaries for the CUAD dataset. This locally loadable summarizer produced results comparable to larger models and occasionally outperformed them (Tab 7). However, we did not finetune models for the finance domain due to the absence of a suitable dataset, as even GPT-4 struggled to   summarize information-dense financial information reports.  
**Fig 10:** Finetuning Llama2-7B LoRA Adapter     

      **Tab 8:** Finance Bench \+ CUAD Reasoning Results Interleaving : (Jina \+ Jina) vs (Jina \+ Raptor)

  This study highlights how RAPTOR summaries may omit crucial details in information-dense datasets like FinanceBench. Table 8 shows that while RAPTOR often outperforms Vanilla RAG, the latter can provide more accurate results when RAPTOR fails to capture all necessary information. To address this, a document classification module was added to the RAG Agent pipeline, assessing the information density of each document. If the density surpasses a threshold, the system bypasses RAPTOR's summarization, using direct Vanilla RAG retrieval to ensure accuracy and relevance.

In this work we present a Multi-Agent Dynamic RAG framework designed for accurate handling of long legal and financial documents. Leveraging Pathway’s dynamic data indexing, our framework supports multi-document question answering. It includes a RAG agent with a retrieval pipeline based on Jina Embeddings, RAPTOR indexing, and Pathway VectorStore, alongside an interleaved reasoning strategy for dynamic retrieval and reasoning decisions. A dynamic memory cache index enables fast retrieval of previously queried information. To manage complex multi-hop queries, we developed a Code & Reasoning Agent that uses the RAG agent and other tools within a dynamic toolset, employing the Chain of Function Call paradigm. Robust tool failure handling, dynamic tool generation, and guardrails ensure effective query resolution while filtering offensive content. 


    

    

    

    

    

    

             
                **Fig 11:** Tool Generation for Calculator

        **Fig 10:** Tool Generation for ChartAgent 

                                                **Algo 1 :** User Usecase Prompt Optimization  
    **B.1. Incorrect Arguments :** In order to illustrate this scenario when incorrect arguments can be passed to functions, we made use of an input corrupter agent that deliberately corrupted the arguments actually passed by the LLM while making the tool call, and the critic agent then rightfully raised a silent reflexion flag and refactored the tool call : 

           

            **B.2. Inconsistent Responses :** Again, to simulate this scenario we made use of an output corrupter agent that deliberately changed the actual output generated by making the tool call, and the critic agent flagged those tools as faulty tools and invoked API reflexion.

In our testing, we made use of and curated multiple datasets to ensure the robustness of the system. The Finance Bench dataset evaluates financial reasoning, logical deduction, and quantitative analysis, focusing on interpreting financial statements and market trends. Finance 10K emphasizes factual knowledge, testing models’ ability to retrieve financial facts and recognize entities. Finance Multihop, created from FinanceBench 10K reports, challenges multihop reasoning across contexts. Similarly, CUAD Multihop, derived from CUAD, assesses legal reasoning over multiple pages and contexts. The Open Australian Legal Q\&A dataset evaluates performance on openly available legal queries. Lastly, the Supervisor Dataset includes custom legal and finance queries curated from FinanceBench and CUAD to test both reasoning and retrieval in our system.  
While implementing our workflow with Pathway, we improved upon the following challenges:  
**D.1. Absence of Reasoning Agent Modules :** We observed that there were no modules for tool based LLM reasoning. To resolve this, we implemented the Code & Reasoning Agent in the question\_answerer.py module to enable LLM based tool reasoning. Further, we also extended the BaseRAGQuestionAnswerer class within the module to enable interleaved retrieval-and-reasoning.  
**D.2.** **Inconsistent Documentation** **:** The Pathway documentation is sometimes out of sync with the source code, making it challenging to set up and use features like the VectorStoreServer and BaseRAGQuestionAnswerer. Documents in some cases lack crucial details, such as the Timeout error caused by latency during SentenceTransformer model setup, which can be mitigated by specifying GPU usage in the source code. To resolve this, we extended the BaseEmbedding class to integrate the Jina Embedding API, enabling seamless embedding module functionality within Pathway's VectorStore.  
**D.3.**  **Restrictions in** **Pathway.llm.xpacks.parsers.py :** The parsers.py file initially restricted data input to a specific format (bytes), limiting user flexibility in experimenting with various data sources sent to the Pathway server. To address this limitation, we extended the data format being sent to “string”. This enhancement ensures compatibility with data sources preferred by users, while maintaining consistency across all related functions within the system.  
This section contains illustrations of how our robust Guardrail system filters out harmful content from both retrieved contexts as well as the Chat LLM response.

**Fig 12:** Guardrail Implementation within Retriever         **Fig 13:** Guardrail Implementation within LLM Response  
We have prepared comprehensive documentation to help readers understand the various functionalities of our Multi-Agent RAG system. This documentation highlights the contributions we made to the project, organized into separate modules such as **Pathway Integration**, **Supervisor Agent**, **RAG Agent**, and **Dynamic Cache Index**, to name a few. Each module includes a brief overview of its purpose, detailed explanations of its implementation, and guidelines on how to use it effectively. The entire documentation can be found in the supplementary code material.

We have developed a fully functional web application using a modern tech stack to ensure scalability, performance, and ease of deployment. The frontend has been built using Next.js, providing a seamless and dynamic user experience. The frontend connects to a FastAPI-based backend gateway, which serves as the central communication hub, managing interactions with various subsystems.

The backend gateway integrates with an Amazon S3 bucket for efficient and secure storage of PDF files. Additionally, it connects to a Flask-based microservice that powers an agentic Retrieval-Augmented Generation (RAG) system, enabling intelligent and context-aware document processing. For data management, we have utilized PostgreSQL as the database management system (DBMS), ensuring reliable and scalable data storage and retrieval. 

All services, including the frontend, backend, microservices, and database, have been containerized using Docker. This approach simplifies development workflows and ensures the application is production-ready with consistent and isolated environments across various stages of deployment. These technologies collectively enable a robust, efficient, and easily deployable web application suitable for diverse use cases.

If the available set of tools are insufficient for a specific subtask, users can either provide their own Python function or submit a description of the required tool, as shown in the figure below.  
   
To address challenges related to domain-specific jargon, a functionality has been incorporated into the RAG tool, enabling users to input clarifications for such terms. Additionally, one endpoint has been introduced in the UI: \\get\_history, which allows users to view their conversation history. These enhancements collectively aim to provide a more robust and user-friendly application.

## Conclusion

Thanks for reading! Feel free to contribute.
