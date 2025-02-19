

 	

In this work, we develop a dynamic agentic RAG system for long, intricate legal and financial documents. Such a system requires orchestrating multiple agents for efficient context retrieval and reasoning, along with tool-reasoning capabilities to handle domain-specific tasks. The system should be robust and have decision making abilities to handle different user use cases and failure scenarios.

Current RAG systems treat retrieval and reasoning as separate entities, first performing context retrieval and then providing the context as a prompt to the LLM to perform reasoning. However, complex multihop queries require multi-step retrieval and reasoning in an interleaved manner. To address this, we introduce a novel interleaving RAG reasoning approach that allows LLMs to dynamically decide when to reason and retrieve, allowing the system to resolve multihop contextual queries.

Additionally, most of the current systems focus on caching conversation memory for future reference; we go beyond this standard practice and build a retrieval memory module that leverages previously retrieved context and user queries to construct a retrieval memory cache for efficient future retrieval. Furthermore, current failure handling mechanisms within Tool Agents fail to account for inevitable silent API failures, resulting in incorrect responses. We build an efficient tool reasoning methodology that achieves Level-2 tool failure handling capabilities in various such failure scenarios. 

We integrated our RAG agent within Pathway using the VectorStore Server & Client. We added the Jina Embedding API within the BaseEmbedder class, enhanced the BaseRAGQuestionAnswering class to support  
our interleaving approach, built a new tool reasoning class, and extended the BaseChat class to incorporate GroqChat Models for improved functionality. This integration will allow developers to use our agentic RAG pipeline using Pathway and its diverse functionalities. **Note:** All the experiments are done on Google Colab CPU environment.   
                                             

**System Workflow:**  The workflow as shown in Fig 1 begins by receiving a query *Q*, a set of documents *D,* and a set of tools T from the user. The tools can be user-provided or pre-defined as well. Given the above information, the supervisor agent first activates the Code & Reasoning agent, which can interact with tools and the RAG agent. Upon activation, the RAG Agent builds a document index for *D* using Pathway's VectorStore Server. It then utilizes Jina Embeddings to perform page-level retrieval and extract the top-k most relevant pages for *Q*. The pages are chunked and indexed using RAPTOR, forming a hierarchical structure over the summary of the chunks. Once indexing is complete, the RAG agent uses an interleaving approach to iterate between reasoning and retrieval to perform multi-hop contextual reasoning and return the RAG response. The Code & Reasoning (C\&R) agent can further utilize the tools for any tool-specific task based on the RAG agent's response and user query. Finally, the Supervisor Agent consolidates the outputs and returns the response to the user.

**Fig 1:** System Architecture integrated with Pathway  
Retrieving information from large documents, like financial and legal reports, is challenging for existing RAG systems due to their inherent hierarchies and diverse entities such as tables, charts, and images. To address this, we designed a custom two-stage retrieval pipeline, as shown in Fig. 2, featuring page-level Jina retrieval followed by RAPTOR-based retrieval. It comprises of the following components:

**1.1 Retriever Module**

**A. Page-level retrieval using Jina Embeddings**   
We use Jina Embeddings-v3\[1\], which is specifically trained on embedding generation for long-context document retrieval. Given an initial query Q and a long document D with N pages, we generate query embeddings and page-level embeddings using each page's text content alone, over which we then construct a FAISS index . Based on the input query, we retrieve the top-k pages of the document relevant to the query.

**A.1. Integration with Pathway**  
Pathway’s VectorStoreServer indexes documents by storing page-level content. For a long document D with N pages, a JSONL file is created, where each entry contains page text and its corresponding page number as metadata. The top-k pages are retrieved based on the embedding similarity between the query and page text.   
                                                             **Fig 2:** Retrieval Module in RAG Agent

These pages are extracted from the original long document using the page number metadata to form a smaller document D′. In our experiments, we set k=30, as it reliably ensures the retrieval of relevant context. We also extended the embedders module to create a new JinaEmbedder class based on the Jina API.

**B. Page Level Preprocessing using Unstructured**  
Often, documents have tables and images that contain critical information. To capture information from these modalities for the smaller document D’, we use the Unstructured library to parse the retrieved pages into images, plain text, and tables each with associated metadata (page number). The extracted data is then passed to an LLM, which summarizes the data , recombines elements page wise and indexes them.

**C. RAPTOR Index for Context Retrieval**  
RAPTOR\[2\] is a bottom-up indexing approach that involves segmenting the document into text chunks, embedding them, clustering the embeddings, and summarizing each cluster with an LLM, and essentially constructing a hierarchical tree structure. The initial page-level retrieval using JINA helps in greatly cutting down the indexing time (\~ 4 minutes for 20 pages). The generated summaries form a tree structure, with higher nodes providing more abstract summaries, aiding in domain-specific query answering.

**C.1. Integration of RAPTOR with Pathway**  
We have integrated the entire Raptor pipeline for document clustering and retrieval search ("collapsed" and "tree-traversal") using Pathway’s VectorStore as illustrated in Fig 3 below. Given page-wise content C′, we apply the Raptor Clustering algorithm to form a hierarchical tree. A JSONL file is then created, where the "cluster summary" is the primary data field, and metadata includes "level" and "parent\_id" for hierarchical information. This enables metadata-level filtering for document extraction and helps us use raptor retrieval for the retrieval of docs.

                     

      **Fig 3**. Each document chunk is summarized and added to the Pathway vector store

**D. Dynamic Memory Cache Module**  
 For each retrieval query, we extract the top-k chunks most relevant to the query. We then synthetically generate utility queries based on the content present in each chunk. The generated queries are cross-referenced with the current query bank to eliminate redundant queries. We then construct a dynamic memory cache using the “**nmslib”** library. Specifically, we prepare an HNSW graph over the utility queries and tag each query with the corresponding chunk. For any future retrieval query, we first check the query bank for similar queries (based on an appropriate minimum similarity value) and provide direct access to the chunk if there is a match, thus enabling fast retrieval. Fig 4 shows an illustration of our Dynamic Memory Module and HNSW graph creation. 

                                             **Fig 4**. Retrieval Memory Cache using Utility HNSW Graph 

**1.2. Reasoning Module**  
**A. Dynamic Retrieval and Reasoning Agent**   
We introduce a novel interleaving approach where the LLM acts as a dynamic decision-maker (Fig 5), iterating between retrieval and reasoning at each step. The interleaving agent intelligently decides whether to generate a retrieval thought to gather additional context or a reasoning thought to infer answers from existing information. This method minimizes reliance on external context by optimizing retrieval calls, making them only when  
necessary. Our system supports dynamic decision-making and real-time transformation of retrieval queries to ensure the successful retrieval of relevant context, eliminating the need for a separate critic agent.

			              **Fig 5**. Interleaving approach iterating between retrieval and reasoning  
**A.1. Integration with Pathway**  
We also extended the BaseRAGQuestionAnswerer class and integrated the interleaved retrieval and reasoning to develop the InterleavedRAGQuestionAnswerer class. This class can make use of any Pathway VectorStore client to perform retrieval. We also took the liberty of extending the current LLM services to include GroqChat Models by developing the GroqLLM Class. 

**B. Human-in-the-loop for Jargon Correction**  
We use a human-in-the-loop module for jargon correction. After generating the final answer, user feedback is collected. If negative, jargon or unclear terms in the query are identified, and the user is asked for clarification. Page-level retrieval is then performed for each jargon term, with the top pages processed using Unstructured and added to the retriever index. The RAG process is repeated, incorporating the clarified terms for improved results.

**C. Guardrails**  
	We integrated guardrails into the retrieval and reasoning modules to ensure safe and reliable LLM outputs. Using the Guardrails AI framework, we apply toxic language validators to filter out risks like offensive content in retrieved data. Additionally, Nemo AI Guardrails are used during Chat LLM interactions to filter responses, preventing harmful content from reaching users.  
Most real-world use cases involve the utilization of tools along with RAG to answer user queries. In order to automate this process accurately, we have constructed a Multi-Agent Dynamic RAG Framework capable of orchestrating various agents in a synchronized manner to address user queries.   
**A. Chain of Function Call**  
The Code & Reasoning Agent is based on a Chain of Function Call tool reasoning paradigm, where at each step, a single python function call is performed, based on the previous history of python function calls and responses. Each tool call is executed using an interpreter to generate the function tool response.  While this approach shares similarities with ReAct\[4\], the Evaluations below clearly show that it is more token and cost-efficient due to its reliance on code-driven planning.

**B. Error Handling and Reflexion**  
In a real world scenario it is natural that APIs can fail owing to various reasons. We have developed ingenious mechanisms to handle API errors, which we broadly classify into two types:

1. **“Loud” Tool Failure or Failed Function Calls** : These errors occur owing to (i) incorrect python syntax, or (ii) an API failure. In order to handle (i), we make use of a code reflexion agent to correct the python function call. To handle (ii), we identify the faulty function tool and discard it.  
2. **“Silent” Tool  Failure or Inconsistent Function Responses** : These errors can occur owing to (i) incorrect input arguments, or (ii) incorrect function tool response (rogue tools) . These failures are called “silent”\[3\] errors because they do not raise python code errors; the output is obtained however it is not coherent with the context. 

	

    **Fig 6**. Explanation of *silent* tool failure correction: incorrect arguments and inconsistent responses

To handle these special failures, we came up with the Critic Agent \[Fig 6\]; that is capable of identifying fallacies in both the arguments passed to the tool, and in judging the logical relevance of the tool response to the present context and the history of responses. For incorrect arguments, we invoke code reflexion to fix the arguments passed, and for irrelevant API responses, we discard the rogue API and resume the reasoning process.

**B.1. Integration with Pathway**  
We developed a new CodeAndReasoningAgent within Pathway’s question\_answering module for tool reasoning, that implements the features of the C\&R agent described earlier; Chain of Function Call, handling of silent API errors and python syntax errors, and dynamic tool generation.

While the proposed reflexion policies effectively manage tool failures, there are scenarios where the available toolset may not contain the necessary tools for answering a user’s query, or where all relevant tools are corrupt. We propose two methods to handle such cases: 

1. **Human-In-The-Loop** : The supervisor can prompt the user to provide function tools along with proper descriptions in order to address the query.   
2. **Dynamic Tool Generator Agent :**  If no relevant tool is available and the user does not provide one, the supervisor employs a dynamic tool generator to create real-time agentic tools tailored to the user's needs. It uses a use case driven prompt-refinement algorithm to dynamically generate the appropriate agent to mimic tool response. For a formal explanation, refer to **Algo1** in **Appendix A.**

In the real world, it can be expected that the user would have multiple queries related to a given document. It is important to develop a component that supports conversations with the user and maintains track of the history of interactions with the user. We have developed the following modules to support conversations : 

1. **Conversational Module** : This is a module that supports human-in-the-loop in order to answer any follow-up queries that the user may have.  The follow up query is requested from the user and we incorporate the most relevant answers that have already been generated by the Supervisor with the help of the Supervisor Memory Module in the follow-up query.  
2. **Supervisor Memory Module** :  This memory module is implemented as a vector index over the previous history of generated supervisor answers. Once a follow up query is received, we perform retrieval over the memory module and provide the top-k most similar answers along with the follow-up query to aid in rapid-answering of the same.

We have developed a web application with a UI using Next.js, backend services using FastAPI, AWS S3 for storage, and a PostgreSQL database. It features a model microservice using Flask for our agentic RAG system. All components are containerized with Docker, ensuring consistency across development and deployment. A Makefile with setup commands simplifies setup, running, and maintenance for developers. Refer to **Appendix G** for **complete architecture** and **User Interface**. 

**Challenges Faced** and **Pathway Integration Documentation** details can be found in **Appendix D** and **F.** 

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

1. Saba Sturua, Isabelle Mohr, Mohammad Kalim Akram, Michael Günther, Bo Wang, Markus Krimmel, Feng Wang, Georgios Mastrapas, Andreas Koukounas, Nan Wang, Han Xiao, jina-embeddings-v3: Multilingual Embeddings With Task LoRA, [https://arxiv.org/abs/2409.10173](https://arxiv.org/abs/2409.10173)   
2. Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, Christopher D. Manning, RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval, [https://arxiv.org/abs/2401.18059](https://arxiv.org/abs/2401.18059)    
3. Jimin Sun, So Yeon Min, Yingshan Chang, Yonatan Bisk, Tools Fail: Detecting Silent Errors in  
   Faulty Tools, [https://arxiv.org/abs/2406.19228](https://arxiv.org/abs/2406.19228)   
4. Shengran Hu, Cong Lu, Jeff Clune, Automated Design of Agentic Systems,   
5. Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao,   
   ReAct: Synergizing Reasoning and Acting in Language Models, [https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)   
6. Gangwoo Kim, Sungdong Kim, Byeongguk Jeon, Joonsuk Park, Jaewoo Kang, Tree of clarifications: Answering ambiguous questions with retrieval-augmented large language models,  
   [https://arxiv.org/abs/2310.14696](https://arxiv.org/abs/2310.14696)   
7. Dan Hendrycks, Collin Burns, Anya Chen, Spencer Ball, CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review, [https://arxiv.org/abs/2103.06268](https://arxiv.org/abs/2103.06268)   
8. Pranab Islam, Anand Kannappan, Douwe Kiela, Rebecca Qian, Nino Scherrer, Bertie Vidgen, FinanceBench: A New Benchmark for Financial Question Answering, [https://arxiv.org/abs/2311.11944](https://arxiv.org/abs/2311.11944)   
9. Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, LoRA: Low-Rank Adaptation of Large Language Models, [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)   
10. Yu. A. Malkov, D. A. Yashunin, Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs, [https://arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320)   
11. Elad Levi, Eli Brosh, Matan Friedmann, Intent-based Prompt Calibration: Enhancing prompt optimization with synthetic boundary cases, [https://arxiv.org/abs/2402.03099](https://arxiv.org/abs/2402.03099)

    

    

    

    

    

    

             
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