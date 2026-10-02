import os
from operator import itemgetter

import bs4
from langchain import hub
from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def configure_environment():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = "<your-api-key>"
    os.environ["OPENAI_API_KEY"] = "<your-api-key>"


def build_blog_retriever():
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    blog_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=50,
    )
    splits = text_splitter.split_documents(blog_docs)

    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    return vectorstore.as_retriever()


def get_unique_union(documents: list[list]):
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]


def build_multi_query_chain(retriever):
    template = """You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)
    generate_queries = (
        prompt_perspectives
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    return retrieval_chain


def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    return [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]


def build_rag_fusion_chain(retriever):
    template = """You are a helpful assistant that generates multiple search queries based on a single input query.
Generate multiple search queries related to: {question}
Output (4 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)
    generate_queries = (
        prompt_rag_fusion
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )
    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
    return retrieval_chain_rag_fusion


def build_decomposition_prompt():
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question.
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation.
Generate multiple search queries related to: {question}
Output (3 queries):"""
    return ChatPromptTemplate.from_template(template)


def format_qa_pair(question, answer):
    return f"Question: {question}\nAnswer: {answer}\n\n".strip()


def build_step_back_prompt():
    examples = [
        {"input": "Could the members of The Police perform lawful arrests?", "output": "what can the members of The Police do?"},
        {"input": "Jan Sindel’s was born in what country?", "output": "what is Jan Sindel’s personal history?"},
    ]
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        few_shot_prompt,
        ("user", "{question}"),
    ])


def build_step_back_chain(retriever):
    prompt = build_step_back_prompt()
    generate_queries_step_back = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
    response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

# {normal_context}
# {step_back_context}

# Original Question: {question}
# Answer:"""
    response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

    chain = (
        {
            "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
            "step_back_context": generate_queries_step_back | retriever,
            "question": lambda x: x["question"],
        }
        | response_prompt
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
    )
    return chain


if __name__ == "__main__":
    configure_environment()
    retriever = build_blog_retriever()
    question = "What is task decomposition for LLM agents?"

    multi_query_chain = build_multi_query_chain(retriever)
    docs = multi_query_chain.invoke({"question": question})
    print("Multi-query docs:", len(docs))

    rag_fusion_chain = build_rag_fusion_chain(retriever)
    fusion_docs = rag_fusion_chain.invoke({"question": question})
    print("RAG-fusion docs:", len(fusion_docs))

    decomposition_prompt = build_decomposition_prompt()
    generate_queries_decomposition = decomposition_prompt | ChatOpenAI(temperature=0) | StrOutputParser() | (lambda x: x.split("\n"))
    questions = generate_queries_decomposition.invoke({"question": question})
    print("Generated sub-questions:", questions)

    chain = build_step_back_chain(retriever)
    print(chain.invoke({"question": question}))
