import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter

from agents.qa_memory import QAMemory

load_dotenv()


class ChatAgent:
    def __init__(self):
        print("ctor CandidateAgent")
        apiKey = os.environ["GOOGLE_API_KEY"]
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-pro", google_api_key=apiKey, temperature=0
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        self.chain = None

        self.loadDocuments()
        self.buildAnalyzerPromptTemplate()

    def chat(self, query):
        try:
            result = self.chain.invoke(
                {"question": query, "history": self.memory.getHistory()}
            )

            # update the conversation history
            self.memory.add(query, result)
            return result

        except Exception as err:
            return err

    def buildAnalyzerPromptTemplate(self):
        self.memory = QAMemory(3)
        self.analyzer_chain = None

        # Prompt Template
        template = """You are a helpful assistant that is able to retrieve information and answer my questions succinctly and accurately.
            You will only analyze and answer questions based on the following context:
            context: {resumes}

            If you do not know the answer, just say you do not know and don't try to guess.

            Question: {question}
            """

        prompt = ChatPromptTemplate.from_template(template)

        retriever = self.abbreviationsDB.as_retriever()

        # Build the langchain
        self.chain = (
            {
                "resumes": itemgetter("question") | retriever,
                "question": itemgetter("question"),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        print("recruiter agent analyzer-template initialized...")

    def loadDocuments(self):
        self.documents = []

        # load abbreviations document and store into a Document list for vector database
        abbreviationDocument = []
        with open("./documents/asrs_abbreviations.txt", "rb") as file:
            page_text = file.read()
            abbreviationDocument.append(Document(page_content=page_text))

        # load abbreviations document into vector database
        self.abbreviationsDB = FAISS.from_documents(
            abbreviationDocument, self.embeddings
        )

        # load reports.
