import os
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from agents.qa_memory import QAMemory

load_dotenv()


class ChatAgent:
    def __init__(self):
        # apiKey = os.environ["GOOGLE_API_KEY"]
        # self.llm = ChatGoogleGenerativeAI(
        #     model="models/gemini-pro", google_api_key=apiKey, temperature=0
        # )
        # self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # print("ctor CandidateAgent - using Google Gemini Pro.")

        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.embeddings = OpenAIEmbeddings()
        print("ctor CandidateAgent - using ChatGPT")

        self.abbreviationsDB = None
        self.reportsDB = None
        self.chain = None

        self.loadAbbrDocument()
        self.loadIssuesDocument()
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
            You will only analyze and answer questions based on the following abbreviation keys and context:
            
            ### Key to reading abbreviations specific to flight:
            {abbreviations}
            
            ### The following is a list of issues or reports:
            {reports}

            All your answers should be based on the context of the reports and abbreviations provided above.  When answering a question,
            always include the ACN reference number in the title line if available.  

            Current conversation:
            {history}
            Question: {question}
            """

        # important to set k because default is 4.
        prompt = ChatPromptTemplate.from_template(template, kwargs={"k": 4})
        retriever = self.reportsDB.as_retriever(search_kwargs={"k": 4})

        # Build the langchain
        self.chain = (
            {
                "abbreviations": itemgetter("question")
                | self.abbreviationsDB.as_retriever(),
                "reports": itemgetter("question") | retriever,
                "question": itemgetter("question"),
                "history": itemgetter("history"),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        # self.chain = RunnableParallel(
        #     {
        #         "reports": retriever,
        #         "abbreviations": self.abbreviationsDB.as_retriever(),
        #         "question": RunnablePassthrough(),
        #     }
        # ).assign(answer=rag_chain_from_docs)

        print("recruiter agent analyzer-template initialized...")

    def loadIssuesDocument(self):
        # load issues document and store into a Document list for vector database
        # the second line is the header.
        row = 0
        documents: List[Document] = []

        with open("./documents/dataset1-small.csv", "r") as file:
            while True:
                line = file.readline()
                row += 1
                if not line:
                    break

                # line 2 is the header
                if row == 2:
                    headerArr = line.strip().split(",")
                    # Filter out any empty strings from the list because there is an extra , at the end of the line.
                    headerArr = [x for x in headerArr if x]

                if row > 3:  # skip the first three lines.
                    columns = line.strip().split(",")[:-1]  # skip the last column.

                    issueArr = []
                    for i in range(len(columns)):
                        issueArr.append(f"{headerArr[i]} : {columns[i]}")

                    # build document
                    documents.append(
                        Document(
                            page_content="\n".join(issueArr),
                            metadata={"source": columns[0]},
                        )
                    )

        # load issues document into vector database
        self.reportsDB = FAISS.from_documents(documents, self.embeddings)

    def loadAbbrDocument(self):
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
