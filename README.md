### FAA LLM sandbox

### Quick Start Guide.

1. Don't forget to install requirement packages defined in requirements.txt
2. Currently agent is set to use ChatGPT. You can switch to Gemini-Pro in the chat_agent.py file

```
# uncomment to use Google Gemini-Pro
# apiKey = os.environ["GOOGLE_API_KEY"]
# self.llm = ChatGoogleGenerativeAI(
# model="models/gemini-pro", google_api_key=apiKey, temperature=0)
# self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# print("ctor CandidateAgent - using Google Gemini Pro.")

# uncomment to use ChatGPT
self.llm  =  ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
self.embeddings  =  OpenAIEmbeddings()
print("ctor CandidateAgent - using ChatGPT")
```

3. Make a copy of the .env-template file and rename it as .env. Enter the api-key.
4. There are 2 modes of operation.

```
# Option 1.  Runs a basic command-line
python3 app.py

# Option 2. Runs a streamlit UI
streamlitstream run streamlit_app.py
```

Both options run the same agent code. Just a different UI for your convenience.

# Dataset

1. The datasets are the csv files called dataset-small.csv and dataset-large.csv located in the documents folder.
2. There is also a second file called asrs_abbreviations.txt which contains some additional key information from the website.
