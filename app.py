from flask import Flask, request, jsonify
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()

llm = GoogleGenerativeAI(model = "gemini-pro") 
loader = DirectoryLoader('Data/', glob="**/*.txt")
docs = loader.load()
train_docs = []
for doc in docs:
    if "test" in doc.metadata["source"].lower():
        test_doc = doc
    elif "validation" in  doc.metadata["source"].lower():
        validation_doc = doc
    else:
        train_docs.append(doc)
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap = 100,
    length_function=len,
    is_separator_regex=False,
)
split_training_docs = text_splitter.split_documents(train_docs)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
db = FAISS.from_documents(split_training_docs, embeddings)
retriever = db.as_retriever()

template = """You are given a list of past diabetic patient treatments as recommnedations and you are an expert medical practitioner.
Your task is to recommend a treatment for the question asked which will be a patient description including Age, BMI, primary symptoms,
secondary complications and allergies if any present. Generate a custom treatment such that it includes medications for all the symptoms,
complications and allergies if listed along with their description. Use the provided context as a reference. Stick strictly to the context for answering:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.route('/', methods=['GET'])
def home():
    return "Hello world"

@app.route('/get_chat', methods=['POST'])
def get_chat():
    # Get input prompt from request body
    input_prompt = request.json.get('input_prompt', '')
    
    # Invoke the chain with the input prompt
    result = chain.invoke(input_prompt)

    return jsonify({'response': result})

if __name__ == '__main__':
    if __name__ == '__main__':
    # Define Gunicorn configuration
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    workers = int(os.environ.get('WEB_CONCURRENCY', 1))
    from gunicorn.app.base import BaseApplication

    class StandaloneApplication(BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super(StandaloneApplication, self).__init__()

        def load_config(self):
            config = {key: value for key, value in self.options.items() if key in self.cfg.settings and value is not None}
            for key, value in config.items():
                self.cfg.set(key.lower(), value)

        def load(self):
            return self.application

    options = {
        'bind': '%s:%s' % (host, port),
        'workers': workers,
    }

    # Run the Flask app with Gunicorn
    StandaloneApplication(app, options).run()
