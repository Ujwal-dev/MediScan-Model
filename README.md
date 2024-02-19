```python
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
```


```python
 # type: ignore
```


```python

```


```python
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
results = chain.invoke(""" Age: 46
BMI: 27
Symptoms: Joint Pain, Increased Thirst, Dry Eyes, Increased Hunger
Complications: Rheumatoid Arthritis
Habits: Regular Exercise, Non-Smoker
Allergies: Tree Pollen """)
db.save_local("Gemini_Indexes")
```


```python
results.replace("*" , "")
```




    'Treatment:\n\nMedications:\n\n DMARDs: To address joint pain associated with rheumatoid arthritis.\n SGLT2 Inhibitor (Empagliflozin): To regulate blood sugar levels and address increased thirst.\n Artificial Tears: To alleviate dry eyes.\n GLP-1 Receptor Agonist (Dulaglutide): To reduce appetite and address increased hunger.\n\nGuidance on Secondary Complications:\n\n Monitor for signs of rheumatoid arthritis. Regular check-ups with a rheumatologist are recommended.\n\nAllergy Consideration:\n\n Patient is allergic to Tree Pollen. Alternative DMARDs, SGLT2 inhibitors, and artificial tears may be considered. Consult with a rheumatologist and endocrinologist for suitable alternatives.\n\nRecommendations:\n\n Emphasize joint care, lifestyle modifications, and follow prescribed medication regimen.'




