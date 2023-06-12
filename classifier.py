import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import intentLoader as loader


class IntentClassifier():
    def __init__(self, file_path = "./intents.json", save_folder = "./savedEmbeddings", refresh = False) -> None:
        self.file_path = file_path


        chroma_client = chromadb.Client(Settings(
                chroma_db_impl = "duckdb+parquet",
                persist_directory = save_folder # Optional, defaults to .chromadb/ in the current directory
        ))

        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = chroma_client.get_or_create_collection(name="intents", embedding_function=sentence_transformer_ef)

        if refresh:
            loader.load_data(file_path)
            self.initialize_collection()

    def update(self):
        loader.load_data(self.file_path)
        self.initialize_collection()

    def initialize_collection(self):
        grouped = loader.get_grouped() # [{tag: "", pattern: ""},...]
        file_classes = loader.get_classes()
        
        print(f"Initializing: {len(file_classes)} classes")

        docs = []
        metas = []
        ids = []

        for i in range(len(grouped)):
            intent = grouped[i]
            tag = intent['tag']
            pattern = intent['pattern']
            metadata = {'className':tag}

            docs.append(pattern)
            metas.append(metadata)
            ids.append(str(i))

        self.collection.upsert(
            documents=docs,
            metadatas=metas,
            ids=ids
        )

        current = self.collection.get()
        for i in range(len(current['ids'])):        
            if current['metadatas'][i]['className'] not in file_classes:
                self.collection.delete(current['ids'][i])



    def classify(self,query):

        results = self.collection.query(
            query_texts=[query.lower()],
            n_results=1
        )

        class_name = results['metadatas'][0][0]['className']
        distance = results['distances'][0][0]
        return distance, class_name