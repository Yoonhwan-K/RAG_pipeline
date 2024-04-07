import os
import json
import pickle
import torch

from dotenv import load_dotenv

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

from transformers import AutoModel, AutoTokenizer

class MilvusEmbedding :

    def __init__(self, config) :

        self.config = config

    def create_connection(self) :

        milvus_alias = self.config.milvus_alilas
        milvus_uri = self.config.milvus_uri

        connections.connect(alias = milvus_alias, uri = milvus_uri)

    def check_collection_name(self) :

        self.collection_name = self.config.collection_name

        if utility.has_collection(self.collection_name) :

            utility.drop_collection(self.collection_name)

        else :

            pass

    def load_data(self) :

        data_path = self.config.data_path

        if os.path.splitext(data_path)[1] == ".json":

            with open(data_path, "rb") as f :
                raw_data = json.load(f)

            documents = []
            for key, value in raw_data.items() :
                
                if self.config.embedding_target_value.lower() == "key" : 

                    documents.append(key)
                
                elif self.config.embedding_target_value.lower() == "value" :

                    documents.append(value)

            return documents

        elif os.path.splitext(data_path)[1] == ".pkl" :

            with open(data_path, "rb") as f :
                raw_data = pickle.load(f)

            documents = []
            for key, value in raw_data.items() :
                
                if self.config.embedding_target_value.lower() == "key" : 

                    documents.append(key)
                
                elif self.config.embedding_target_value.lower() == "value" :

                    documents.append(value)

            return documents

    def embedding_data(self, document, tokenizer, model) :
        '''
        document : embedding을 적용할 변수의 데이터 (string)
        '''

        inputs = tokenizer(document, max_length = self.config.model_max_length, truncation = True, return_tensors = "pt")

        with torch.no_grad() :
            output = model(**inputs)

        embedding = output.last_hidden_state[ : , 0, : ].tolist()

        return embedding

    def insert_embedding_data(self, documents, tokenizer, model) :

        max_length = self.config.max_length
        num_documents = len(documents)
        model_dim = self.config.dim

        fields = [
            FieldSchema(name = "documents", dtype= DataType.VARCHAR, is_primary = True, auto_id = False, max_length = max_length),
            FieldSchema(name = "embeddings", dtype = DataType.FLOAT_VECTOR, dim = model_dim)
        ]

        schema = CollectionSchema(fields = fields, description = '')
        collection = Collection(name = self.collection_name, schema = schema)

        for i in range(len(documents)) :

            document = documents[i]
            embedding = self.embedding_data(document, tokenizer, model)
            insert_document = [document]

            entities = [insert_document, embedding]

            collection.insert(entities)
            

        collection.flush()

        index_params = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": num_documents}}
        collection.create_index(field_name = 'embeddings', index_params = index_params)
        collection.load()











