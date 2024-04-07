import os
import sys

sys.path.append(os.path.dirname((os.path.abspath(os.path.dirname((__file__))))))

import torch

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

from transformers import AutoModel, AutoTokenizer

class MilvusRetriever :

    def __init__(self, config) :

        self.config = config

    def create_connection(self) :

        milvus_alias = self.config.milvus_alilas
        milvus_uri = self.config.milvus_uri

        connections.connect(alias = milvus_alias, uri = milvus_uri)      

    def load_model(self) :

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        model = AutoModel.from_pretrained(self.config.model_path)

        return tokenizer, model

    def query_embedding(self, query, tokenizer, model) :
        '''
        document : embedding을 적용할 변수의 데이터 (string)
        '''

        inputs = tokenizer(query, return_tensors = "pt")

        with torch.no_grad() :
            output = model(**inputs)

        embedding = output.last_hidden_state[ : , 0, : ].tolist()

        return embedding

    def milvus_retriever(self, query, tokenizer, model) :

        collection_name = self.config.collection_name

        collection = Collection(name = collection_name)
        collection.load() # 이미 로드된 컬렉션을 로드하려고 하면 "로드 성공"을 반환

        embedding = self.query_embedding(query, tokenizer, model)

        search_params = {"metric_type": "IP", "params": {"nprobe": 8,  "offset": 1}}
        retriever_result = collection.search(embedding, "embeddings", search_params, 
                                            limit = self.config.limit, 
                                            round_decimal = self.config.round_decimal,
                                            output_fields = ["documents"])

        documents = []
        for i in range(self.config.top_k) :

            document = retriever_result[0][i].entity.get("documents")
            documents.append(document)

        return documents
