import os
import sys

sys.path.append(os.path.dirname((os.path.abspath(os.path.dirname((__file__))))))

from embedding.embedding_config import EmbeddingConfig
from embedding.embedding_pipeline import MilvusEmbedding

def main() :

    # 1. milvus pipeline setting
    milvus_embedding = MilvusEmbedding(EmbeddingConfig)

    # 2. vectorDB connection
    milvus_embedding.create_connection()

    # 3. vectorDB check collection name
    milvus_embedding.check_collection_name()

    # 4. load dataset
    documents = milvus_embedding.load_data()

    # 5. load model
    tokenizer, model = milvus_embedding.load_model()

    # 6. insert embedding dataset
    milvus_embedding.insert_embedding_data(documents, tokenizer, model)

    print("embedding end")


if __name__ == '__main__':
    main()
