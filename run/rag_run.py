import os
import sys

sys.path.append(os.path.dirname((os.path.abspath(os.path.dirname((__file__))))))

from rag.rag_pipeline import RagPipeline

from retriever.retriever_config import RetrieverConfig
from generate.generate_config import GenerateConfig

def main() :

    rag_pipeline = RagPipeline(RetrieverConfig, GenerateConfig)

    embedding_tokenizer, embedding_model = rag_pipeline.load_retriever_model()

    generate_tokenizer, generate_model = rag_pipeline.load_generate_model()

    rag_pipeline.chat(embedding_tokenizer, embedding_model, generate_tokenizer, generate_model)


if __name__ == '__main__':
    main()
