from dataclasses import dataclass
from dataclasses import field

@dataclass
class RetrieverConfig :
    
    milvus_alilas : str = field(default = "", metadata = {"help" : "milvus connect alilas"})
    milvus_uri : str = field(default = "", metadata = {"help" : "milvus connect uri"})
    
    model_path : str = field(default = "bert-base-uncased", metadata = {"help" : "embedding model path"})

    collection_name : str = field(default = "embedding_collection", metadata = {"help" : "milvus collection name"})
    
    limit : int = field(default = 10, metadata = {"help" : "number of the nearest records to return"})
    round_decimal : int = field(default = -1, metadata = {"help" : "number of the decimal places of the returned distance"})

    top_k : int = field(default = 1, metadata = {"help" : "the maximum number of documents to retrieve"})
