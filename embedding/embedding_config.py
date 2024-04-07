from dataclasses import dataclass
from dataclasses import field

@dataclass
class EmbeddingConfig :

    milvus_alilas : str = field(default = "", metadata = {"help" : "milvus connect alilas"})
    milvus_uri : str = field(default = "", metadata = {"help" : "milvus connect uri"})
    collection_name : str = field(default = "embedding_collection", metadata = {"help" : "milvus collection name"})

    data_path : str = field(default = "", metadata = {"help" : "upload data set path"})

    embedding_target_value : str = field(default = "key", metadata = {"help" : "json / pickle file embedding target variable"})

    model_path : str = field(default = "bert-base-uncased", metadata = {"help" : "embedding model path"})

    model_max_length : int = field(default = 512, metadata = {"help" : "The maximum length for the inputs to the transformers model"})
    dim : int = field(default = 768, metadata = {"help" : "Dimension of the vector"})

    max_length : int = field(default = 16384, metadata = {"help" : "Maximum length of strings allowed to be inserted"})
