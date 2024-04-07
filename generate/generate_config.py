from dataclasses import dataclass
from dataclasses import field

@dataclass
class GenerateConfig :

    model_path : str = field(default = "Upstage/SOLAR-10.7B-Instruct-v1.0", metadata = {"help" : "generate model path"})

    max_length : int = field(default = 4096, metadata = {"help" : "Maximum length of strings allowed to be inserted"})
