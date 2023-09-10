from dataclasses import dataclass

@dataclass
class DataIngestArtifact:
    train_file_path: str
    test_file_path: str
    