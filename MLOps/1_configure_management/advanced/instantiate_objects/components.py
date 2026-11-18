"""
Components module providing target classes and functions for Hydra instantiation.
"""

class DatabaseConnection:
    """Base interface for database connections."""
    def connect(self) -> str:
        raise NotImplementedError


class MySQLConnection(DatabaseConnection):
    def __init__(self, host: str = "localhost", port: int = 3306, user: str = "root", timeout: int = 10):
        self.host = host
        self.port = port
        self.user = user
        self.timeout = timeout

    def connect(self) -> str:
        return f"[MySQL] Connected to {self.host}:{self.port} as user='{self.user}' (timeout={self.timeout}s)"

    def __repr__(self) -> str:
        return f"MySQLConnection(host='{self.host}', port={self.port}, user='{self.user}', timeout={self.timeout})"


class PostgreSQLConnection(DatabaseConnection):
    def __init__(self, host: str = "localhost", port: int = 5432, database: str = "mlops_db", user: str = "postgres", timeout: int = 15):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.timeout = timeout

    def connect(self) -> str:
        return f"[PostgreSQL] Connected to {self.host}:{self.port}/{self.database} as user='{self.user}' (timeout={self.timeout}s)"

    def __repr__(self) -> str:
        return f"PostgreSQLConnection(host='{self.host}', port={self.port}, db='{self.database}', user='{self.user}', timeout={self.timeout})"


class Client:
    def __init__(self, endpoint: str, retry_count: int = 3):
        self.endpoint = endpoint
        self.retry_count = retry_count

    def __repr__(self) -> str:
        return f"Client(endpoint='{self.endpoint}', retry_count={self.retry_count})"


class Pipeline:
    """Demonstrates recursive instantiation with a nested Client object."""
    def __init__(self, name: str, client: Client, batch_size: int = 32):
        self.name = name
        self.client = client
        self.batch_size = batch_size

    def run(self) -> str:
        return f"Pipeline '{self.name}' running with batch_size={self.batch_size} via {self.client}"

    def __repr__(self) -> str:
        return f"Pipeline(name='{self.name}', client={self.client}, batch_size={self.batch_size})"


class ModelTrainer:
    """Demonstrates partial instantiation where runtime parameters (e.g. dataset) are supplied later."""
    def __init__(self, model_name: str, learning_rate: float, dataset: str):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.dataset = dataset

    def train(self) -> str:
        return f"Training {self.model_name} on dataset='{self.dataset}' with lr={self.learning_rate}"

    def __repr__(self) -> str:
        return f"ModelTrainer(model='{self.model_name}', lr={self.learning_rate}, dataset='{self.dataset}')"


def calculate_accuracy(correct: int, total: int) -> float:
    """Demonstrates calling a standalone function via hydra.utils.call / instantiate."""
    if total == 0:
        return 0.0
    return round((correct / total) * 100, 2)
