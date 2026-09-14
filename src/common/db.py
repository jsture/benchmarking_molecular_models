import os
from pathlib import Path
from peewee import Model, Proxy, CharField, FloatField, SqliteDatabase, IntegerField
from playhouse.shortcuts import ThreadSafeDatabaseMetadata
from .config import EmbeddingConfig, BASE_DIR

proxy = Proxy()


class BaseModel(Model):
    class Meta:
        database = proxy
        model_metadata_class = ThreadSafeDatabaseMetadata


class EmbeddingMeta(BaseModel):
    dataset = CharField()
    embedder = CharField()
    embedding_loc = CharField()
    embedding_time = FloatField()


class ClusterizationReport(BaseModel):
    dataset = CharField()
    embedder = CharField()

    rand_score = FloatField(null=True)
    davies_bouldin_score = FloatField(null=True)
    noise_perc = FloatField(null=True)


class ClassificationReport(BaseModel):
    dataset = CharField()
    task = CharField()
    embedder = CharField()

    model = CharField()
    hyperparams = CharField()
    library_hash = CharField()

    cv_metric_name = CharField()
    cv_metric = FloatField()

    test_metric_name = CharField()
    test_metric = FloatField()


class Runtime(BaseModel):
    dataset = CharField()
    embedder = CharField()
    device = CharField()

    mean_runtime = FloatField()
    std_runtime = FloatField()
    n_samples = IntegerField()


def init_db(config: EmbeddingConfig):
    """Initializes the database schema once before running evaluations."""
    db_path = str(Path(config.database) if Path(config.database).is_absolute() else BASE_DIR / config.database)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db = SqliteDatabase(
        db_path,
        timeout=60.0,
        pragmas={'journal_mode': 'wal', 'cache_size': -1024 * 64}
    )
    proxy.initialize(db)
    with db:
        proxy.create_tables([EmbeddingMeta, Runtime, ClassificationReport, ClusterizationReport], safe=True)
    return db


def close_db():
    if not proxy.is_closed():
        proxy.close()


class DbContex:
    def __init__(self, config: EmbeddingConfig):
        self._config = config
        self._database = None

    def __enter__(self):
        db_path = str(Path(self._config.database) if Path(self._config.database).is_absolute() else BASE_DIR / self._config.database)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._database = SqliteDatabase(
            db_path,
            timeout=60.0,
            pragmas={'journal_mode': 'wal', 'cache_size': -1024 * 64}
        )
        proxy.initialize(self._database)
        if self._database.is_closed():
            self._database.connect()
        return self._database

    def __exit__(self, *args, **kwargs):
        if self._database and not self._database.is_closed():
            self._database.close()


