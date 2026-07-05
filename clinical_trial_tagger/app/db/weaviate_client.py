from urllib.parse import urlparse

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.embedded import EmbeddedOptions

from app.core.config import settings

COLLECTION_NAME = "ClinicalTrialChunk"

PROPERTIES = [
    Property(name="nct_id", data_type=DataType.TEXT),
    Property(name="filename", data_type=DataType.TEXT),
    Property(name="category", data_type=DataType.TEXT),  # ground truth label
    Property(name="chunk_index", data_type=DataType.INT),
    Property(name="chunk_position", data_type=DataType.TEXT),  # "head" | "tail"
    Property(name="page_range", data_type=DataType.TEXT),  # e.g. "1-3"
    Property(name="content", data_type=DataType.TEXT),  # raw chunk text
    Property(name="source_type", data_type=DataType.TEXT),  # "bootstrap" | "feedback"
]


def _connect() -> weaviate.WeaviateClient:
    if settings.weaviate_url == "embedded" or settings.use_weaviate_embedded:
        client = weaviate.WeaviateClient(
            embedded_options=EmbeddedOptions(persistence_data_path="./weaviate_data"),
        )
        client.connect()
        return client

    parsed = urlparse(settings.weaviate_url)
    auth = Auth.api_key(settings.weaviate_api_key) if settings.weaviate_api_key else None
    return weaviate.connect_to_custom(
        http_host=parsed.hostname or "localhost",
        http_port=parsed.port or 8080,
        http_secure=parsed.scheme == "https",
        grpc_host=parsed.hostname or "localhost",
        grpc_port=50051,
        grpc_secure=parsed.scheme == "https",
        auth_credentials=auth,
        skip_init_checks=True,
    )


class WeaviateStore:
    """Owns the ClinicalTrialChunk collection: schema setup + CRUD."""

    def __init__(self):
        self.client = _connect()
        if self.client.is_ready():
            self._ensure_schema()

    def _ensure_schema(self):
        if not self.client.collections.exists(COLLECTION_NAME):
            self.client.collections.create(
                name=COLLECTION_NAME,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=PROPERTIES,
            )

    @property
    def collection(self):
        return self.client.collections.get(COLLECTION_NAME)

    def is_ready(self) -> bool:
        try:
            return self.client.is_ready()
        except Exception:
            return False

    def close(self):
        self.client.close()

    def add_chunk(self, vector: list[float], properties: dict) -> str:
        uuid = self.collection.data.insert(properties=properties, vector=vector)
        return str(uuid)

    def add_chunks_batch(self, items: list[dict]) -> None:
        """items: [{"vector": [...], "properties": {...}}]"""
        with self.collection.batch.dynamic() as batch:
            for item in items:
                batch.add_object(properties=item["properties"], vector=item["vector"])

    def query_near_vector(self, vector: list[float], limit: int = 10) -> list[dict]:
        result = self.collection.query.near_vector(
            near_vector=vector,
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
        )
        return [
            {**obj.properties, "uuid": str(obj.uuid), "distance": obj.metadata.distance}
            for obj in result.objects
        ]

    def find_by_filename(self, filename: str) -> list[dict]:
        result = self.collection.query.fetch_objects(
            filters=Filter.by_property("filename").equal(filename),
            limit=1000,
        )
        return [{**obj.properties, "uuid": str(obj.uuid)} for obj in result.objects]

    def update_category(self, uuid: str, category: str, source_type: str = "feedback") -> None:
        self.collection.data.update(
            uuid=uuid,
            properties={"category": category, "source_type": source_type},
        )

    def count(self) -> int:
        return self.collection.aggregate.over_all(total_count=True).total_count

    def count_by_category(self) -> dict:
        result = self.collection.aggregate.over_all(total_count=True, group_by="category")
        return {group.grouped_by.value: group.total_count for group in result.groups}


_store: WeaviateStore | None = None


def get_weaviate_store() -> WeaviateStore:
    global _store
    if _store is None:
        _store = WeaviateStore()
    return _store
