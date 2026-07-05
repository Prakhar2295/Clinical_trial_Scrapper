import logging
import os
import signal
import socket
import subprocess
import time
from urllib.parse import urlparse

import weaviate
from weaviate.classes.aggregate import GroupByAggregate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.embedded import EmbeddedOptions

from app.core.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "ClinicalTrialChunk"

EMBEDDED_HOST = "127.0.0.1"
EMBEDDED_PORT = 8079
EMBEDDED_GRPC_PORT = 50060


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _kill_stale_embedded_weaviate() -> None:
    """Kills any orphaned embedded Weaviate binary (e.g. left behind by a dev-server
    reload that didn't shut it down cleanly), so a fresh instance can bind the ports."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "weaviate-embedded"], capture_output=True, text=True, timeout=5
        )
        pids = [int(p) for p in result.stdout.split() if p.strip()]
        for pid in pids:
            os.kill(pid, signal.SIGKILL)
        if pids:
            logger.warning("Killed stale embedded Weaviate process(es): %s", pids)
            time.sleep(1)  # give the OS a moment to release the ports
    except Exception:
        logger.exception("Failed to clean up stale embedded Weaviate process(es)")


PROPERTIES = [
    Property(name="filename", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
    # ground truth label; FIELD tokenization so category filters are exact-match, not fuzzy
    Property(name="category", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
    Property(name="chunk_index", data_type=DataType.INT),
    # "head" | "tail"; FIELD tokenization so chunk_position filters are exact-match
    Property(name="chunk_position", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
    Property(name="page_range", data_type=DataType.TEXT),  # e.g. "1-3"
    Property(name="content", data_type=DataType.TEXT),  # raw chunk text — word tokenization for BM25/hybrid search
    # "bootstrap" | "feedback"; FIELD tokenization for exact-match filters
    Property(name="source_type", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
]


def _connect() -> weaviate.WeaviateClient:
    if settings.weaviate_url == "embedded" or settings.use_weaviate_embedded:
        http_open = _port_open(EMBEDDED_HOST, EMBEDDED_PORT)
        grpc_open = _port_open(EMBEDDED_HOST, EMBEDDED_GRPC_PORT)

        if http_open and grpc_open:
            # A healthy embedded instance (e.g. from a previous process that didn't shut
            # down cleanly) is already listening. Starting a new one would crash with
            # WeaviateStartUpError, so reuse the existing instance instead.
            logger.info(
                "Embedded Weaviate already running on %s:%d — reusing existing instance.",
                EMBEDDED_HOST,
                EMBEDDED_PORT,
            )
            return weaviate.connect_to_local(
                host=EMBEDDED_HOST, port=EMBEDDED_PORT, grpc_port=EMBEDDED_GRPC_PORT
            )

        if http_open != grpc_open:
            # One port is up and the other isn't — a broken/orphaned process (e.g. left
            # behind by a dev-server auto-reload) is holding a port half-alive. Reusing
            # it fails on whichever protocol is down; starting fresh fails on the port
            # collision. Clean up the stale process first, then start fresh.
            logger.warning(
                "Embedded Weaviate is in an inconsistent state (http_open=%s, grpc_open=%s) "
                "— cleaning up stale process before starting a fresh instance.",
                http_open,
                grpc_open,
            )
            _kill_stale_embedded_weaviate()

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
        if self.client.collections.exists(COLLECTION_NAME):
            config = self.client.collections.get(COLLECTION_NAME).config.get()
            filename_prop = next((p for p in config.properties if p.name == "filename"), None)

            if filename_prop is not None and filename_prop.tokenization.value.lower() == "field":
                return  # schema already has correct tokenization, nothing to do

            logger.warning(
                "ClinicalTrialChunk collection recreated with correct tokenization. "
                "All data has been cleared and must be re-ingested."
            )
            self.client.collections.delete(COLLECTION_NAME)

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

    def query_hybrid(self, query_text: str, vector: list[float], chunk_position: str, limit: int) -> list[dict]:
        results = self.collection.query.hybrid(
            query=query_text,
            vector=vector,
            limit=limit,
            filters=Filter.by_property("chunk_position").equal(chunk_position),
            return_metadata=MetadataQuery(score=True),
            return_properties=["content", "category", "chunk_position", "filename"],
        )
        return [
            {
                "content": obj.properties.get("content", ""),
                "category": obj.properties.get("category"),
                "score": obj.metadata.score,
                "filename": obj.properties.get("filename"),
                "chunk_position": obj.properties.get("chunk_position"),
            }
            for obj in results.objects
        ]

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

    def purge_all(self) -> int:
        """Deletes the entire collection and recreates it empty. Returns the deleted count."""
        deleted_count = self.count()
        self.client.collections.delete(COLLECTION_NAME)
        self._ensure_schema()
        return deleted_count

    def delete_by_category(self, category: str) -> int:
        result = self.collection.data.delete_many(where=Filter.by_property("category").equal(category))
        return result.successful

    def delete_by_filename(self, filename: str) -> int:
        result = self.collection.data.delete_many(where=Filter.by_property("filename").equal(filename))
        return result.successful

    def list_files(self) -> list[dict]:
        results = self.collection.query.fetch_objects(
            limit=10_000,
            return_properties=["filename", "category", "chunk_position", "source_type"],
        )

        files: dict[str, dict] = {}
        for obj in results.objects:
            props = obj.properties
            fname = props.get("filename", "unknown")
            if fname not in files:
                files[fname] = {
                    "filename": fname,
                    "category": props.get("category", "unknown"),
                    "source_type": props.get("source_type", "unknown"),
                    "head_chunks": 0,
                    "tail_chunks": 0,
                    "total_chunks": 0,
                }
            files[fname]["total_chunks"] += 1
            if props.get("chunk_position") == "head":
                files[fname]["head_chunks"] += 1
            elif props.get("chunk_position") == "tail":
                files[fname]["tail_chunks"] += 1

        return sorted(files.values(), key=lambda x: x["filename"])

    def stats(self) -> dict:
        total_chunks = self.count()

        by_category_result = self.collection.aggregate.over_all(
            total_count=True, group_by=GroupByAggregate(prop="category")
        )
        by_category = {group.grouped_by.value: group.total_count for group in by_category_result.groups}

        by_position_result = self.collection.aggregate.over_all(
            total_count=True, group_by=GroupByAggregate(prop="chunk_position")
        )
        by_position = {group.grouped_by.value: group.total_count for group in by_position_result.groups}

        all_objects = self.collection.query.fetch_objects(limit=10_000)
        unique_files = len({obj.properties.get("filename") for obj in all_objects.objects})

        return {
            "total_chunks": total_chunks,
            "by_category": by_category,
            "by_position": by_position,
            "unique_files": unique_files,
        }


_store: WeaviateStore | None = None


def get_weaviate_store() -> WeaviateStore:
    global _store
    if _store is None:
        _store = WeaviateStore()
    return _store
