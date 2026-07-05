from enum import Enum
from typing import Optional

from pydantic import BaseModel


class CollectionAction(str, Enum):
    purge_all = "purge_all"
    delete_by_category = "delete_by_category"
    delete_by_filename = "delete_by_filename"
    stats = "stats"
    list_files = "list_files"


class CollectionManageRequest(BaseModel):
    action: CollectionAction
    category: Optional[str] = None
    filename: Optional[str] = None


class CollectionManageResponse(BaseModel):
    action: str
    status: str
    message: str
    deleted_chunks: Optional[int] = None
    stats: Optional[dict] = None
    files: Optional[list[dict]] = None
