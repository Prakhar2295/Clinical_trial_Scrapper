from pydantic import BaseModel


class IngestAcceptedResponse(BaseModel):
    status: str
    filename: str
    message: str
