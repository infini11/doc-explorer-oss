from pydantic import BaseModel
from datetime import datetime

class UploadDocument(BaseModel):
    document_id: str
    filename : str
    size_bytes: int
    checksum_sha256: str
    stored_path: str
    uploaded_at: datetime