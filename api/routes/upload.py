import uuid
import logging
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException

from api.schemas.document import UploadDocument
from api.services.validation import is_validate_document
from api.services.storage import save_file

router = APIRouter(tags=["ingestion"])
logger = logging.getLogger(__name__)

@router.post("/upload", response_model=UploadDocument, status_code=201)
async def upload_document(file: UploadFile = File(...)):
    document_id = str(uuid.uuid4())

    logger.info("upload_received", extra={"document_id": document_id, "filename": file.filename})

    resp = is_validate_document(file=file)
    if resp:
        try:
            path, size, checksum = await save_file(file, document_id)
        except Exception as e:
            logger.error("upload_failed", extra={"document_id": document_id, "error": str(e)})
            raise HTTPException(status_code=500, detail="Failed to store document.")

        logger.info("upload_success", extra={
            "document_id": document_id,
            "filename": file.filename,
            "size_bytes": size,
            "checksum": checksum,
        })

        return UploadDocument(
            document_id=document_id,
            filename=file.filename,
            size_bytes=size,
            checksum_sha256=checksum,
            stored_path=str(path),
            uploaded_at=datetime.now(datetime.timezone.utc),
        )
    
    return
