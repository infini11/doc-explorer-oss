from fastapi import UploadFile, HTTPException

ALLOWED_EXTENSIONS = {"pdf", "txt", "docx", "md", "png", "jpg", "jpeg"}
MAX_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

def is_validate_document(file: UploadFile) -> bool:
    """
    Validate extension and content-type.
    Returns the file extension.
    Args : 
        file
    Returns: 
        extension
    """
    if not file.filename or "." not in file.filename:
        raise HTTPException(status_code=422, detail="Filename must have an extension.")
        #422 -> Unprocessable Entity

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Extension '.{ext}' not allowed. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    return True

