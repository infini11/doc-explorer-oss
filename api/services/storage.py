import hashlib
from typing import Tuple
from pathlib import Path
from datetime import datetime
from fastapi import UploadFile

UPLOAD_DIR = Path("storage/uploads")

async def save_file(file: UploadFile, document_id: str) -> Tuple[Path, int, str]:
    """Persist file to disk.

    Args:
        file (UploadFile): file
        document_id (str): document id

    Returns:
        Tuple[Path, int, str]: (path, size_bytes, sha256)
    """
    dest_dir = UPLOAD_DIR / datetime.utcnow().strftime("%Y/%m/%d") / document_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file.filename

    sha256 = hashlib.sha256()
    size = 0

    with dest.open("wb") as f:
        while chunk := await file.read(65_536):
            sha256.update(chunk)
            size += len(chunk)
            f.write(chunk)

    return dest, size, sha256.hexdigest()