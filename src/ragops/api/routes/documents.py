from __future__ import annotations

from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    status,
)

from ragops.api.deps import get_ingestion, require_admin
from ragops.api.dispatch import dispatch_ingest
from ragops.api.schemas import IngestCorpusResponse
from ragops.config import get_settings
from ragops.ingestion.parsers import SUPPORTED_SUFFIXES
from ragops.ingestion.registry import DocumentRecord
from ragops.types import Chunk

router = APIRouter(prefix="/api/documents", tags=["documents"])



@router.get("", response_model=list[DocumentRecord])
async def list_documents() -> list[DocumentRecord]:
    svc = get_ingestion()
    return svc.registry.list(svc.index_name)


@router.post("", response_model=DocumentRecord, status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_admin)])
async def upload_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    acl_groups: str = Form("public"),
    title: str | None = Form(None),
    doc_id: str | None = Form(None),
) -> DocumentRecord:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(415, f"Unsupported file type '{suffix}'. Allowed: {sorted(SUPPORTED_SUFFIXES)}")
    data = await file.read()
    limit_mb = get_settings().max_upload_mb
    if len(data) > limit_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {limit_mb} MB (MAX_UPLOAD_MB)")
    if not data:
        raise HTTPException(400, "Empty file")
    groups = [g.strip() for g in acl_groups.split(",") if g.strip()]
    svc = get_ingestion()
    rec = svc.register(filename=file.filename or f"upload{suffix}", data=data, acl_groups=groups, title=title or None, doc_id=doc_id or None)
    dispatch_ingest(background, [rec.doc_id])
    return rec


@router.post("/ingest-corpus", response_model=IngestCorpusResponse, status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_admin)])
async def ingest_corpus(background: BackgroundTasks) -> IngestCorpusResponse:
    ids = get_ingestion().register_corpus()
    dispatch_ingest(background, ids)
    return IngestCorpusResponse(queued=ids)


@router.get("/{doc_id}", response_model=DocumentRecord)
async def get_document(doc_id: str) -> DocumentRecord:
    svc = get_ingestion()
    rec = svc.registry.get(svc.index_name, doc_id)
    if rec is None:
        raise HTTPException(404, f"Document '{doc_id}' not found")
    return rec


@router.get("/{doc_id}/chunks", response_model=list[Chunk])
async def get_chunks(doc_id: str) -> list[Chunk]:
    return await get_ingestion().store.document_chunks(doc_id)


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_admin)])
async def delete_document(doc_id: str) -> Response:
    await get_ingestion().delete(doc_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
