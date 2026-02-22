"""
Responsibilities:
  1. Read plain text (from a file or directly)
  2. Split into chunks (LangChain RecursiveCharacterTextSplitter)
  3. Generate embeddings for each chunk (HuggingFace)
  4. Build and save a local FAISS index

Why these three steps?
  - The plain text is too long to be sent in its entirety to the LLM
  - It is split into small pieces (chunks), each of which has meaning
  - Each chunk is transformed into a numerical vector (embedding)
  - FAISS allows the chunks closest to a given question to be found quickly
    (search by vector similarity)

Flux :
  text file
      ↓
  [load_text]       raw reading
      ↓
  [chunk_text]      cutting into pieces of 512 characters
      ↓
  [build_index]     embedding of each chunk → FAISS index
      ↓
  storage/faiss_index/{document_id}/
"""
import sys
import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Size of a chunk in characters.
# 512 is a good compromise: large enough to provide context,
# small enough to fit within the LLM's attention window.
CHUNK_SIZE = 512

# Overlap between two consecutive chunks.
# Avoid splitting a sentence in two and losing the meaning at the junction.
CHUNK_OVERLAP = 64

# Lightweight open-source embedding model (~90MB, dimension 384).
# Automatically downloaded from HuggingFace on first call.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# FAISS index storage directory
FAISS_DIR = Path("storage/faiss_index")


class Chunker:
    """Responsible for:
        - Reading raw text
        - Splitting it into chunks
        - Generating embeddings for each chunk
        - Building and saving a local FAISS index
    """

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " "], # priority order for splitting for coherence and context preservation
        )
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def load_text(self, file_path: str) -> str:
        """Reads the raw content of a text file.
           Supports: .txt, .md, and any file encoded in UTF-8.

        Args:
            file_path (str): input file path

        Raises:
            FileNotFoundError: error if the file does not exist

        Returns:
            str: the raw content of the file
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"file not found : {file_path}")
        text = path.read_text(encoding="utf-8")
        logger.info(f"File loaded : {path.name} ({len(text)} characters)")
        return text

    def chunk_text(self, text: str, source: str = "unknown") -> list[Document]:
        """Split the text into chunks with overlap.
            RecursiveCharacterTextSplitter Preserves meaning as long as possible before splitting.

        Args:
            text (str): input text
            source (str, optional): source of the text (file path or document_id). Defaults to "unknown".

        Returns:
            list[Document]: list of LangChain Documents (text + metadata).
        """
        chunks = self.text_splitter.split_text(text)

        # Encapsulate in LangChain Documents with metadata
        documents = [
            Document(
                page_content=chunk,
                metadata={"source": source, "chunk_index": i}
            )
            for i, chunk in enumerate(chunks)
        ]

        logger.info(f"Chunking : {len(documents)} genered chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
        return documents

    def build_index(self, documents: list[Document], document_id: str) -> FAISS:
        """ Generates embeddings for each chunk and builds the FAISS index.

        Each chunk is transformed into a 384-dimensional vector.
        FAISS indexes these vectors to enable fast searching
        by cosine similarity (finding the chunks closest
        to a given question).

        The index is saved in storage/faiss_index/{document_id}/
        to be reused by inference/serve.py.

        Args:
            documents (list[Document]): documents to index
            document_id (str): document identifier used for naming the FAISS index directory

        Returns:
            FAISS: document index for similarity search
        """
        logger.info(f"Loading the embedding model : {EMBEDDING_MODEL}")
        logger.info(f"Generation of embeddings for {len(documents)} chunks...")
        vectorstore = FAISS.from_documents(documents, self.embedding_model)

        # Save the FAISS index to disk for later use in inference/serve.py
        save_path = FAISS_DIR / document_id
        save_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(save_path))

        logger.info(f"FAISS index saved → {save_path}")
        return vectorstore

    def load_index(self, document_id: str) -> FAISS:
        """Loads an existing FAISS index from disk.
        Used by inference/serve.py for searching when questions are asked.

        Args:
            document_id (str): document identifier used for naming the FAISS index directory

        Raises:
            FileNotFoundError: error if the FAISS index does not exist

        Returns:
            FAISS: document index for similarity search
        """
        index_path = FAISS_DIR / document_id
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS Index not found : {index_path}")

        vectorstore = FAISS.load_local(
            str(index_path),
            self.embedding_model,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"FAISS Index loaded ← {index_path}")
        return vectorstore

    def run(
        self,
        file_path: str = None,
        document_id: str = None,
        text: str = None,
        ) -> tuple[list[Document], FAISS]:
        """ Complete pipeline: reading → chunking → embeddings → FAISS.
            Accepts either a file path or plain text directly.

        Args:
            file_path (str, optional): file path to be processed. Defaults to None.
            document_id (str, optional): document identifier used for naming the FAISS index directory. Defaults to None.
            text (str, optional): text to be processed directly. Defaults to None.

        Raises:
            ValueError: if neither file_path nor text is provided

        Returns:
            tuple[list[Document], FAISS]: documents and FAISS vector store
        """
        if text is None and file_path is None:
            raise ValueError("Provide either file_path or text")

        if text is None:
            text = self.load_text(file_path)

        source = file_path or document_id or "inline"
        documents = self.chunk_text(text, source=source)
        vectorstore = self.build_index(documents, document_id or "default")

        return documents, vectorstore
    

if __name__ == "__main__":
    chuncker = Chunker()
    logging.basicConfig(level=logging.INFO)

    # from a text file
    if len(sys.argv) >= 3:
        file_path  = sys.argv[1]
        doc_id     = sys.argv[2]
        docs, vs   = chuncker.run(file_path=file_path, document_id=doc_id)

    # from direct text input (for testing/demo purposes)
    else:
        print("ℹ️  No argument provided — using demo text")
        sample_text = """
        Type 2 diabetes is a chronic metabolic disease characterized by high blood sugar.
        It is caused by insulin resistance and progressive loss of beta cell function.
        Common symptoms include fatigue, frequent urination, increased thirst, and blurred vision.
        Treatment includes lifestyle modifications, metformin as first-line medication,
        and blood glucose monitoring. If uncontrolled, it can lead to cardiovascular disease,
        kidney failure, and neuropathy.

        Hypertension, also known as high blood pressure, is a condition where the force
        of blood against artery walls is too high. It is often called the silent killer
        because it has no obvious symptoms. Risk factors include obesity, sedentary lifestyle,
        high sodium diet, and family history. Treatment includes ACE inhibitors, beta-blockers,
        and lifestyle changes such as reduced salt intake and regular exercise.
        """
        docs, vs = chuncker.run(text=sample_text, document_id="demo-medical-001")

    print(f"\n✅ Pipeline completed")
    print(f"   {len(docs)} generated chunks")
    print(f"\n   First chunk :")
    print(f"   {docs[0].page_content[:150]}...")

    # Démontrer la recherche par similarité
    print(f"\n🔍 Search test: 'symptoms of diabetes''")
    results = vs.similarity_search("symptoms of diabetes", k=2)
    for i, r in enumerate(results):
        print(f"\n   Result {i+1} (chunk #{r.metadata['chunk_index']}) :")
        print(f"   {r.page_content[:200]}...")