from pathlib import Path
import sys
import gc
import pprint
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_pymupdf4llm import PyMuPDF4LLMParser, pymupdf4llm_loader
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_pymupdf4llm import PyMuPDF4LLMParser
import logging


# 1. Grab paths safely using Path
project_root = Path(__file__).resolve().parent.parent
assets_dir = project_root / "assets"

# 2. Configure the Generic Loader
loader = GenericLoader(
    blob_loader=FileSystemBlobLoader(
        path=str(assets_dir),  
        glob="*.pdf",
    ),
    blob_parser=PyMuPDF4LLMParser(
        mode="single",
        pages_delimiter="\n\f",
        use_layout=False,
        table_strategy="lines",
    ),
)

# 3. Stream and parse files sequentially
processed_count = 0
try:
    # Use lazy_load() to stream documents one-by-one from the disk generator
    for doc in loader.lazy_load():
        processed_count += 1
        
        source_path = Path(doc.metadata.get("source"))
        base_name = source_path.stem 
        output_path = assets_dir / f"{base_name}_content.md"
        try:
            # Open a write stream context manager to write cleanly to disk
            with open(output_path, mode="w", encoding="utf-8") as file:
                for line in doc.page_content.splitlines():
                    file.write(line + "\n")
            print(f"{processed_count} Successfully saved: {output_path.name}")
        except IOError as e:
            print(f"Error writing file {output_path.name}: {e}", file=sys.stderr)
        finally:
            # CRITICAL FOR BULK RAG: Actively purge parsed data structures from RAM
            del doc
            gc.collect()
except Exception as e:
    print(f"Critical error inside the loader stream pipeline: {e}", file=sys.stderr)
print(f"\n Process Complete! Total files accurately structured: {processed_count}")

