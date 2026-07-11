import os
from pathlib import Path
from markitdown import MarkItDown 
from config import DOCUMENTS_DIR

def load_documents(folder_path: Path) -> list[dict]:
    """
    itterate over all files in the folder and uses MarkItDown
    to convert any suuported document into Markdown.
    Return a list of dictionaries each containing: 
    'name' (document name)
    'content' (Markdown content)
    """
    documents = []

    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Warning: Directory not found-> {folder_path}")
        return documents

    # intializing converter
    md_converter = MarkItDown()

    for item in folder_path.iterdir():
        if not item.is_file():
            continue
        try:
            #markitdown autometically detcts the file type using file extension and and converts it to markdown
            #supports: PDF, DOCX, XLSX, PPTX, HTML, TXT, CSV, JSON, XML etc

            result = md_converter.convert(str(item))
            md_text = result.text_content

            documents.append({
                "file_name": item.name,
                "raw_text":md_text
            })
        
        except Exception as e:
            print(f"Failed to convert{item.name}.Error:{e}")
            continue

    return documents

if __name__ =="__main__":
    print(f"Loading doccuments from{DOCUMENTS_DIR}\n")
    docs = load_documents(DOCUMENTS_DIR)

    print(f"\nLoaded {len(docs)} documents total.:")
    for doc in docs:
        print(f"\n---{doc['file_name']}---")
        print(doc['raw_text'][:500])
        print("...\n")
        