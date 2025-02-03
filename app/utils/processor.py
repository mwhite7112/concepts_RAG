import re
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

def clean_latex(text: str) -> str:
    text = re.sub(r'%.*?\n', '\n', text)
    text = re.sub(r'\\[a-zA-Z]+{([^}]*)}', r'\1', text)
    return text

def process_chapters(base_dir: Path) -> Chroma:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    texts = []
    metadatas = []
    
    for chapter_dir in sorted(base_dir.iterdir()):
        if not chapter_dir.is_dir():
            continue
            
        chapter_name = chapter_dir.name
        
        for tex_file in chapter_dir.glob('**/*.tex'):
            with open(tex_file, 'r') as f:
                content = f.read()
                cleaned_content = clean_latex(content)
                chunks = text_splitter.split_text(cleaned_content)
                texts.extend(chunks)
                chunk_metadatas = [{
                    "chapter": chapter_name,
                    "source": str(tex_file.relative_to(base_dir)),
                    "chunk_index": i
                } for i in range(len(chunks))]
                metadatas.extend(chunk_metadatas)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore