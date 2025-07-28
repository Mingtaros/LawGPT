# 1) Disable Streamlit’s file watcher so it won’t introspect torch.classes
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"  

# 2) Monkey-patch torch.Module.to to fall back to to_empty() on meta-tensor errors
from textwrap import dedent
import json
import torch
from torch.nn.modules.module import Module as _TorchModule
_orig_to = _TorchModule.to
def _safe_to(self, *args, **kwargs):
    try:
        # 1) normal behavior
        return _orig_to(self, *args, **kwargs)
    except NotImplementedError as e:
        # 2a) fallback if it's the meta-tensor error
        if "Cannot copy out of meta tensor" in str(e):
            # Extract the device that was requested
            # args[0] is usually the device (e.g. "cpu" or "cuda")
            device = args[0] if len(args) >= 1 else kwargs.get("device")
            return self.to_empty(device=device)
        raise
    except TypeError:
        # 2b) fallback if to_empty() signature mismatch
        # Extract the device that was requested
        # args[0] is usually the device (e.g. "cpu" or "cuda")
        device = args[0] if len(args) >= 1 else kwargs.get("device")
        return self.to_empty(device=device)
_TorchModule.to = _safe_to  # now all Module.to() calls will handle meta tensors :contentReference[oaicite:0]{index=0}

from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.openai import OpenAIChat
from agno.tools.googlesearch import GoogleSearchTools
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.knowledge.combined import CombinedKnowledgeBase
from agno.vectordb.pgvector import PgVector
# __import__('pysqlite3')
import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from agno.vectordb.chroma import ChromaDb

from agno.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.document.chunking.semantic import SemanticChunking

from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HF_TOKEN")


def create_uud_knowledge_base(pdf_path="documents"):
    # make new collection for law
    law_vector_db = PgVector(
        table_name=f"law_kb",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        # embedder=SentenceTransformerEmbedder(),
        # ^ NOTE: if this is turned off, by default use OpenAIEmbedder.
        # Can only work if OPENAI_API_KEY is valid
    )
    law_kb = PDFKnowledgeBase(
        path=pdf_path,
        vector_db=law_vector_db,
    )

    with open("data/peraturan_go_id_output.jsonl", "r") as f:
        peraturan_go_id_urls = [json.loads(line)["url"] for line in f]
        # deduplicate
        peraturan_go_id_urls = list(set(peraturan_go_id_urls))

    if not peraturan_go_id_urls:
        # if there's no peraturan go id urls, return law kb only
        return law_kb
    
    # make new collection for peraturan go id
    peraturan_vector_db = PgVector(
        table_name="peraturan_go_id_kb",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        # embedder=SentenceTransformerEmbedder(),
    )

    peraturan_go_id_kb = PDFUrlKnowledgeBase(
        urls=peraturan_go_id_urls,
        vector_db=peraturan_vector_db,
    )

    # make a combined kb to use
    combined_kb = CombinedKnowledgeBase(
        sources=[
            law_kb,
            peraturan_go_id_kb,
        ],
        vector_db=peraturan_vector_db
    )

    return combined_kb
    # return peraturan_go_id_kb

def create_agent(system_prompt_path="data/system_prompt.txt", debug_mode=True):
    combined_kb = create_uud_knowledge_base(pdf_path="documents")
    combined_kb.load(recreate=False)
    # Instead of combined_kb.load(), do a manual insert with per-doc error handling
    # try:
    #     # Get only the docs that aren’t already in the collection
    #     docs_to_load = combined_kb.filter_existing_documents()
        
    #     safe_docs = []
    #     safe_filters = []
    #     for doc in docs_to_load:
    #         content = doc.content.strip()
    #         if not content:
    #             # skip empty chunks
    #             continue

    #         try:
    #             # attempt embedding
    #             doc.embed(embedder=combined_kb.embedder)
    #             safe_docs.append(doc)
    #             safe_filters.append(doc.meta_data or {})
    #         except Exception as e:
    #             # skip any chunk that fails to embed
    #             continue
        
    #     if safe_docs:
    #         combined_kb.vector_db.insert(
    #             documents=safe_docs,
    #             filters=safe_filters
    #         )
    # except Exception:
    #     # collection already exists — ignore
    #     pass

    # get system prompt
    with open(system_prompt_path, 'r') as system_prompt_f:
        system_prompt = system_prompt_f.read()
    
    # Define which provider to use: 'groq' or 'openai'
    MODEL_PROVIDER = "groq"

    # Select model based on provider
    if MODEL_PROVIDER == "groq":
        model = Groq(
            id="llama-3.3-70b-versatile",
            temperature=0.2 
        )
    elif MODEL_PROVIDER == "openai":
        model = OpenAIChat(
            id="gpt-4o",
            response_format="json",
            temperature=0.2,
            top_p=0.2
        )
    else:
        raise ValueError(f"Unsupported model provider: {MODEL_PROVIDER}")

    agent = Agent(
        name="law-agent",
        agent_id="law-agent",
        model=model,
        description=(
            "Anda adalah seorang ahli hukum Indonesia. "
            "Tugas Anda adalah menganalisis kasus hukum, mengidentifikasi pelanggaran, menjelaskan penanganannya, "
            "serta menyebutkan sanksi yang mungkin dikenakan sesuai dengan hukum dan peraturan yang berlaku di Indonesia."
        ),
        instructions=[dedent(system_prompt)],
        knowledge=combined_kb,
        search_knowledge=True,
        tools=[
            GoogleSearchTools()
        ],
        show_tool_calls=True,
        debug_mode=debug_mode,
        markdown=True,
    )

    return agent
