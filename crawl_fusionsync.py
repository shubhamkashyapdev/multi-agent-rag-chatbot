import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
from upstash_vector import Index

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI

load_dotenv()

# Initialize OpenAI and Upstash Vector clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
vector_index = Index(
    url=os.getenv("UPSTASH_VECTOR_REST_URL"),
    token=os.getenv("UPSTASH_VECTOR_REST_TOKEN")
)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

# ... existing code for chunk_text, get_title_and_summary, and get_embedding functions ...

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Upstash Vector."""
    try:
        # Create a unique ID for the vector
        vector_id = f"{chunk.url}:{chunk.chunk_number}"
        
        # Prepare the metadata
        metadata = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata
        }
        
        # Upsert the vector with its metadata
        vector_index.upsert(
            vectors=[
                {
                    "id": vector_id,
                    "vector": chunk.embedding,
                    "metadata": metadata
                }
            ]
        )
        
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return True
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    try:
        # Get title and summary
        title, summary = await get_title_and_summary(chunk)
        
        # Get embedding
        embedding = await get_embedding(chunk)
        
        # Create metadata
        metadata = {
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "source": "fusionsync.ai"
        }
        
        return ProcessedChunk(
            url=url,
            chunk_number=chunk_number,
            title=title,
            summary=summary,
            content=chunk,
            metadata=metadata,
            embedding=embedding
        )
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return None

def chunk_text(text: str, max_chunk_size: int = 1500) -> List[str]:
    """Split text into chunks of maximum size."""
    chunks = []
    current_chunk = ""
    
    for paragraph in text.split("\n\n"):
        if len(current_chunk) + len(paragraph) < max_chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

async def get_title_and_summary(text: str) -> tuple[str, str]:
    """Get title and summary using OpenAI."""
    prompt = f"""Given the following text, provide a concise title and summary. 
    Format: Title: <title>
    Summary: <summary>
    
    Text: {text[:1000]}..."""
    
    response = await openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    
    result = response.choices[0].message.content
    title_line = result.split("\n")[0]
    summary_line = result.split("\n")[1]
    
    title = title_line.replace("Title:", "").strip()
    summary = summary_line.replace("Summary:", "").strip()
    
    return title, summary

async def get_embedding(text: str) -> List[float]:
    """Get embedding using OpenAI."""
    response = await openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)
    print(f"Processing {len(chunks)} chunks for {url}")
    
    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks 
        if chunk is not None  # Filter out None results
    ]
    await asyncio.gather(*insert_tasks)

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=True,  # Added verbose for debugging
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                try:
                    print(f"Crawling: {url}")
                    result = await crawler.arun(
                        url=url,
                        config=crawl_config,
                        session_id="fusionsync1"
                    )
                    if result.success:
                        print(f"Successfully crawled: {url}")
                        await process_and_store_document(url, result.markdown_v2.raw_markdown)
                    else:
                        print(f"Failed: {url} - Error: {result.error_message}")
                except Exception as e:
                    print(f"Error processing {url}: {e}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def get_fusionsync_urls() -> List[str]:
    """Get URLs from FusionSync sitemap."""
    sitemap_url = "https://fusionsync.ai/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        # For testing, return a single URL if sitemap fails
        return ["https://fusionsync.ai"]

async def main():
    # Get URLs from FusionSync
    urls = get_fusionsync_urls()
    if not urls:
        print("No URLs found to crawl")
        return
    
    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main()) 