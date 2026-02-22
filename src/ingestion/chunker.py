"""Text chunking strategies."""

import re
from typing import List, Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    """A text chunk with metadata."""
    content: str
    index: int
    metadata: dict


class Chunker:
    """Base chunker with configurable size and overlap."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict | None = None) -> List[Chunk]:
        """Split text into chunks."""
        raise NotImplementedError


class TextChunker(Chunker):
    """Simple text chunker by character count."""

    def chunk(self, text: str, metadata: dict | None = None) -> List[Chunk]:
        """Split text into overlapping chunks."""
        if not text:
            return []

        chunks = []
        start = 0
        index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            chunks.append(Chunk(
                content=chunk_text,
                index=index,
                metadata=metadata or {},
            ))

            start = end - self.overlap
            index += 1

        return chunks


class MarkdownChunker(Chunker):
    """Markdown-aware chunker that respects headers."""

    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```')

    def chunk(self, text: str, metadata: dict | None = None) -> List[Chunk]:
        """Split markdown by headings while respecting code blocks."""
        if not text:
            return []

        # Find all heading positions
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        current_heading = ""

        for line in lines:
            line_size = len(line) + 1  # +1 for newline

            # Check if this is a heading
            heading_match = self.HEADING_PATTERN.match(line)

            if heading_match and current_size > 0:
                # Start new chunk at heading (use current_heading for content before this heading)
                chunks.append(Chunk(
                    content='\n'.join(current_chunk),
                    index=chunk_index,
                    metadata={**(metadata or {}), "heading": current_heading},
                ))
                current_chunk = [line]
                current_size = line_size
                current_heading = heading_match.group(2)
                chunk_index += 1
            elif heading_match:
                # First heading - just track it
                current_heading = heading_match.group(2)
                current_chunk.append(line)
                current_size += line_size
            elif current_size + line_size > self.chunk_size and current_chunk:
                # Chunk is full, start new one
                chunks.append(Chunk(
                    content='\n'.join(current_chunk),
                    index=chunk_index,
                    metadata={**(metadata or {}), "heading": current_heading},
                ))
                # Overlap: keep last few sentences/lines
                overlap_lines = self._get_overlap(current_chunk)
                current_chunk = overlap_lines + [line]
                current_size = sum(len(l) + 1 for l in current_chunk)
                chunk_index += 1
            else:
                current_chunk.append(line)
                current_size += line_size

        # Add final chunk
        if current_chunk:
            chunks.append(Chunk(
                content='\n'.join(current_chunk),
                index=chunk_index,
                metadata={**(metadata or {}), "heading": current_heading},
            ))

        return chunks

    def _get_overlap(self, lines: List[str]) -> List[str]:
        """Get last few lines for overlap."""
        overlap_lines = []
        overlap_size = 0
        for line in reversed(lines):
            if overlap_size + len(line) + 1 > self.overlap:
                break
            overlap_lines.insert(0, line)
            overlap_size += len(line) + 1
        return overlap_lines


def read_documents(path: str | Path) -> List[tuple[str, str]]:
    """Read documents from a path (returns uri, content)."""
    path = Path(path)
    documents = []

    if path.is_file():
        content = path.read_text()
        documents.append((str(path), content))
    elif path.is_dir():
        for ext in ('*.md', '*.txt', '*.rst'):
            for file in path.rglob(ext):
                content = file.read_text(encoding='utf-8', errors='ignore')
                documents.append((str(file), content))

    return documents
