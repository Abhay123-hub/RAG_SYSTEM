import hashlib
from pathlib import Path


class FileRegistry:
    """Hash files and track indexed document counts for the active session."""

    def __init__(self) -> None:
        self._files: dict[str, dict[str, int | str]] = {}

    @staticmethod
    def hash_file(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def register(self, path: str | Path, pages: int, chunks: int) -> tuple[str, bool]:
        file_hash = self.hash_file(path)
        duplicate = file_hash in self._files
        self._files[file_hash] = {"file_name": Path(path).name, "pages": pages, "chunks": chunks}
        return file_hash, duplicate

    def list_files(self) -> list[dict[str, int | str]]:
        return list(self._files.values())

    def remove(self, file_hash: str) -> None:
        self._files.pop(file_hash, None)