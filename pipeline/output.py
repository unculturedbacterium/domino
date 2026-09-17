"""Incremental, atomic Parquet output."""
from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq


class AtomicParquetWriter:
    """Write DataFrame blocks to one Parquet file without retaining them.

    A uniquely named partial file is atomically promoted only after every row
    group has been written.  Interrupted jobs therefore never look complete.
    """

    def __init__(self, path, compression="zstd"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.partial_path = self.path.with_name(f".{self.path.name}.{uuid4().hex}.partial")
        self.compression = compression
        self._writer = None
        self._schema = None
        self.rows_written = 0
        self.row_groups_written = 0

    def write(self, frame):
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self._writer is None:
            self._schema = table.schema
            self._writer = pq.ParquetWriter(
                self.partial_path, self._schema, compression=self.compression
            )
        elif table.schema != self._schema:
            table = table.cast(self._schema)
        self._writer.write_table(table)
        self.rows_written += len(frame)
        self.row_groups_written += 1

    def close(self):
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None
        os.replace(self.partial_path, self.path)

    def abort(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        try:
            self.partial_path.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False
