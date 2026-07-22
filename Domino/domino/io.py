"""PLINK1 ``.bed/.bim/.fam`` reader with streaming block iteration.

Genotypes are decoded on demand from the memory-mapped ``.bed`` file in
blocks of variants, so the full genotype matrix is never materialised. This
is what lets the pipeline handle millions of variants at n up to ~10k.
"""
import numpy as np
import pandas as pd

# PLINK 2-bit codes -> allele-a1 dosage. 0b00 hom-first, 0b01 missing,
# 0b10 het, 0b11 hom-second. Samples are packed 4-per-byte, first sample low bits.
_CODE = np.array([0, np.nan, 1, 2], dtype=np.float32)
_B = np.arange(256, dtype=np.uint8)
_UNPACK = _CODE[np.stack(((_B >> 0) & 3, (_B >> 2) & 3, (_B >> 4) & 3, (_B >> 6) & 3), axis=1)]


def read_bim(prefix):
    return pd.read_csv(f"{prefix}.bim", sep=r"\s+", header=None,
                       names=["chrom", "snp", "cm", "pos", "a0", "a1"],
                       dtype={"chrom": str, "snp": str, "cm": float,
                              "pos": int, "a0": str, "a1": str})


def read_fam(prefix):
    return pd.read_csv(f"{prefix}.fam", sep=r"\s+", header=None,
                       names=["fid", "iid", "father", "mother", "sex", "pheno"],
                       dtype={"fid": str, "iid": str, "father": str, "mother": str})


class PlinkReader:
    """Lazy reader over a PLINK1 fileset given by ``prefix`` (no extension)."""

    def __init__(self, prefix, dtype=np.float32):
        self.prefix = prefix
        self.fam = read_fam(prefix)
        self.bim = read_bim(prefix)
        self.n_samples = len(self.fam)
        self.n_variants = len(self.bim)
        self.dtype = dtype
        self._bpv = (self.n_samples + 3) // 4
        self._bed = np.memmap(f"{prefix}.bed", mode="r", dtype=np.uint8)
        if not (self._bed[0] == 0x6c and self._bed[1] == 0x1b):
            raise ValueError(f"{prefix}.bed: bad magic bytes (not a PLINK .bed)")
        if self._bed[2] != 0x01:
            raise ValueError(f"{prefix}.bed: only SNP-major .bed is supported")

    def read_block(self, start, count):
        """Decode ``count`` consecutive variants starting at index ``start``."""
        off = 3 + start * self._bpv
        raw = self._bed[off: off + count * self._bpv]
        calls = _UNPACK[raw].reshape(-1)
        cpv = self._bpv * 4
        calls = calls[: count * cpv].reshape(count, cpv)
        return np.ascontiguousarray(calls[:, : self.n_samples].T.astype(self.dtype))

    def read_block_samples(self, start, count, sample_index):
        """Decode a variant block only for selected samples.

        PLINK stores four sample calls per byte. Selecting the packed bytes
        before lookup avoids materializing all cohort samples when an analysis
        uses a much smaller phenotype subset.
        """
        sidx = np.asarray(sample_index, dtype=np.int64)
        if np.any(sidx < 0) or np.any(sidx >= self.n_samples):
            raise IndexError("sample_index is outside the PLINK sample range")
        off = 3 + start * self._bpv
        raw = self._bed[off: off + count * self._bpv].reshape(count, self._bpv)
        packed = raw[:, sidx // 4]
        codes = (packed >> (2 * (sidx % 4))[None, :]) & 3
        calls = _CODE[codes]
        return np.ascontiguousarray(calls.T.astype(self.dtype))

    def iter_blocks(self, block_size=8192, variant_index=None, sample_index=None):
        """Yield ``(bim_slice, geno_block)`` over the requested variants.

        ``geno_block`` has shape (n_samples, block_len). If ``variant_index``
        is given it must be sorted; a contiguous span is read then subset,
        which is efficient for per-chromosome scans.
        """
        if variant_index is None:
            for s in range(0, self.n_variants, block_size):
                c = min(block_size, self.n_variants - s)
                block = self.read_block(s, c) if sample_index is None else self.read_block_samples(s, c, sample_index)
                yield self.bim.iloc[s:s + c], block
        else:
            variant_index = np.asarray(variant_index)
            for s in range(0, len(variant_index), block_size):
                idx = variant_index[s:s + block_size]
                lo, hi = int(idx.min()), int(idx.max())
                if sample_index is None:
                    block = self.read_block(lo, hi - lo + 1)[:, idx - lo]
                else:
                    block = self.read_block_samples(lo, hi - lo + 1, sample_index)[:, idx - lo]
                yield self.bim.iloc[idx], np.ascontiguousarray(block)

    def chrom_variant_index(self, chrom):
        return np.where(self.bim["chrom"].values == str(chrom))[0]

    @property
    def chroms(self):
        return list(dict.fromkeys(self.bim["chrom"].values))

    def close(self):
        """Release the memory-mapped BED file, especially on Windows."""
        bed = getattr(self, "_bed", None)
        mmap = getattr(bed, "_mmap", None)
        if mmap is not None:
            mmap.close()
        self._bed = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def __del__(self):
        self.close()
