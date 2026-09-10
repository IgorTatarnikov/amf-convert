# amf-convert

Convert microscopy stacks to BigDataViewer HDF5 (multi-resolution,
64³-chunked) with a companion BigStitcher/BigDataViewer XML.

## Install

    pip install -e ".[dev]"

## Use

    amf-convert input.nd2 [output.h5] [--levels 4] [--chunk 64] \
        [--downsample decimate|mean]

Python:

    from amf_convert.pipeline import convert
    convert("input.nd2", "output.h5", n_levels=4, chunk=64)

`.zarr` output raises `NotImplementedError` — ome-zarr is a planned
extension, only the dispatch seam exists today.

## Memory

Peak RAM ≈ 6 GB (`--downsample decimate`) / 9 GB (`--downsample mean`) for a
5-channel 2048² stack at `--chunk 64`. The pyramid pass holds a full
single-channel level-0 volume plus an all-channel slab; lower `--chunk` to
reduce it.

## Why plain h5py writes, no compression?

Measured on a 60 GB uncompressed ND2:

| operation | throughput |
|---|---|
| ND2 read, per-channel strided | 0.15–0.48 GB/s |
| ND2 read, contiguous Z-slab, all channels | 1.43 GB/s |
| h5py chunked uint16, `ds[z0:z1] = slab` | 3.06 GB/s |
| h5py `write_direct_chunk` per 64³ cube | 3.02 GB/s |

The pipeline is read-bound: the simplest h5py write already runs at ~2x
the best ND2 read rate. `write_direct_chunk`, chunk-cache tuning, and
blosc/zstd compression add nothing here (and blosc/zstd would not open
in Fiji's bundled HDF5). pyfive is a read-only HDF5 library and cannot
participate in a write comparison at all. So: read all channels per
slab, split in RAM, write uncompressed chunk-aligned slabs.
