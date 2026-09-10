from pathlib import Path

from amf_convert.bdv_h5 import BdvH5Writer
from amf_convert.bdv_xml import write_bdv_xml
from amf_convert.nd2_source import iter_slabs, open_nd2
from amf_convert.pyramid import level_factors

_H5_SUFFIXES = (".h5", ".hdf5")


def convert(
    input_path,
    output_path=None,
    *,
    n_levels=4,
    chunk=64,
    downsample="decimate",
):
    input_path = str(input_path)
    if output_path is None:
        output_path = str(Path(input_path).with_suffix(".h5"))
    output_path = str(output_path)
    suffix = Path(output_path).suffix.lower()
    if suffix == ".zarr":
        raise NotImplementedError("ome-zarr output not implemented")
    if suffix not in _H5_SUFFIXES:
        raise ValueError(f"unsupported output format: {output_path}")

    src = open_nd2(input_path)
    print(
        f"tiles={src.num_tiles} channels={src.num_channels} "
        f"shape_zyx={src.shape_zyx} voxel_um={src.voxel_size}"
    )
    cum = level_factors(src.voxel_size, n_levels)
    writer = BdvH5Writer(
        output_path,
        src.num_tiles,
        src.num_channels,
        src.shape_zyx,
        cum,
        chunk=chunk,
        downsample=downsample,
    )
    for tile in range(src.num_tiles):
        print(f"tile {tile}: streaming {n_levels} levels")
        writer.write_tile(tile, iter_slabs(src, tile, chunk))
    writer.close()
    src.file.close()

    z, y, x = src.shape_zyx
    write_bdv_xml(
        str(Path(output_path).with_suffix(".xml")),
        Path(output_path).name,
        src.num_tiles,
        src.num_channels,
        x,
        y,
        z,
        src.voxel_size,
        src.stage_positions,
    )
    return output_path
