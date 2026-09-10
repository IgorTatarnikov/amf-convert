import xml.etree.ElementTree as ET
from pathlib import Path

import nd2
import h5py
import numpy as np


def _extract_stage_positions(nd2_file, num_tiles):
    """Return per-tile stage positions in microns, ordered to match the
    position (P) axis of the ND2 data. Falls back to zeros if the file has no
    XY position loop (e.g. a single, already-stitched image)."""
    points = None
    for loop in nd2_file.experiment:
        if getattr(loop, "type", None) == "XYPosLoop":
            points = loop.parameters.points
            break

    if points and len(points) >= num_tiles:
        return [
            (p.stagePositionUm.x, p.stagePositionUm.y, p.stagePositionUm.z)
            for p in points[:num_tiles]
        ]
    return [(0.0, 0.0, 0.0)] * num_tiles


def write_bdv_xml(
    xml_path,
    h5_filename,
    num_tiles,
    num_channels,
    x_size,
    y_size,
    z_size,
    voxel_size,
    stage_positions,
):
    """Write a BigDataViewer/BigStitcher SpimData XML alongside the HDF5 file.

    The ViewSetup ids match the ``s{id:02}`` datasets written to the HDF5
    (``id = tile * num_channels + channel``). Each view gets two transforms,
    matching the BigStitcher convention:

    * a translation placing the tile at its stage position (in pixel units,
      i.e. ``stage_um / voxel_size``), and
    * a ``calibration`` transform carrying the Z anisotropy
      (``voxel_z / voxel_x``), applied last.

    This gives BigDataViewer a sensible initial layout of the tiles that
    BigStitcher can then refine.
    """
    vx, vy, vz = (v or 1.0 for v in voxel_size)
    z_stretch = vz / vx

    # Anchor the grid near the origin so tile coordinates stay small. The stage
    # X axis runs opposite to the image X axis (determined empirically by
    # maximising overlap correlation between neighbouring tiles), so X is
    # anchored at its maximum and Y/Z at their minimum.
    x1 = max(p[0] for p in stage_positions)
    y0 = min(p[1] for p in stage_positions)
    z0 = min(p[2] for p in stage_positions)

    spim = ET.Element("SpimData", version="0.2")
    ET.SubElement(spim, "BasePath", type="relative").text = "."

    seq = ET.SubElement(spim, "SequenceDescription")
    loader = ET.SubElement(seq, "ImageLoader", format="bdv.hdf5")
    ET.SubElement(loader, "hdf5", type="relative").text = h5_filename

    setups = ET.SubElement(seq, "ViewSetups")
    for i in range(num_tiles):
        for j in range(num_channels):
            setup_id = i * num_channels + j
            vs = ET.SubElement(setups, "ViewSetup")
            ET.SubElement(vs, "id").text = str(setup_id)
            ET.SubElement(vs, "name").text = f"setup {setup_id}"
            ET.SubElement(vs, "size").text = f"{x_size} {y_size} {z_size}"
            voxel = ET.SubElement(vs, "voxelSize")
            ET.SubElement(voxel, "unit").text = "um"
            ET.SubElement(voxel, "size").text = f"{vx} {vy} {vz}"
            attrs = ET.SubElement(vs, "attributes")
            ET.SubElement(attrs, "illumination").text = "0"
            ET.SubElement(attrs, "channel").text = str(j)
            ET.SubElement(attrs, "tile").text = str(i)
            ET.SubElement(attrs, "angle").text = "0"

    illum = ET.SubElement(setups, "Attributes", name="illumination")
    el = ET.SubElement(illum, "Illumination")
    ET.SubElement(el, "id").text = "0"
    ET.SubElement(el, "name").text = "0"

    chan = ET.SubElement(setups, "Attributes", name="channel")
    for j in range(num_channels):
        el = ET.SubElement(chan, "Channel")
        ET.SubElement(el, "id").text = str(j)
        ET.SubElement(el, "name").text = str(j)

    tile = ET.SubElement(setups, "Attributes", name="tile")
    for i in range(num_tiles):
        el = ET.SubElement(tile, "Tile")
        ET.SubElement(el, "id").text = str(i)
        ET.SubElement(el, "name").text = str(i)

    angle = ET.SubElement(setups, "Attributes", name="angle")
    el = ET.SubElement(angle, "Angle")
    ET.SubElement(el, "id").text = "0"
    ET.SubElement(el, "name").text = "0"

    tps = ET.SubElement(seq, "Timepoints", type="range")
    ET.SubElement(tps, "first").text = "0"
    ET.SubElement(tps, "last").text = "0"
    ET.SubElement(seq, "MissingViews")

    regs = ET.SubElement(spim, "ViewRegistrations")
    for i in range(num_tiles):
        sx, sy, sz = stage_positions[i]
        # Stage position expressed in pixels (calibration handles Z anisotropy).
        # Image X runs opposite to stage X, so it is measured from the max.
        tx = (x1 - sx) / vx
        ty = (sy - y0) / vy
        tz = (sz - z0) / vz
        for j in range(num_channels):
            setup_id = i * num_channels + j
            reg = ET.SubElement(
                regs, "ViewRegistration", timepoint="0", setup=str(setup_id)
            )
            t = ET.SubElement(reg, "ViewTransform", type="affine")
            ET.SubElement(t, "Name").text = "Translation from Stage Position"
            ET.SubElement(t, "affine").text = (
                f"1.0 0.0 0.0 {tx} 0.0 1.0 0.0 {ty} 0.0 0.0 1.0 {tz}"
            )
            c = ET.SubElement(reg, "ViewTransform", type="affine")
            ET.SubElement(c, "Name").text = "calibration"
            ET.SubElement(c, "affine").text = (
                f"1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 {z_stretch} 0.0"
            )

    tree = ET.ElementTree(spim)
    ET.indent(tree, space="  ")
    tree.write(str(xml_path), encoding="UTF-8", xml_declaration=True)
    print(f"Wrote BigDataViewer XML to {xml_path}")


def convert_nd2_to_h5(input_file, output_file):
    """
    Convert an ND2 file to HDF5 format.

    Parameters:
    input_file (str): Path to the input ND2 file.
    output_file (str): Path to the output HDF5 file.
    """
    # Read the ND2 file
    nd2_file = nd2.ND2File(input_file)

    metadata = nd2_file.metadata
    full_data = nd2_file.to_dask()
    print("Metadata:", metadata)
    print("Data shape:", full_data.shape)

    z_size, y_size, x_size = full_data.shape[-4], full_data.shape[-2], full_data.shape[-1]
    num_channels = full_data.shape[-3]
    num_tiles = full_data.shape[-5] if full_data.ndim == 5 else 1

    print(f"Z size: {z_size}, Y size: {y_size}, X size: {x_size}, Channels: {num_channels}, Tiles: {num_tiles}")

    # Stage positions / voxel size for the BigDataViewer XML (read before close).
    voxel_size = tuple(nd2_file.voxel_size())  # (x, y, z) in microns
    stage_positions = _extract_stage_positions(nd2_file, num_tiles)

    h5_path = output_file or input_file.replace('.nd2', '.h5')
    output_file = h5py.File(h5_path, 'w')
    num_resolutions = 4

    resolution = np.ones((num_resolutions, 3), dtype="i4")
    subdivision = np.ones((num_resolutions, 3), dtype="i4") * 64

    for i in range(1, num_resolutions):
        resolution[i, 0] = resolution[i - 1, 0] * 2
        resolution[i, 1] = resolution[i - 1, 1] * 2

    for i in range(num_channels * num_tiles):
        res_ds = output_file.require_dataset(
            f"s{i:02}/resolutions",
            dtype="i4",
            shape=resolution.shape,
        )
        sub_ds = output_file.require_dataset(
            f"s{i:02}/subdivisions",
            dtype="i4",
            shape=subdivision.shape,
        )
        res_ds[:] = resolution
        sub_ds[:] = subdivision

    for i in range(num_tiles):
        tile_data = full_data[i, :, :, :, :] if num_tiles > 1 else full_data[:, :, :, :]
        for j in range(num_channels):
            print(f"Writing tile {i} channel {j} to t00000/s{(i*num_channels)+j:02}/0/cells")
            data = np.array(tile_data[:, j, :, :].squeeze())
            ds = output_file.create_dataset(
                f"t00000/s{(i*num_channels)+j:02}/0/cells",
                shape=(z_size, y_size, x_size),
                chunks=tuple(subdivision[0]),
                dtype="uint16",
            )
            ds[:] = data

            for k in range(1, num_resolutions):
                factors = tuple(int(res) for res in resolution[k][::-1])
                slices = tuple(
                    slice(None, None, factor) for factor in factors
                )
                downscaled_data = data[slices]
                print(f"Writing downscaled tile {i} channel {j} resolution {k} to t00000/s{(i*num_channels)+j:02}/{k}/cells")
                ds = output_file.create_dataset(
                    f"t00000/s{(i*num_channels)+j:02}/{k}/cells",
                    shape=downscaled_data.shape,
                    chunks=tuple(subdivision[k]),
                    dtype="uint16",
                )
                ds[:] = downscaled_data

    output_file.close()
    nd2_file.close()

    xml_path = str(Path(h5_path).with_suffix('.xml'))
    write_bdv_xml(
        xml_path,
        Path(h5_path).name,
        num_tiles,
        num_channels,
        x_size,
        y_size,
        z_size,
        voxel_size,
        stage_positions,
    )


if __name__ == "__main__":
    file_list = [
                 "/mnt/Data/in-vivo-reg/Sara_E_2023/052/confocal/Region052_ChannelBackground,EGFP,tdTomato_Seq0001.nd2"
                 ]

    for file_path in file_list:
        input_path = Path(file_path)
        output_path = input_path.with_suffix('.h5')
        convert_nd2_to_h5(str(input_path), str(output_path))
