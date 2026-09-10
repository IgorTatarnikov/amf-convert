import xml.etree.ElementTree as ET


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
        # Stage position expressed in pixels (calibration handles
        # Z anisotropy).
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
