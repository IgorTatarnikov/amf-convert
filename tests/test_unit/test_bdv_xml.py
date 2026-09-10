import xml.etree.ElementTree as ET

from amf_convert.bdv_xml import write_bdv_xml


def test_write_bdv_xml_structure(tmp_path):
    xml_path = tmp_path / "out.xml"
    write_bdv_xml(
        xml_path=str(xml_path),
        h5_filename="out.h5",
        num_tiles=2,
        num_channels=3,
        x_size=2048,
        y_size=2048,
        z_size=356,
        voxel_size=(0.426, 0.426, 2.0),
        stage_positions=[(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)],
    )
    root = ET.parse(str(xml_path)).getroot()
    assert root.tag == "SpimData"
    # one ViewSetup per (tile, channel)
    assert len(root.findall(".//ViewSetup")) == 6
    # one ViewRegistration per (tile, channel), each with 2 transforms
    regs = root.findall(".//ViewRegistration")
    assert len(regs) == 6
    assert all(len(r.findall("ViewTransform")) == 2 for r in regs)
    loader = root.find(".//ImageLoader")
    assert loader.get("format") == "bdv.hdf5"
