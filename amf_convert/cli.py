import argparse

from amf_convert.pipeline import convert


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="amf-convert",
        description="Convert an ND2 stack to BigDataViewer HDF5 + XML.",
    )
    p.add_argument("input", help="input .nd2 file")
    p.add_argument("output", nargs="?", help="output .h5 (default: input .h5)")
    p.add_argument("--levels", type=int, default=4,
                   help="number of resolution levels (default: 4)")
    p.add_argument("--chunk", type=int, default=64,
                   help="cube chunk / slab size (default: 64)")
    p.add_argument("--downsample", choices=["decimate", "mean"],
                   default="decimate",
                   help="pyramid downsampling method (default: decimate)")
    a = p.parse_args(argv)
    convert(a.input, a.output, n_levels=a.levels, chunk=a.chunk,
            downsample=a.downsample)


if __name__ == "__main__":
    main()
