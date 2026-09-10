import h5py
import numpy as np

from amf_convert.pyramid import downsample, step_factors


class BdvH5Writer:
    def __init__(
        self,
        h5_path,
        num_tiles,
        num_channels,
        shape_zyx,
        cum_factors,
        chunk=64,
        downsample="decimate",
    ):
        self.f = h5py.File(h5_path, "w")
        self.num_channels = num_channels
        self.shape_zyx = tuple(int(v) for v in shape_zyx)
        self.cum = np.asarray(cum_factors, dtype="i4")
        self.steps = step_factors(self.cum)
        self.chunk = chunk
        self.method = downsample
        n = len(self.cum)
        # BDV metadata is (x,y,z); cell arrays are (z,y,x) — resolutions is
        # reversed, subdivisions is cubic so reversal-invariant. If chunk ever
        # goes per-axis, subdivisions needs its own [::-1].
        res_xyz = self.cum[:, ::-1]  # (z,y,x) cumulative -> (x,y,z)
        subdiv = np.full((n, 3), chunk, dtype="i4")
        for sid in range(num_tiles * num_channels):
            self.f.create_dataset(f"s{sid:02}/resolutions", data=res_xyz)
            self.f.create_dataset(f"s{sid:02}/subdivisions", data=subdiv)

    def _chunks(self, shape):
        return tuple(min(self.chunk, int(s)) for s in shape)

    def write_tile(self, tile, slabs):
        c = self.num_channels
        level0 = {}
        for z0, slab in slabs:  # slab: (z, C, Y, X)
            z = slab.shape[0]
            for j in range(c):
                sid = tile * c + j
                if j not in level0:
                    level0[j] = self.f.create_dataset(
                        f"t00000/s{sid:02}/0/cells",
                        shape=self.shape_zyx,
                        chunks=self._chunks(self.shape_zyx),
                        dtype="uint16",
                    )
                level0[j][z0 : z0 + z] = slab[:, j]
        del slab  # free ~2.7 GB held by the last slab during the pyramid pass
        self._build_pyramid(tile)

    def _build_pyramid(self, tile):
        c = self.num_channels
        for j in range(c):
            sid = tile * c + j
            # ponytail: hold one channel's full-res volume in RAM (~3 GB
            # for our scopes) and derive every coarser level from it.
            # Switch to slab-wise read-back if a machine can't hold one.
            prev = np.asarray(self.f[f"t00000/s{sid:02}/0/cells"])
            for k in range(1, len(self.cum)):
                prev = downsample(prev, tuple(self.steps[k]), self.method)
                print(
                    f"  tile {tile} pyramid: channel {j} level {k} "
                    f"-> {prev.shape}"
                )
                d = self.f.create_dataset(
                    f"t00000/s{sid:02}/{k}/cells",
                    shape=prev.shape,
                    chunks=self._chunks(prev.shape),
                    dtype="uint16",
                )
                d[:] = prev

    def close(self):
        self.f.close()
