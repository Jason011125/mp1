import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal

@nki.jit
def conv2d(X, W, bias):
    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0
    assert out_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    # gemm_moving_fmax is max number of pixels that can be processed in one cycle
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array/ allocate space to be stored in HBM
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_height, out_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions
    c_in_pmax = nl.tile_size.pmax  # 128; max number of channels
    c_out_pmax = c_in_pmax  # 128
    n_tiles_c_in = in_channels // c_in_pmax  # num of tiles for input channels
    n_tiles_c_out = out_channels // c_out_pmax  # num of tiles for output channels

    # Reshape weights and break channels into multiple tiles
    W = W.reshape((n_tiles_c_out, c_out_pmax, n_tiles_c_in,
                  c_in_pmax, filter_height, filter_width))

    # allocate sbuf space
    w_sbuf = nl.ndarray((n_tiles_c_out, nl.par_dim(c_out_pmax), n_tiles_c_in,
                        c_in_pmax, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)

    # load w to sbuf
    for oc_tile in nl.affine_range(n_tiles_c_out):
        w_sbuf[oc_tile] = nl.load(W[oc_tile])

    for b in nl.affine_range(batch_size):
        for start_row in nl.affine_range(input_height):
            x_sbuf = nl.ndarray((n_tiles_c_in, nl.par_dim(c_in_pmax), filter_height, input_width),
                                dtype=X.dtype, buffer=nl.sbuf)
            for ic_tile in nl.affine_range(n_tiles_c_in):
                for fh in nl.affine_range(filter_height):
                    global_row = start_row + fh
                    if global_row < input_height:
                        x_sbuf[ic_tile, :, fh, :] = nl.load(
                            X[b, ic_tile * c_in_pmax: (ic_tile + 1) * c_in_pmax, global_row, :]
                        )

            #Convolution partial sums for the current row
            for oc_tile in nl.affine_range(n_tiles_c_out):

                tile_psum = nl.zeros(
                    (nl.par_dim(c_out_pmax), 1, out_width),
                    dtype=nl.float32, buffer=nl.psum
                )
                # add partial sum through tiles
                for ic_tile in nl.affine_range(n_tiles_c_in):

                    for fh in nl.affine_range(filter_height):
                        for fw in nl.affine_range(filter_width):

                            w_tile = w_sbuf[oc_tile, :, ic_tile, :, fh, fw]

                            window = x_sbuf[ic_tile, :, fh: fh + 1, fw: fw + out_width]

                            tile_psum += nl.matmul(w_tile, window)

                # Add bias and store the result in HBM
                bias_sbuf = nl.ndarray(
                    (nl.par_dim(c_out_pmax),), dtype=bias.dtype, buffer=nl.sbuf)

                bias_slice = bias[oc_tile * c_out_pmax: (oc_tile + 1) * c_out_pmax]
                bias_sbuf = nl.load(bias_slice)

                tile_psum = nisa.tensor_scalar(tile_psum, nl.add, bias_sbuf)

                # Store back
                for i_col in nl.affine_range(out_width):
                    out_val = tile_psum[:, 0, i_col]
                    out_c_global = oc_tile * c_out_pmax
                    out_r_global = start_row

                    if out_r_global < out_height:
                        nl.store(
                            X_out[b, out_c_global: out_c_global + c_out_pmax, out_r_global, i_col],
                            value=out_val
                        )

    return X_out
