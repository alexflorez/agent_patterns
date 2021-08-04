import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk
from skimage.util import img_as_ubyte

MAX_HEIGHT = 200
MIN_HEIGHT = 5


def data_flat(rows, cols):
    return np.full((rows, cols), MIN_HEIGHT, dtype=np.uint8)


def data_slope(rows, cols):
    """
    slope
    #
    #  #
    #  #  #
    #  #  #  #
    #  #  #  #  #
    """
    values = np.linspace(cols, 0, cols) + MIN_HEIGHT
    slope = np.full((rows, cols), values, dtype=np.uint8)
    return slope


def data_vhalf(rows, cols):
    """
    vhalf
    #
    #  #
    #  #  #
    #  #  #  #
    #  #  #  #  #  #  #  #
    """
    left = data_slope(rows, cols//2)
    right = np.ones((rows, cols//2), dtype=np.uint8) * MIN_HEIGHT

    half_v = np.hstack([left, right])
    return half_v


def data_v(rows, cols):
    """
    v
    #                 #
    # #             # #
    # # #         # # #
    # # # #     # # # #
    # # # # # # # # # #
    """
    left = data_slope(rows, cols//2)
    right = np.flip(left, axis=1)
    full_v = np.hstack([left, right])
    return full_v


def data_vinv(rows, cols):
    """
    vinv
            # #
          # # # #
        # # # # # #
      # # # # # # # #
    # # # # # # # # # #
    """
    v = data_v(rows, cols)
    vinv = np.max(v) + MIN_HEIGHT - v
    return vinv


def data_pyramid(rows, cols):
    """
    pyramid: rows and cols must be equal
    1 1 1 1 1 1 1 1 1 1
    1 2 2 2 2 2 2 2 2 1
    1 2 3 3 3 3 3 3 2 1
    1 2 3 4 4 4 4 3 2 1
    1 2 3 4 5 5 4 3 2 1
    1 2 3 4 5 5 4 3 2 1
    1 2 3 4 4 4 4 3 2 1
    1 2 3 3 3 3 3 3 2 1
    1 2 2 2 2 2 2 2 2 1
    1 1 1 1 1 1 1 1 1 1
    """
    full_vinv = data_vinv(rows, cols)
    # upper triangle
    triu = np.triu(full_vinv)
    # bottom triangle
    trib = np.flip(triu, axis=0)
    half_tie = np.triu(trib)
    h_tie = half_tie + np.fliplr(half_tie)
    v_tie = h_tie.T.copy()
    np.fill_diagonal(v_tie, 0)
    np.fill_diagonal(np.fliplr(v_tie), 0)
    pyramid = h_tie + v_tie
    return pyramid


def data_inv_pyramid(rows, cols):
    """
    inv_pyramid
    5 5 5 5 5 5 5 5 5 5
    5 4 4 4 4 4 4 4 4 5
    5 4 3 3 3 3 3 3 4 5
    5 4 3 2 2 2 2 3 4 5
    5 4 3 2 1 1 2 3 4 5
    5 4 3 2 1 1 2 3 4 5
    5 4 3 2 2 2 2 3 4 5
    5 4 3 3 3 3 3 3 4 5
    5 4 4 4 4 4 4 4 4 5
    5 5 5 5 5 5 5 5 5 5
    """
    pyramid = data_pyramid(rows, cols)
    inv_pyramid = pyramid.max() + MIN_HEIGHT - pyramid
    return inv_pyramid


def scale(data):
    prev_min = data.min()
    prev_max = data.max()
    prev_range = (prev_max - prev_min)
    new_range = MAX_HEIGHT // MIN_HEIGHT - MIN_HEIGHT
    scaled = (((data - prev_min) * new_range) / prev_range) + MIN_HEIGHT
    return scaled


def data_gaussian(rows, cols):
    # in the range -2 to +2
    x, y = np.meshgrid(np.linspace(-5, 5, rows), np.linspace(-5, 5, cols))
    dst = np.sqrt(x * x + y * y)

    sigma = 1
    mu = 0

    gauss = np.exp(-((dst - mu)**2 / (2 * sigma**2)))
    scaled = scale(gauss).astype(np.uint8)
    # scaled = img_as_ubyte(gauss)
    return scaled


def data_inv_gaussian(rows, cols):
    # gauss is a uint8 array
    gauss = data_gaussian(rows, cols)
    return gauss.max() + MIN_HEIGHT - gauss


def data_disk(rows, cols, radius):
    center = (rows // 2, cols // 2)
    rr, cc = disk(center, radius)
    data = np.ones((rows, cols), dtype=np.uint8) * MIN_HEIGHT
    data[rr, cc] = MAX_HEIGHT // MIN_HEIGHT
    return data


def data_circle(rows, cols, radius, width):
    int_circle = data_disk(rows, cols, radius - width)
    ext_circle = data_disk(rows, cols, radius)

    circle = ext_circle - int_circle + MIN_HEIGHT
    return circle


def data_chess(rows, cols):
    table = np.zeros((rows, cols), dtype=np.uint8)
    table[1::2, ::2] = 1
    table[::2, 1::2] = 1
    return table + MIN_HEIGHT


if __name__ == '__main__':
    xrows = 100
    ycols = 100

    dt_slope = data_slope(xrows, ycols)
    dt_half_v = data_vhalf(xrows, ycols)
    dt_full_v = data_v(xrows, ycols)
    dt_full_vinv = data_vinv(xrows, ycols)
    dt_pyramid = data_pyramid(xrows, ycols)
    dt_inv_pyramid = data_inv_pyramid(xrows, ycols)
    dt_gaussian = data_gaussian(xrows, ycols)
    dt_inv_gauss = data_inv_gaussian(xrows, ycols)
    dt_flat = data_flat(xrows, ycols)
    radius_d = 5
    dt_disco = data_disk(xrows, ycols, radius_d)
    width_d = 2
    dt_circle = data_circle(xrows, ycols, radius_d, width_d)
    dt_chess = data_chess(xrows, ycols)

    imgs = [dt_slope, dt_half_v, dt_full_v, dt_full_vinv, dt_pyramid,  # chess
            dt_inv_pyramid, dt_gaussian, dt_inv_gauss, dt_disco, dt_circle]
    names = ['slope', 'half_v', 'full_v', 'full_vinv', 'pyramid',  # 'chess',
             'inv_pyramid', 'gaussian', 'inv_gauss', 'disco', 'circle']
    fig, axs = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axs.flat):
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.imshow(imgs[i], cmap="viridis")
        ax.set_xlabel(names[i])
    plt.show()
