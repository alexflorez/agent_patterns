import numpy as np
import matplotlib.pyplot as plt


def data_slope(rows, cols):
    """
    slope
    #
    # # 
    # # #
    # # # #
    # # # # #
    # # # # # #
    """        
    slope = np.full((rows, cols), np.arange(rows, 0, -1), dtype=np.uint8)
    return slope


def data_half_v(rows, cols):
    """
    half_v
    #
    # # 
    # # #
    # # # #
    # # # # #
    - - - - - - - - - -
    """
    left = np.full((rows, cols//2), np.arange(rows//2, 0, -1), dtype=np.uint8)
    right = np.zeros((rows, cols//2), dtype=np.uint8)
    
    half_v = np.hstack([left, right])
    return half_v


def data_full_v(rows, cols):
    """
    full_v
    #                 #
    # #             # #
    # # #         # # #
    # # # #     # # # #
    # # # # # # # # # #
    """
    half_v = data_half_v(rows, cols)
    full_v = half_v + np.flip(half_v, axis=1)
    return full_v


def data_full_vinv(rows, cols):
    """
    full_vinv
            # #
          # # # #
        # # # # # #
      # # # # # # # #
    # # # # # # # # # #
    """
    full_v = data_full_v(rows, cols)
    full_vinv = full_v.max() + 1 - full_v
    return full_vinv


def data_pyramid(rows, cols):
    """
    pyramid
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
    full_vinv = data_full_vinv(rows, cols)
    # upper triangle
    triu = np.triu(full_vinv)
    # bottom triangle
    trib = np.flip(triu, axis=0)
    half_tie = np.triu(trib)
    tie = half_tie + np.flip(half_tie, axis=1)
    step_pyramid = tie + tie.T
    
    vals = list(range(1, rows//2 + 1))
    to_correct = vals + vals[::-1]
    diag = np.diag(to_correct)
    inv_diag = np.flip(diag, axis=1)
    pyramid = step_pyramid - diag - inv_diag
    pyramid = pyramid.astype(np.uint8)
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
    inv_pyramid = pyramid.max() + 1 - pyramid
    return inv_pyramid


if __name__ == '__main__':
    rows = 10
    cols = 10

    slope = data_slope(rows, cols)
    half_v = data_half_v(rows, cols)
    full_v = data_full_v(rows, cols)
    pyramid = data_pyramid(rows, cols)
    inv_pyramid = data_inv_pyramid(rows, cols)
    full_vinv = data_full_vinv(rows, cols)
    
    np.save("images/slope.npy", slope)
    # np.save("half_v.npy", half_v)
    # np.save("full_v.npy", full_v)
    # np.save("full_vinv.npy", full_vinv)
    # np.save("pyramid.npy", pyramid)
    # np.save("inv_pyramid.npy", inv_pyramid)

    imgs = [slope, half_v, full_v, pyramid, inv_pyramid, full_vinv]
    names = ['slope', 'half_v', 'full_v', 'pyramid', 'inv_pyramid', 'full_vinv']
    fig, axs = plt.subplots(2, 3, figsize=(7, 5))
    for i, ax in enumerate(axs.flat):
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.imshow(imgs[i], cmap="viridis")
        ax.set_xlabel(names[i])

    plt.show()
