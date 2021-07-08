# %matplotlib qt
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import h5py
plt.style.use('seaborn')


def plot_features(ax, data, ylabel, n_smp=5):
    def plot_lines(plotdata, linecolor):
        lines = []
        for d in plotdata:
            f_line, = ax.plot(d, color=linecolor)
            lines.append(f_line)
        return lines

    f_data = data[first: first + n_smp]
    s_data = data[second: second + n_smp]
    t_data = data[third: third + n_smp]
    fo_data = data[fourth: fourth + n_smp]
    first_plt = plot_lines(f_data, 'blue')
    second_plt = plot_lines(s_data, 'green')
    third_plt = plot_lines(t_data, 'red')
    fourth_plt = plot_lines(fo_data, 'orange')

    ax.set_title(ylabel)
    return first_plt, second_plt, third_plt, fourth_plt


def change_plot(value, index, text, plt1, plt2, plt3, plt4):
    step = int(value)
    index = step * n_samples
    text.set_text(labels[index])

    for i, pt in enumerate(plt1):
        pt.set_ydata(plant[index + i])

    for i, pt in enumerate(plt2):
        pt.set_ydata(water[index + i])

    for i, pt in enumerate(plt3):
        pt.set_ydata(energy[index + i])

    for i, pt in enumerate(plt4):
        pt.set_ydata(mass[index + i])


def update_funct(val):
    global first
    step1 = slider_first.val
    global second
    step2 = slider_second.val
    global third
    step3 = slider_third.val
    global fourth
    step4 = slider_fourth.val
    change_plot(step1, first, first_text, fp1, fp2, fp3, fp4)
    change_plot(step2, second, second_text, sp1, sp2, sp3, sp4)
    change_plot(step3, third, third_text, tp1, tp2, tp3, tp4)
    change_plot(step4, fourth, fourth_text, fop1, fop2, fop3, fop4)


if __name__ == "__main__":
    datafile = "brodatz_features_all.hdf5"
    with h5py.File(datafile, 'r') as f:
        feats = f["features"][:]
        labels = f["labels"][:]

    # 1. plant_count
    # 2. water_count
    # 3. energy_count
    # 4. plant_sum
    # 5. water_sum
    # 6. energy_sum
    # 7. plant_max
    # 8. water_max
    # 9. energy_max
    n = 100
    plant = feats[:, : n]
    water = feats[:, n: 2*n]
    energy = feats[:, 2*n: 3*n]
    mass = feats[:, 3*n: 4*n]

    # number of samples in this particular dataset
    n_samples = 16
    # choosing 2 classes
    f_val = 50
    s_val = 20
    t_val = 30
    fo_val = 70
    first = f_val * n_samples
    second = s_val * n_samples
    third = t_val * n_samples
    fourth = fo_val * n_samples

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex="col")
    plt.subplots_adjust(bottom=0.25)
    first_text = ax1.text(0.05, 0.90, labels[first], transform=fig.transFigure,
                          fontsize=16, color='blue', weight='bold')
    second_text = ax1.text(0.05, 0.86, labels[second], transform=fig.transFigure,
                           fontsize=16, color='green', weight='bold')
    third_text = ax1.text(0.05, 0.82, labels[third], transform=fig.transFigure,
                  fontsize=16, color='red', weight='bold')
    fourth_text = ax1.text(0.05, 0.78, labels[fourth], transform=fig.transFigure,
                   fontsize=16, color='orange', weight='bold')

    fp1, sp1, tp1, fop1 = plot_features(ax1, plant, "Plant height")
    fp2, sp2, tp2, fop2 = plot_features(ax2, water, "Count water")
    fp3, sp3, tp3, fop3 = plot_features(ax3, energy, "Sum energy")
    fp4, sp4, tp4, fop4 = plot_features(ax4, mass, "Mass plant")

    num_cls = len(set(labels))

    ax_first = plt.axes([0.20, 0.16, 0.70, 0.03])
    slider_first = Slider(ax_first, 'First class', 0, num_cls - 1,
                          valinit=f_val, valstep=1)

    ax_second = plt.axes([0.20, 0.11, 0.70, 0.03])
    slider_second = Slider(ax_second, 'Second class', 0, num_cls - 1,
                           valinit=s_val, valstep=1)

    ax_third = plt.axes([0.20, 0.06, 0.70, 0.03])
    slider_third = Slider(ax_third, 'Third class', 0, num_cls - 1,
                          valinit=t_val, valstep=1)

    ax_fourth = plt.axes([0.20, 0.01, 0.70, 0.03])
    slider_fourth = Slider(ax_fourth, 'Fourth class', 0, num_cls - 1,
                           valinit=fo_val, valstep=1)

    slider_first.on_changed(update_funct)
    slider_second.on_changed(update_funct)
    slider_third.on_changed(update_funct)
    slider_fourth.on_changed(update_funct)
    plt.show()
