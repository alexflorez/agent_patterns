# To use with 
# %matplotlib qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import RadioButtons
from itertools import repeat
plt.style.use('seaborn')

feat_data = np.load("feature_data.npy", allow_pickle=True)
# feats_plant + feats_water + feats_energy + [name_class]
plant = np.array(feat_data[:, : 100], dtype=np.float32)
water = np.array(feat_data[:, 100: 200], dtype=np.float32)
energy = np.array(feat_data[:, 200: 300], dtype=np.float32)
mass = np.array(feat_data[:, 300: 400], dtype=np.float32)
cls_ = feat_data[:, -1].tolist()

# number of samples in this particular dataset
smps = 16
# choosing 2 classes
f_val = 50
s_val = 20
t_val = 30
fo_val = 70
first = f_val * smps
second = s_val * smps
third = t_val * smps
fourth = fo_val * smps

def plot_features(ax, data, cls_, indexes, ylabel, n=5):
    f_data = data[first: first + n]
    s_data = data[second: second + n]
    t_data = data[third: third + n]
    fo_data = data[fourth: fourth + n]
    f_cls = cls_[indexes[0]]
    s_cls = cls_[indexes[1]]
    t_cls = cls_[indexes[2]]
    fo_cls = cls_[indexes[3]]

    def plot_lines(plotdata, linecolor):
        lines = []
        for d in plotdata:
            f_line, = ax.plot(d, color=linecolor)
            lines.append(f_line)
        return lines

    first_plt = plot_lines(f_data, 'blue')
    second_plt = plot_lines(s_data, 'green')
    third_plt = plot_lines(t_data, 'red')
    fourth_plt = plot_lines(fo_data, 'orange')

    ax.set_title(ylabel)

    return first_plt, second_plt, third_plt, fourth_plt

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
plt.subplots_adjust(bottom=0.25)
ft = ax1.text(0.05, 0.90, cls_[first], transform=fig.transFigure, 
              fontsize=16, color='blue', weight='bold')
st = ax1.text(0.05, 0.86, cls_[second], transform=fig.transFigure, 
              fontsize=16, color='green', weight='bold')
tt = ax1.text(0.05, 0.82, cls_[third], transform=fig.transFigure, 
              fontsize=16, color='red', weight='bold')
fot = ax1.text(0.05, 0.78, cls_[fourth], transform=fig.transFigure, 
              fontsize=16, color='orange', weight='bold')
idxs = [first, second, third, fourth]

fp1, sp1, tp1, fop1 = plot_features(ax1, plant, cls_, idxs, "Plant height")
fp2, sp2, tp2, fop2 = plot_features(ax2, water, cls_, idxs, "Count water")
fp3, sp3, tp3, fop3 = plot_features(ax3, energy, cls_, idxs, "Sum energy")
fp4, sp4, tp4, fop4 = plot_features(ax4, mass, cls_,  idxs, "Mass plant")

num_cls = len(set(cls_))

ax_first = plt.axes([0.20, 0.16, 0.70, 0.03])
slider_first = Slider(ax_first, 'First', 0, num_cls - 1, 
                      valinit=f_val, valstep=1)

ax_second = plt.axes([0.20, 0.11, 0.70, 0.03])
slider_second = Slider(ax_second, 'Second', 0, num_cls - 1, 
                      valinit=s_val, valstep=1)

ax_third = plt.axes([0.20, 0.06, 0.70, 0.03])
slider_third = Slider(ax_third, 'Third', 0, num_cls - 1, 
                      valinit=t_val, valstep=1)

ax_fourth = plt.axes([0.20, 0.01, 0.70, 0.03])
slider_fourth = Slider(ax_fourth, 'Fourth', 0, num_cls - 1, 
                      valinit=fo_val, valstep=1)

def update_first(val):
    step = slider_first.val
    step = int(step)
    global first
    first = step * smps
    ft.set_text(cls_[first])

    for i, fp in enumerate(fp1):
        fp.set_ydata(plant[first + i])
    
    for i, fp in enumerate(fp2):
        fp.set_ydata(water[first + i])
    
    for i, fp in enumerate(fp3):
        fp.set_ydata(energy[first + i])

    for i, fp in enumerate(fp4):
        fp.set_ydata(mass[first + i])

slider_first.on_changed(update_first)

def update_second(val):
    step = slider_second.val
    step = int(step)
    global second
    second = step * smps
    st.set_text(cls_[second])

    for i, sp in enumerate(sp1):
        sp.set_ydata(plant[second + i])
    
    for i, sp in enumerate(sp2):
        sp.set_ydata(water[second + i])
    
    for i, sp in enumerate(sp3):
        sp.set_ydata(energy[second + i])

    for i, sp in enumerate(sp4):
        sp.set_ydata(mass[second + i])

slider_second.on_changed(update_second)

def update_third(val):
    step = slider_third.val
    step = int(step)
    global third
    third = step * smps
    tt.set_text(cls_[third])

    for i, tp in enumerate(tp1):
        tp.set_ydata(plant[third + i])
    
    for i, tp in enumerate(tp2):
        tp.set_ydata(water[third + i])
    
    for i, tp in enumerate(tp3):
        tp.set_ydata(energy[third + i])

    for i, tp in enumerate(tp4):
        tp.set_ydata(mass[third + i])

slider_third.on_changed(update_third)

def update_fourth(val):
    step = slider_fourth.val
    step = int(step)
    global fourth
    fourth = step * smps
    fot.set_text(cls_[fourth])

    for i, fop in enumerate(fop1):
        fop.set_ydata(plant[fourth + i])
    
    for i, fop in enumerate(fop2):
        fop.set_ydata(water[fourth + i])
    
    for i, fop in enumerate(fop3):
        fop.set_ydata(energy[fourth + i])

    for i, fop in enumerate(fop4):
        fop.set_ydata(mass[fourth + i])

slider_fourth.on_changed(update_fourth)

plt.show()
