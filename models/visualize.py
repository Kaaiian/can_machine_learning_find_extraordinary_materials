import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

plt.rcParams.update({'font.size': 20})

props_dict = {'ael_bulk_modulus_vrh': 'Bulk Modulus',
         'ael_debye_temperature': 'Debye Temperature',
         'ael_shear_modulus_vrh': 'Log Shear Modulus',
         'agl_thermal_conductivity_300K': 'Log Thermal Conductivity',
         'agl_thermal_expansion_300K': 'Log Thermal Expansion',
         'Egap': 'Band Gap'}

def make_figure(threshold,
                y_act,
                y_pred,
                formula,
                gap_size,
                test_threshold_x,
                prop='prop',
                gap=0,
                model_type='default',
                classification=None,
                holdout_elem=None,
                structure=None,
                holdout_struct=None,
                folder='figures'):
    color = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    y_act_labeled = [1 if x > test_threshold_x else 0 for x in y_act]
    y_pred_labeled = [1 if x >= threshold else 0 for x in y_pred]
    prfs = precision_recall_fscore_support(y_act_labeled, y_pred_labeled)
    precision, recall, fscore, support = prfs

    print('precision: {:0.2f}\nrecall: {:0.2f}'.format(precision[1],
                                                       recall[1]))

    tn, fp, fn, tp = confusion_matrix(y_act_labeled, y_pred_labeled).ravel() /\
        len(y_act_labeled) * 100

    plt.figure(1, figsize=(8, 8))
    left, width = 0.1, 0.65

    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.15]
    rect_histy = [left_h, bottom, 0.15, height]

    ax1 = plt.axes(rect_histx)
    ax1.hist(y_act, bins=31, color='silver', edgecolor='k')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = plt.axes(rect_scatter)
    if gap == 0:
        gap_size = 0

    labels = [None, None, None, None]
    if holdout_elem is None and holdout_struct is None:
        labels = ['false negative ({:0.0f}%)'.format(fn),
                  'false postive ({:0.0f}%)'.format(fp),
                  'true positive ({:0.0f}%)'.format(tp),
                  'true negative ({:0.0f}%)'.format(tn)]

    rect1 = patches.Rectangle((test_threshold_x, -100),
                              2600,
                              threshold + 100,
                              linewidth=1,
                              edgecolor='k',
                              facecolor=color[3],
                              alpha=0.25,
                              label=labels[0])
    rect2 = patches.Rectangle((-50, threshold),
                              test_threshold_x - gap_size + 50,
                              2600,
                              linewidth=1,
                              edgecolor='k',
                              facecolor=color[1],
                              alpha=0.25,
                              label=labels[1])
    rect3 = patches.Rectangle((test_threshold_x, threshold),
                              2600,
                              2600,
                              linewidth=1,
                              edgecolor='k',
                              facecolor=color[2],
                              alpha=0.25,
                              label=labels[2])
    rect4 = patches.Rectangle((-50, -50),
                              test_threshold_x - gap_size + 50,
                              threshold+50,
                              linewidth=1,
                              edgecolor='k',
                              facecolor='w',
                              alpha=0.25,
                              label=labels[3])


    ax2.add_patch(rect1)
    ax2.add_patch(rect2)
    ax2.add_patch(rect3)
    ax2.add_patch(rect4)

    if gap_size > 0:
        rect5 = patches.Rectangle((test_threshold_x - gap_size, -50),
                                  gap_size,
                                  1200,
                                  linewidth=1,
                                  edgecolor='k',
                                  facecolor='gray',
                                  hatch="//",
                                  alpha=0.25,
                                  label='artificial gap'.format(tn))
        ax2.add_patch(rect5)

    ax2.tick_params(direction='in',
                    length=5,
                    bottom=True,
                    top=True,
                    left=True,
                    right=True)

    ax2.plot(y_act, y_pred, 'o', mfc='#C0C0C0', alpha=0.5, label=None,
             mec='#2F4F4F', mew=1.3, markersize=10)


    ax2.plot([-100000, 60000], [threshold, threshold], 'k--',
             label='threshold', linewidth=3)
    ax2.plot([test_threshold_x, test_threshold_x],
             [-100000, 20000], 'k:', linewidth=3)

    ax2.set_ylabel('Predicted '+props_dict[prop])
    ax2.set_xlabel(props_dict[prop])
    x_range = max(y_act) - min(y_act)
    ax2.set_xlim(max(y_act) - x_range*1.05,
                 min(y_act) + x_range*1.05)
    if classification:
        y_pred_range = max(y_pred) - min(y_pred)
        ax2.set_ylim(max(y_pred) - y_pred_range*1.05,
                     min(y_pred) + y_pred_range*1.05)
    else:
        ax2.set_ylim(max(y_act) - x_range*1.05,
                     min(y_act) + x_range*1.05)

    ax1.set_xlim(ax2.get_xlim())
    ax1.axis('off')

    if holdout_elem is not None:
        comps = formula.str.split(r'([A-Z][a-z]*)')
        boolean = [True if holdout_elem in comp else False for comp in comps]
        y_holdout_act = y_act[boolean]
        y_holdout_pred = y_pred[boolean]
        ax2.plot(y_holdout_act, y_holdout_pred, 'o', mfc='gold', alpha=0.7,
                 mec='orange', mew=1.3, markersize=10,
                 label='element: '+str(holdout_elem))

    if holdout_struct is not None:
        boolean = [True if holdout_struct == comp else False for comp in structure]
        y_holdout_act = y_act[boolean]
        y_holdout_pred = y_pred[boolean]
        ax2.plot(y_holdout_act, y_holdout_pred, 'o', mfc='pink', alpha=0.7,
                 mec='red', mew=1.3, markersize=10,
                 label='space group: '+str(holdout_struct))

    if not classification:
        ax3 = plt.axes(rect_histy)
        ax3.hist(y_pred,
                 bins=31,
                 color='silver',
                 edgecolor='k',
                 orientation='horizontal')
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_ylim(ax2.get_ylim())
        ax3.axis('off')

    ax2.legend(loc=2, framealpha=0.15, handlelength=1.5)
    save_dir = folder + '/' + prop + '/' + str(gap) + '/'
    fig_name = save_dir + model_type + '.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.draw()
    plt.pause(0.001)
    plt.close()

if __name__ == '__main__':
    threshold = 5
    y_act = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    y_pred = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    test_threshold_x = 7
    make_figure(threshold,
                y_act,
                y_pred,
                test_threshold_x,
                mat_prop='prop',
                classification=False)
