import os
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# 生成混淆矩阵函数
def cmpic(y_true, y_pred,name):
    def plot_confusion_matrix(cm, title='Confusion Matrix On SCENE', cmap = plt.cm.binary):
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title,fontsize=16)
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
       # plt.xticks(xlocations, ('MITcoast',  'MITforest', 'MIThighway', 'MITinsidecity', 'MITmountain',
        #                        'MITopencountry', 'MITstreet','MITtallbuilding'),rotation=45,size=16) # jaffe分类
        #plt.yticks(xlocations, ('MITcoast',  'MITforest', 'MIThighway', 'MITinsidecity', 'MITmountain',
        #                        'MITopencountry', 'MITstreet','MITtallbuilding'),size=16)

        plt.xticks(xlocations, (
        'bedroom', 'CALsuburb', 'industrial', 'kitchen','livingroom','MITcoast', 'MITforest', 'MIThighway', 'MITinsidecity',
        'MITmountain', 'MITopencountry', 'MITstreet', 'MITtallbuilding', 'PARoffice', 'store'), rotation=90,size=16)  # jaffe分类
        plt.yticks(xlocations, (
        'bedroom', 'CALsuburb', 'industrial', 'kitchen','livingroom','MITcoast', 'MITforest', 'MIThighway', 'MITinsidecity',
        'MITmountain', 'MITopencountry', 'MITstreet', 'MITtallbuilding', 'PARoffice', 'store'), size=16)

    cm = confusion_matrix(y_true, y_pred)
    # print(cm)
    labels = np.arange(len(cm))
    # print(labels)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    # print(cm_normalized)
    plt.figure(figsize=(24,16), dpi=120)
    #set the fontsize of label.
    #for label in plt.gca().xaxis.get_ticklabels():
    #    label.set_fontsize(8)
    #text portion
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        # c = cm_normalized[y_val][x_val]
        c = cm_normalized[y_val][x_val]
        if (c >= 0):
            plt.text(x_val, y_val, "%0.4f" %(c,), color="white" if c > 0.6 else "black", fontsize=16, va='center', ha='center')
            # plt.text(x_val, y_val, "%0.2f" %(c,), color='red', fontsize=7, va='center', ha='center')
    #offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plot_confusion_matrix(cm_normalized, title='Confusion Matrix On SCENE')
    #plot_confusion_matrix(cm_normalized, title='Confusion Matrix On CK+')
    #plot_confusion_matrix(cm_normalized, title='Confusion Matrix On Oulu-CASIA')
    out_dir=r"eff_image_cm/test"
    # out_dir = r"image/ck+"
    # plt.savefig('/HAR_cm.png', format='png')
    # show confusion matrix

    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix_' + str(name)+ '.png'))
    plt.show()
    # plt.savefig(os.path.join(out_dir, 'Confusion_Matrix_' + set_name + '.png'))