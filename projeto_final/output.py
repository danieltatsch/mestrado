import json
import numpy as np
from   pprint import pprint
from   sklearn.metrics            import confusion_matrix, classification_report
from    sklearn.metrics           import precision_recall_fscore_support as score

path = 'ouput_final.json'

with open (path) as json_file:
    data = json.load(json_file)

data = {int(key):value for key, value in data.items()}

accuracy      = np.array([data[i]['accuracy']     for i in data.keys()])
val_accuracy  = np.array([data[i]['val_accuracy'] for i in data.keys()])
loss          = np.array([data[i]['loss']         for i in data.keys()])
val_loss      = np.array([data[i]['val_loss']     for i in data.keys()])
y_tests       = np.array([data[i]['y_test']       for i in data.keys()])
y_preds       = np.array([data[i]['y_pred']       for i in data.keys()])

mean_accuracy     = float('%.5f' % np.mean(accuracy))
mean_val_accuracy = float('%.5f' % np.mean(val_accuracy))
mean_loss         = float('%.5f' % np.mean(loss))
mean_val_loss     = float('%.5f' % np.mean(val_loss))

acc_assoc      = []
val_acc_assoc  = []
loss_assoc     = []
val_loss_min   = []
epochs_list    = []
precision_list = []
recall_list    = []
fscore_list    = []

for i in range(len(val_loss)):
    min_val_loss = min(val_loss[i])
    epoch        = list(val_loss[i]).index(min_val_loss)

    precision, recall, fscore, support=score(y_tests[i], y_preds[i], average='weighted', zero_division=1)
    precision_list.append(precision)
    recall_list.append(recall)
    fscore_list.append(fscore)

    acc_assoc.append((accuracy[i])[epoch])
    val_acc_assoc.append((val_accuracy[i])[epoch])
    loss_assoc.append((loss[i])[epoch])
    val_loss_min.append(min_val_loss)
    epochs_list.append(epoch)

mean_acc_assoc     = float('%.5f' % np.mean(np.array(acc_assoc)))
mean_val_acc_assoc = float('%.5f' % np.mean(np.array(val_acc_assoc)))
mean_loss_assoc    = float('%.5f' % np.mean(np.array(loss_assoc)))
mean_val_loss_min  = float('%.5f' % np.mean(np.array(val_loss_min)))

mean_precision     = float('%.5f' % np.mean(np.array(precision_list)))
mean_recall        = float('%.5f' % np.mean(np.array(recall_list)))
mean_fscore        = float('%.5f' % np.mean(np.array(fscore_list)))

print('==========================================================')
print('Mean values:\n')

print('Accuracy mean: ' + str(mean_accuracy))
print('Validation accuracy mean: ' + str(mean_val_accuracy))
print('Loss mean: ' + str(mean_loss))
print('Validation loss mean: ' + str(mean_val_loss))
print('Precision mean: ' + str(mean_precision))
print('Recall mean: ' + str(mean_recall))
print('F-Score: ' + str(mean_fscore))

print('==========================================================')
print('Mean values associated with the bests validation losses:\n')

print('Accuracy mean: ' + str(mean_acc_assoc))
print('Validation accuracy mean: ' + str(mean_val_acc_assoc))
print('Loss mean: ' + str(mean_loss_assoc))
print('Validation loss mean: ' + str(mean_val_loss_min))

print('==========================================================')