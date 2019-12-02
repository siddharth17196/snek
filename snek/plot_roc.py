import sys
import pickle
import numpy
import torch
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

y_test = []
y_pred = []

file1 = sys.argv[1]
file2 = sys.argv[2]

color_grayscale = sys.argv[3]

with open(file1, 'rb') as file:
	test_y = pickle.load(file)
	for i in test_y:
		y_test.append(i.cpu().numpy())

with open(file2, 'rb') as file:
	pred_y = pickle.load(file)
	for i in pred_y:
		y_pred.append(i.cpu().numpy())

y_test = numpy.array(y_test)
y_pred = numpy.array(y_pred)

classes = list(set(y_test))
print(len(classes))
print(metrics.accuracy_score(y_test, y_pred))
y_test = label_binarize(y_test,classes)
y_pred = label_binarize(y_pred,classes)
plt.figure()
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = len(classes)

mean_fpr = numpy.linspace(0,1,100)
tprs = []
aucs = []
for i in range(n_classes):
	fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
	roc_auc[i] = auc(fpr[i], tpr[i])
	aucs.append(roc_auc[i])
	tprs.append(interp(mean_fpr, fpr[i], tpr[i]))
	tprs[-1][0] = 0.0
	plt.plot(fpr[i], tpr[i], lw=1, alpha=0.3)

mean_tpr = numpy.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = numpy.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
		 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
		 lw=2, alpha=.8)

std_tpr = numpy.std(tprs, axis=0)
tprs_upper = numpy.minimum(mean_tpr + std_tpr, 1)
tprs_lower = numpy.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
				 label=r'$\pm$ 1 std. dev.')

plt.plot([0, 1], [0, 1], color='navy', alpha=0.8, lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for densenet with '+ color_grayscale +' images')
plt.legend(loc="lower right")
plt.show()