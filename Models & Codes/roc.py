from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
#import sys
import numpy as np
import cPickle as cpkl
import argparse
from scipy import interp
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle

def numericLabels(y,label_dict=None):
        if label_dict:
                y=[label_dict[label] for label in y]
        else:
                labels=list(set(y))
                y=[labels.index(label) for label in y]
        return y

def loadData(filename,label_func=numericLabels,x_end_index=None,label_index=None,label_dict=None):
        #lactation_dict={'A': 1,'B': 2,'C': 2,'D': 3,'E': 4,'F': 4,'G': 4,'H': 4,
        #                'I': 5,'J': 6,'K': 1,'L': 2,'M': 2,'N': 3,'O': 3,'P': 4,
        #                'Q': 4,'R': 4,'S': 5, 'T': 6}


        if label_index==None:
                label_index=-2
        if x_end_index==None:
                x_end_index=100
        x=[]
        y=[]
        print x_end_index,label_index
        if filename.endswith("cpkl"):
                x,y=cpkl.load(open("filename","r"))
        elif filename.endswith("npy"):
                data=np.load(filename)
                x=data[:,:x_end_index]
                y=data[:,label_index]
                y=label_func(y)
        else:
                file_ptr=open(filename,"r")
                for line in file_ptr:
                        x.append(line.split(',')[:x_end_index])
                        y.append(line.split(',')[label_index].strip('"'))
                y=func(y,label_dict)

        x=np.nan_to_num(np.array(x,dtype=np.float32))
	print x.shape
        return x,y

def prepData(x,scaler=None):
        """
        Normalize data
        """
        if not scaler:
                scaler = StandardScaler().fit(x)
        x_transformed= scaler.transform(x)
        return x_transformed,scaler

def calc_fpr_tpr_macro(model,x,y):
	classes=set(y)
	print classes
	y_pred=model.predict_proba(x)
	fpr={}
	tpr={}
	i=0
	for cl in classes:
		fpr[cl],tpr[cl],_=roc_curve(np.array(y),y_pred[:,i],pos_label=cl)
		i+=1
	all_fpr = np.unique(np.concatenate([fpr[cl] for cl in classes]))
	mean_tpr = np.zeros_like(all_fpr)
	for cl in classes:
		mean_tpr += interp(all_fpr, fpr[cl], tpr[cl])
	mean_tpr /= len(classes)
	return all_fpr, mean_tpr
	
	

if __name__=="__main__":
	parser=argparse.ArgumentParser(description='')
	parser.add_argument('--models',nargs='+',required=True)
	parser.add_argument('--labels',nargs='+')
	parser.add_argument('--train_input',required=True)
	parser.add_argument('--label_dict')
	parser.add_argument('--label_index', type=int)
	parser.add_argument('--x_end_index', type=int)
	args=parser.parse_args()
	train_inp=args.train_input
	label_dict=args.label_dict
	label_index=args.label_index
	x_end_index=args.x_end_index

	x_train,y_train = loadData(train_inp,
                                 x_end_index=x_end_index,
                                 label_index=label_index,
                                 label_dict=label_dict)
	x_train,scaler=prepData(x_train)
	x_train=np.nan_to_num(x_train)
	all_fprs=[]
	mean_tprs=[]
	roc_aucs=[]
	for model in args.models:
		f=open(model,"r")
		model=cpkl.load(f)
		all_fpr,mean_tpr=calc_fpr_tpr_macro(model,x_train,y_train)
		all_fprs.append(all_fpr)
		mean_tprs.append(mean_tpr)
		roc_aucs.append(auc(all_fpr,mean_tpr))

	plt.figure()
	plt.xlim((0,1.05))
	plt.ylim((0,1.05))
	colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
	for i,color in zip(range(len(args.labels)),colors):
		plt.plot(all_fprs[i],mean_tprs[i],color=color,
			label='ROC curve of class {0} (area = {1:0.2f})'
             		''.format(args.labels[i], roc_aucs[i]))
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.show()
