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
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier

def create_mlp(num_features=3000,num_classes=20,init='uniform',optimizer='adam'):
	
	model=Sequential()
	model.add(Dropout(0.1,input_shape=(num_features,)))
        model.add(Dense(2000, kernel_initializer=init,
			activation='relu', input_dim=num_features))
        model.add(Dropout(0.5))
       	model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(500, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy',
                     optimizer=optimizer,
                     metrics=['accuracy'])
	return model

def mlp(x,y,parameters):
	features=x.shape[1]
	classes=len(set(y))
	model = KerasClassifier(build_fn=create_mlp, num_features=features, num_classes=classes, verbose=0)
	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(estimator=model,param_grid=parameters, cv=cv, verbose=100, n_jobs=4)
	grid.fit(x,y)
	
def GaussianProcess(x,y,parameters=None):
	cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores=cross_val_score(GaussianProcessClassifier(1.0 * RBF(1.0)), x, y, cv=cv, n_jobs=5, verbose=100)
	model=GaussianProcessClassifier(1.0 * RBF(1.0)).fit(x,y)
	return model

def naiveBayes(x,y,parameters=None):
	cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	scores=cross_val_score(GaussianNB(), x, y, cv=cv, n_jobs=4, verbose=100,
				scoring='accuracy')
	model=GaussianNB().fit(x,y)
	return model

def svm(x,y,parameters):
	print "Running SVM model"
	#C_range = np.logspace(-2, 10, 13)
	#gamma_range = np.logspace(-9, 3, 13)
	#param_grid = dict(gamma=gamma_range, C=C_range)
	inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	#outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	grid = GridSearchCV(SVC(probability=True), param_grid=parameters, cv=inner_cv, verbose=100, 
			n_jobs=4, scoring=['f1_weighted','accuracy'],refit='f1_weighted')
	grid.fit(x, y)
	#non_nested_score=grid.best_score_
	#nested_score = cross_val_score(grid, X=x, y=y, cv=outer_cv)
	#print "Non-nested - Nested: " + str(non_nested_score - nested_score.mean())
	return grid

def evaluate_params(clf_func,x,y,param_dict):
	"""
	Function to evaluate individual parameters as a
	first pass to identify accuracy trends for each
	parameter in an effort to narrow down parameter ranges 
	before running on all parameter combinations.
	"""
	#clf_func=locals[clf_func]
	g=[]
	for param in param_dict:
		g.append({param:[param_dict[param]]})		
	
	grid=clf_func(x,y,ParameterGrid(g))
	return grid

def cross_validation_split(n_splits, shuffle):
	return StratifiedKFold(n_splits=n_splits,shuffle=shuffle, random_state=42)

def randomForest(x,y,parameters):
	print "Running Random Forest model"
	#parameters = dict(n_estimators=[50, 100, 500, 1000, 2000], min_samples_leaf=[50, 75, 100, 200, 500])
	clf=RandomForestClassifier(random_state=42, oob_score=True)
	cv=cross_validation_split(5,True)
	#inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	#outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	grid=GridSearchCV(clf,cv=cv,verbose=100,n_jobs=4,param_grid=parameters,
			scoring=['f1_weighted','accuracy'],refit='f1_weighted')
	grid.fit(x,y)
	#non_nested_score=grid.best_score_
	#nested_score = cross_val_score(grid, X=x, y=y, cv=outer_cv)
	#print "Non-nested - Nested: " + str(non_nested_score - nested_score.mean())
	return grid

def prepData(x,scaler=None):
	"""
	Normalize data
	"""
	if not scaler:
		scaler = StandardScaler().fit(x)
	x_transformed= scaler.transform(x)
	return x_transformed,scaler

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

if __name__=="__main__":
	parser=argparse.ArgumentParser(description='')
	parser.add_argument('--train_input',required=True)
	parser.add_argument('--test', type=float, required=True)
	parser.add_argument('--algorithm',required=True)
	parser.add_argument('--param_dict',required=True)
	parser.add_argument('--label_dict')
	parser.add_argument('--label_index', type=int)
	parser.add_argument('--x_end_index', type=int)
	parser.add_argument('--output_file')
	parser.add_argument('--eval_params', action='store_true')
	parser.add_argument('--dump_data', action='store_true')
	#parser.add_argument('--test', action='store_true')
	args=parser.parse_args()
	method=locals()[args.algorithm]
	train_inp=args.train_input
	label_dict=args.label_dict
	label_index=args.label_index
	x_end_index=args.x_end_index
	test_size=args.test
	if label_dict:
		f=open(label_dict,'r')
		label_dict=cpkl.load(f)
		f.close()
	f=open(args.param_dict,'r')
	param_dict=cpkl.load(f)
	f.close()
	print "Loading Training Data"
	if test_size > 0:
		x_train,y_train, x_test, y_test = train_test_split(loadData(train_inp,
							x_end_index=x_end_index,
							label_index=label_index,
							label_dict=label_dict),
							test_size=test_size, 
							random_state=42)

		x_train,scaler=prepData(x_train)
		x_train=np.nan_to_num(x_train)
		x_test,_=prepData(x_test,scaler)
		x_test=np.nan_to_num(x_test)
	else:
		x_train,y_train = loadData(train_inp,
                                 x_end_index=x_end_index,
                                 label_index=label_index,
                                 label_dict=label_dict)
		x_train,scaler=prepData(x_train)
                x_train=np.nan_to_num(x_train)

	if args.dump_data:
		f=open(train_inp+'.cpkl','w')
        	cpkl.dump([x_train,y_train],f,-1)
        	f.close()
	if args.eval_params:
		model=evaluate_params(method,x_train,y_train,param_dict)
	else:	
		model=method(x_train,y_train,param_dict)
	if test_size > 0:
		test_score=model.score(x_test,y_test)
		print "Test score: "+str(test_score)
	if args.output_file:
		f=open(args.output_file,'w')
		cpkl.dump(model,f,-1)
		f.close()
