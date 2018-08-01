from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, ParameterGrid, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error
#import sys
import numpy as np
import cPickle as cpkl
import argparse
import keras
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
#from keras.utils import plot_model

def create_mlp_old(num_features=3000,num_classes=20,
			encoding_dims=[2000,1000,500],
			regs=[0.1, 0.5, 0.5],init='uniform',
			optimizer='adam'):

        model=Sequential()
        model.add(Dropout(regs[0],input_shape=(num_features,)))
        model.add(Dense(encoding_dims[0], kernel_initializer=init,
                        activation='relu', input_dim=num_features))
        model.add(Dropout(regs[1]))
        model.add(Dense(encoding_dims[1], activation='relu'))
        model.add(Dropout(regs[2]))
        model.add(Dense(encoding_dims[2], activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                     optimizer=optimizer,
                     metrics=['accuracy'])
        return model

def create_mlp(num_features=3000,num_classes=20,encoding_dims=[2000],
		regs=[0.00001],init='uniform',optimizer='adam'):
	
	#inp = Input(shape=(num_features,))
	model = Sequential()
	model.add(Dense(encoding_dims[0], input_shape=(num_features,),activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(regs[0]))
	#x = inp
	for dim, reg in zip(encoding_dims[1:],regs[1:]):
		model.add(Dense(dim, kernel_initializer = init, activation = 'relu'))
	#	x = Dense(dim, kernel_initializer = init,
        #                        activation='relu')(x)
	#	x = Dropout(reg)(x)
		model.add(BatchNormalization())
		model.add(Dropout(reg))
	model.add(Dense(num_classes, activation = 'softmax'))
	#predictions = Dense(num_classes, activation='softmax')(x)
	#model = Model(inputs=inp, outputs=predictions)
	model.compile(loss='categorical_crossentropy',
                     optimizer=optimizer,
                     metrics=['accuracy'])
	#plot_model(model, to_file='dl_model.png')
	print model.summary()
	return model

def mlp(x,y,parameters):
	features=x.shape[1]
	classes=len(set(y))
	models=[]
        test_accs=[]
        test_f1s=[]
	for tr_index, ts_index in StratifiedKFold(n_splits=4, shuffle=True, random_state=42).split(x,y):
                print "\n\nNew Outer CV\n\n"
		stopper = EarlyStopping(monitor='val_loss', min_delta = 0.01, patience=3, verbose=1)
		model = KerasClassifier(build_fn=create_mlp, num_features=features, 
					num_classes=classes, verbose=2)
                grid = GridSearchCV(model, param_grid=parameters, verbose=100,
                                n_jobs=4, scoring=['f1_weighted','accuracy'],refit='f1_weighted')
		stopper = EarlyStopping(monitor='val_loss', min_delta = 0.01, patience=3, verbose=1)
                grid.fit(x[tr_index], y[tr_index])
                best_model = grid.best_estimator_
                best_model_preds = best_model.predict(x[ts_index])
                test_acc = accuracy_score(y[ts_index],best_model_preds)
                test_f1 = f1_score(y[ts_index], best_model_preds, average="weighted")
                models.append(best_model)
                test_accs.append(test_acc)
                test_f1s.append(test_f1)
        best_test_acc = max(test_accs)
        best_test_model = models[test_accs.index(best_test_acc)]
        print "\n\n Nested CV Results: {0} {1} {2} {3}\n\n".format(np.mean(test_accs), np.std(test_accs), np.mean(test_f1s), np.std(test_f1s))
        return best_test_model


def gbm(x,y,parameters=None):
        models=[]
        test_accs=[]
        test_f1s=[]
        for tr_index, ts_index in StratifiedKFold(n_splits=4, shuffle=True, random_state=42).split(x,y):
                print "\n\nNew Outer CV\n\n"
                grid = GridSearchCV(GradientBoostingClassifier(), param_grid=parameters, verbose=100,
                                n_jobs=4, scoring=['f1_weighted','accuracy'],refit='f1_weighted')
                grid.fit(x[tr_index], y[tr_index])
                best_model = grid.best_estimator_
                best_model_preds = best_model.predict(x[ts_index])
                test_acc = accuracy_score(y[ts_index],best_model_preds)
                test_f1 = f1_score(y[ts_index], best_model_preds, average="weighted")
                models.append(best_model)
                test_accs.append(test_acc)
                test_f1s.append(test_f1)
        best_test_acc = max(test_accs)
        best_test_model = models[test_accs.index(best_test_acc)]
        print "\n\n Nested CV Results: {0} {1} {2} {3}\n\n".format(np.mean(test_accs), np.std(test_accs), np.mean(test_f1s), np.std(test_f1s))
        return best_test_model

	
def GaussianProcess(x,y,parameters=None):
	cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores=cross_val_score(GaussianProcessClassifier(1.0 * RBF(1.0)), x, y, cv=cv, n_jobs=5, verbose=100)
	model=GaussianProcessClassifier(1.0 * RBF(1.0)).fit(x,y)
	return model

def naiveBayes(x,y,parameters=None):
	cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
	scores=cross_val_score(GaussianNB(), x, y, cv=cv, n_jobs=4, verbose=100,
				scoring='accuracy')
	model=GaussianNB().fit(x,y)
	return model

def svr(x,y,parameters):
        print "Running SVR model"
        models=[]
        test_accs=[]
        test_f1s=[]
        for tr_index, ts_index in StratifiedKFold(n_splits=4, shuffle=True, random_state=42).split(x,y):
                print "\n\nNew Outer CV\n\n"
                grid = GridSearchCV(SVR(), param_grid=parameters, verbose=100,
                                n_jobs=4, scoring=['neg_mean_squared_error'],refit='neg_mean_squared_error')
                grid.fit(x[tr_index], y[tr_index])
                best_model = grid.best_estimator_
                best_model_preds = best_model.predict(x[ts_index])
                test_acc = mean_squared_error(y[ts_index],best_model_preds)
                models.append(best_model)
                test_accs.append(test_acc)
        best_test_acc = min(test_accs)
        best_test_model = models[test_accs.index(best_test_acc)]
        print "\n\n Nested CV Results: {0} {1}\n\n".format(np.mean(test_accs), np.std(test_accs))
        return best_test_model

def svm(x,y,parameters):
	print "Running SVM model"
	#C_range = np.logspace(-2, 10, 13)
	#gamma_range = np.logspace(-9, 3, 13)
	#param_grid = dict(gamma=gamma_range, C=C_range)
	models=[]
	test_accs=[]
	test_f1s=[]
	for tr_index, ts_index in StratifiedKFold(n_splits=4, shuffle=True, random_state=42).split(x,y):
		print "\n\nNew Outer CV\n\n"	
	#outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
		grid = GridSearchCV(SVC(probability=True), param_grid=parameters, verbose=100, 
				n_jobs=4, scoring=['f1_weighted','accuracy'],refit='f1_weighted')
		grid.fit(x[tr_index], y[tr_index])
		best_model = grid.best_estimator_
		best_model_preds = best_model.predict(x[ts_index])
		test_acc = accuracy_score(y[ts_index],best_model_preds)
		test_f1 = f1_score(y[ts_index], best_model_preds, average="weighted")	
		models.append(best_model)
		test_accs.append(test_acc)
		test_f1s.append(test_f1)	
	#non_nested_score=grid.best_score_
	#nested_score = cross_val_score(grid, X=x, y=y, cv=outer_cv)
	#print "Non-nested - Nested: " + str(non_nested_score - nested_score.mean())
	best_test_acc = max(test_accs)
	best_test_model = models[test_accs.index(best_test_acc)]
	print "\n\n Nested CV Results: {0} {1} {2} {3}\n\n".format(np.mean(test_accs), np.std(test_accs), np.mean(test_f1s), np.std(test_f1s))
	return best_test_model

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
	"""
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
	"""
	models=[]
        test_accs=[]
        test_f1s=[]
        for tr_index, ts_index in StratifiedKFold(n_splits=4, shuffle=True, random_state=42).split(x,y):
                print "\n\nNew Outer CV\n\n"
        #outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                grid = GridSearchCV(RandomForestClassifier(random_state=42, oob_score=True), param_grid=parameters, verbose=100,
                                n_jobs=4, scoring=['f1_weighted','accuracy'],refit='f1_weighted')
                grid.fit(x[tr_index], y[tr_index])
                best_model = grid.best_estimator_
                best_model_preds = best_model.predict(x[ts_index])
                test_acc = accuracy_score(y[ts_index],best_model_preds)
                test_f1 = f1_score(y[ts_index], best_model_preds, average="weighted")
                models.append(best_model)
                test_accs.append(test_acc)
                test_f1s.append(test_f1)
        #non_nested_score=grid.best_score_
        #nested_score = cross_val_score(grid, X=x, y=y, cv=outer_cv)
        #print "Non-nested - Nested: " + str(non_nested_score - nested_score.mean())
        best_test_acc = max(test_accs)
        best_test_model = models[test_accs.index(best_test_acc)]
        print "\n\n Nested CV Results: {0} {1} {2} {3}\n\n".format(np.mean(test_accs), np.std(test_accs), np.mean(test_f1s), np.std(test_f1s))
        return best_test_model

def randomForestRegressor(x,y,parameters):
        models=[]
        test_accs=[]
        for tr_index, ts_index in StratifiedKFold(n_splits=4, shuffle=True, random_state=42).split(x,y):
                print "\n\nNew Outer CV\n\n"
        #outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                grid = GridSearchCV(RandomForestRegressor(random_state=42, oob_score=True), param_grid=parameters, verbose=100,
                                n_jobs=4, scoring=['neg_mean_squared_error'],refit='neg_mean_squared_error')
                grid.fit(x[tr_index], y[tr_index])
                best_model = grid.best_estimator_
                best_model_preds = best_model.predict(x[ts_index])
                test_acc = mean_squared_error(y[ts_index],best_model_preds)
                models.append(best_model)
                test_accs.append(test_acc)
        #non_nested_score=grid.best_score_
        #nested_score = cross_val_score(grid, X=x, y=y, cv=outer_cv)
        #print "Non-nested - Nested: " + str(non_nested_score - nested_score.mean())
        best_test_acc = min(test_accs)
        best_test_model = models[test_accs.index(best_test_acc)]
        print "\n\n Nested CV Results: {0} {1}\n\n".format(np.mean(test_accs), np.std(test_accs))
        return best_test_model

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

		y=label_func(y,label_dict)

	x=np.nan_to_num(np.array(x,dtype=np.float32))
	y=np.nan_to_num(np.array(y,dtype=np.float32))
	print x.shape, y.shape
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
		model = evaluate_params(method,x_train,y_train,param_dict)
	else:	
		model=method(x_train,y_train,param_dict)
	if test_size > 0:
		test_score=model.score(x_test,y_test)
		print "Test score: "+str(test_score)
	if args.output_file:
		f=open(args.output_file,'w')
		cpkl.dump(model,f,-1)
		f.close()
