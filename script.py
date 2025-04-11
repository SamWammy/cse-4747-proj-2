import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):

    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 

    samples, features = X.shape
    classes = np.unique(y)
    numbclasses= len(classes)

    means = np.zeros((features,numbclasses))

    for i in range(len(classes)):
        clas = classes[i]
        means[:, i] = np.mean(X[y.flatten()== clas], axis=0)
    covmat = np.zeros((features, features))

    for i in range(len(classes)):
        clas = classes[i]
        x_class= X[y.flatten()==clas]
        n_class= x_class.shape[0]
        x_centered= x_class - means[:,i].reshape(1,-1)
        class_covearience= (x_centered.T @ x_centered ) / n_class

        covmat+=(n_class / samples) * class_covearience

    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    # get deminitions of the training data
    samples, features = X.shape 
    classes = np.unique(y)
    num_classes= len(classes)
    means = np.zeros((features, num_classes)) # this is means matrix initially
    covmats = []


    for i in range(len(classes)):
        clas =classes[i]
        x_class= X[y.flatten()==clas]
        means[:, i] = np.mean(x_class,axis=0)

        x_centered = x_class - means[:,i].reshape(1,-1)
        class_cov = (x_centered.T @ x_centered) / x_class.shape[0] #@ matrix mul operator
        covmats.append(class_cov)


    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    # n is number of tests
    n =Xtest.shape[0]
    classes_num =means.shape[1]

    comatin= inv(covmat) #inverse of covmat (covarience matrixx)

    ypred = np.zeros(n) # intailize predicions and scores
    verification= np.zeros((n, classes_num))

        # this is the function ,math
    for i in range(classes_num):
        mean = means[:,i]
        # linear discriminant function 
        lindis_function1 = Xtest @ comatin @ mean 
        lindis_function2= -0.5 * mean.T @ comatin @ mean
        verification[:,i] = lindis_function1 + lindis_function2

    # getting accuracy 

    ypred= np.argmax(verification,axis=1) +1

    acc = np.mean(ypred==ytest.flatten())

    ypred= ypred.reshape(-1,1) 


    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    #tests and number of classes
    n= Xtest.shape[0]
    classes = means.shape[1]
    # iknitialization 
    ypred= np.zeros(n)
    verification = np.zeros((n,classes))

    #discriminant score

    for i in range(classes):
        mean= means[:, i]
        cov = covmats[i]
        covinv= inv(cov)
        function= -0.5 *np.log(det(cov))

        for j in range(n):
            xcenter= Xtest[j] - mean
            quad= -0.5 * xcenter.T @ covinv @ xcenter
            verification[j,i] = function +quad
    
    ypred = np.argmax(verification,axis=1) +1
    acc = np.mean(ypred== ytest.flatten())
    ypred= ypred.reshape(-1,1) 


    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
    x_T = np.transpose(X)
    X_Ty = x_T @ y
    X_tX = x_T @ X
    inverse = np.linalg.inv(X_tX)
    w = inverse @ X_Ty                                                 
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1  
    d = X.shape[1]  
    lambda_I = lambd * np.identity(d) 
    X_tX = np.transpose(X) @ X
    X_Ty = np.transpose(X) @ y
    w = np.linalg.inv(lambda_I + X_tX) @ X_Ty                                                            

    # IMPLEMENT THIS METHOD                                                   
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    N = Xtest.shape[0]
    #! MSE = 1/N * 2* J(w)
    J_of_w_times2 = np.transpose((ytest - Xtest @ w)) @ (ytest - Xtest @ w)
    msearray = J_of_w_times2 / N
    mse = msearray[0, 0]
    # IMPLEMENT THIS METHOD
    return mse

def regressionObjVal(w, X, y, lambd):
    w = w.reshape(-1, 1)
    y = y.reshape(-1, 1)
    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda 
    y_minus_Xw = y - X @ w
    w_Tw = np.transpose(w) @ w                                                                  
    J_of_w = 1/2 * (np.transpose(y_minus_Xw) @ y_minus_Xw) + (1/2 * lambd * w_Tw)
    error = J_of_w[0, 0]
    
    X_w_minus_y = (X @ w) - y
    lambd_w = lambd * w
    X_t = np.transpose(X)
    error_grad = (X_t @ X_w_minus_y) + lambd_w  
    error_grad = error_grad.flatten()                                          
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
	
    # IMPLEMENT THIS METHOD

    matrixfeat= x.shape[0]
    Xp= np.zeros((matrixfeat,p+1))

    for i in range(p+1):
        Xp[:, i] =np.power(x,i)
    return Xp


# Main script

# # Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
# print(w_i.flatten())
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
# z = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
   
    w_l = learnRidgeRegression(X_i,y,lambd)
    # if z == 6 or z%10 == 0:
    #     print(lambd)
    #     print(w_l.flatten())
    # z += 1
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1]))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0.06 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
