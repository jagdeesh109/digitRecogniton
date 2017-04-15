# coding: utf-8
# In[3]:
print ('ab');
# In[20]:
import matplotlib.pyplot as plt
from sklearn import datasets
# In[6]:
digits = datasets.load_digits()
# In[7]:
print(digits.data); 
# In[8]:
digits.target
# In[9]:
digits.images[0]
# In[12]:
from sklearn import svm
clf = svm.SVC(gamma=0.001,C=100.);
# In[13]:
clf.fit(digits.data[:-1], digits.target[:-1]) 
# In[18]:
clf.predict(digits.data[-1:])
# In[23]:
plt.matshow(digits.images[0], cmap=plt.cm.Greys);
plt.show()
