
# coding: utf-8

# ## Predicting Dengue
# ## Muhammad Fuzail Zubari
# ## 18101135

# In[264]:


import pandas as pd
import numpy as np
import collections
import seaborn as sns
import collections

import matplotlib.cm as cm
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
import statsmodels.api as sm

from collections import Counter
from datetime import datetime, date, timedelta
from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# In[265]:


# Loading Data

dengue_features_train = pd.read_csv('dengue_features_train.csv')
dengue_features_test = pd.read_csv('dengue_features_test.csv')
dengue_labels_train = pd.read_csv('dengue_labels_train.csv')


# In[266]:


dengue_features_train.head()


# In[267]:


dengue_features_test.head()


# ### Data Preprocessing Test Data

# In[268]:


dengue_features_test.isnull().sum()


# In[269]:


#Check duplicate rows
np.sum(dengue_features_test.duplicated())


# In[270]:


dengue_features_test.fillna(method='ffill', inplace=True)


# In[271]:


dengue_features_test.head()


# In[272]:


dengue_features_test.isnull().sum()


# ### Data Preprocessing Train Data

# In[273]:


dengue_features_train = pd.merge(dengue_labels_train, dengue_features_train, on=['city','year','weekofyear'])


# In[274]:


dengue_features_train.isnull().sum()


# In[275]:


# Check for duplicated values
np.sum(dengue_features_train.duplicated())


# In[276]:


# Forward or backward 'NaN' data filling
dengue_features_train.fillna(method='ffill', inplace=True)


# In[277]:


dengue_features_train.isnull().sum()


# ### EDA

# In[278]:


# Calculating the correlation
correlations = dengue_features_train.corr()


# In[279]:


cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]

correlations.style.background_gradient(cmap, axis=1)    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})    .set_caption("Hover to magify")    .set_precision(2)    .set_table_styles(magnify())


# In[280]:


# Dropping columns with negative corellation in both 

columns_to_drop = ['reanalysis_tdtr_k', 'year', 'station_diur_temp_rng_c', 'ndvi_nw', 'weekofyear', 'ndvi_ne',
                   'ndvi_se', 'ndvi_sw', 'reanalysis_max_air_temp_k', 'reanalysis_relative_humidity_percent',
                   'reanalysis_max_air_temp_k']

# Remove `week_start_date` string.
dengue_features_train.drop(columns_to_drop, axis=1, inplace=True)
dengue_features_test.drop(columns_to_drop, axis=1, inplace=True)


# In[281]:


dengue_features_train.count()


# ### Training and Testing of Data

# In[285]:


subtrain = dengue_features_train.head(1000)
subtest = dengue_features_train.tail(dengue_features_train.shape[0] - 1000)


# In[286]:


def get_best_model(train, test):
    # Step 1: specify the form of the model
    model_formula = "total_cases ~ 1 + "                     "reanalysis_specific_humidity_g_per_kg + "                     "reanalysis_dew_point_temp_k + "                     "reanalysis_min_air_temp_k + "                     "station_min_temp_c + "                     "station_max_temp_c + "                     "station_avg_temp_c + "                     "reanalysis_air_temp_k"
     
    
    grid = 10 ** np.arange(-8, -3, dtype=np.float64)
                    
    best_alpha = []
    best_score = 1000
        
    # Step 2: Find the best hyper parameter, alpha
    for alpha in grid:
        model = smf.glm(formula=model_formula,
                        data=train,
                        family=sm.families.NegativeBinomial(alpha=alpha))

        results = model.fit()
        predictions = results.predict(test).astype(int)
        score = eval_measures.meanabs(predictions, test.total_cases)

        if score < best_score:
            best_alpha = alpha
            best_score = score

    print('Alpha = ', best_alpha)
    print('Score = ', best_score)
            
    # Step 3: refit on entire dataset
    full_dataset = pd.concat([train, test])
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.NegativeBinomial(alpha=best_alpha))

    fitted_model = model.fit()
    return fitted_model


# In[287]:


best_model = get_best_model(subtrain, subtest)
predictions = best_model.predict(dengue_features_test).astype(int)
print(predictions)


# In[288]:


submission = pd.read_csv("submission_format.csv", index_col=[0, 1, 2])

submission.total_cases = np.concatenate([predictions])
submission.to_csv("predicted_values_3.0.csv")

