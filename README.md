<div class="cell markdown">

This Dataset contains tourists by country to Rio de Janeiro from 2006 to
2019.

  - País: Country.
  - Total: Total Number of Tourists arrived.
  - Aérea: Number of Tourists arrived by Air.
  - Marítima: Number of Tourists arrived by Sea.
  - Região: Continent Region.
  - Continente: Continent.
  - Ano: Year.

</div>

<div class="cell markdown">

Importing Libraries

</div>

<div class="cell code" data-execution_count="79">

``` python
import pandas as pd   

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, cross_val_score

from sklearn.ensemble import BaggingRegressor, StackingRegressor, VotingRegressor

from numpy import mean
import scipy
import numpy as np

import warnings 
warnings.filterwarnings('ignore')
```

</div>

<div class="cell markdown">

Importing dataset and displaying it using head() operation

</div>

<div class="cell code" data-execution_count="108">

``` python
df = pd.read_csv('tourists-rj-2006-2019.csv')
```

</div>

<div class="cell code" data-execution_count="109">

``` python
df.head()
```

<div class="output execute_result" data-execution_count="109">

``` 
               País  Total  Aérea  Marítima  Região Continente   Ano
0     África do Sul   3545   3012       533  África     África  2006
1            Angola  21662  21606        56  África     África  2006
2        Cabo Verde   2407   2407         0  África     África  2006
3           Nigéria    238    233         5  África     África  2006
4            Outros   1900   1783       117  África     África  2006
```

</div>

</div>

<div class="cell markdown">

Renaming non english columns names in English language

</div>

<div class="cell code" data-execution_count="110">

``` python
df.rename(columns = {"País":"Country",
                     "Aérea":"Total_Arrived_by_air",
                     "Marítima":"Total_Arrived_by_sea",
                     "Região":"Region",
                     "Continente":"Continent", 
                     "Ano":"Year"}, inplace = True)
```

</div>

<div class="cell code" data-execution_count="83">

``` python
df.head(1)
```

<div class="output execute_result" data-execution_count="83">

``` 
            Country  Total  Total_Arrived_by_air  Total_Arrived_by_sea  \
0     África do Sul   3545                  3012                   533   

   Region Continent  Year  
0  África    África  2006  
```

</div>

</div>

<div class="cell markdown">

Checking Missing Values

</div>

<div class="cell code" data-execution_count="84">

``` python
df.isna().sum()
```

<div class="output execute_result" data-execution_count="84">

    Country                 0
    Total                   0
    Total_Arrived_by_air    0
    Total_Arrived_by_sea    0
    Region                  0
    Continent               0
    Year                    0
    dtype: int64

</div>

</div>

<div class="cell markdown">

One-Hot Encoding on dataset before model implementation

</div>

<div class="cell code" data-execution_count="85">

``` python
df_onehot = pd.get_dummies(df)
```

</div>

<div class="cell code" data-execution_count="86">

``` python
df_onehot.head()
```

<div class="output execute_result" data-execution_count="86">

``` 
   Total  Total_Arrived_by_air  Total_Arrived_by_sea  Year  \
0   3545                  3012                   533  2006   
1  21662                 21606                    56  2006   
2   2407                  2407                     0  2006   
3    238                   233                     5  2006   
4   1900                  1783                   117  2006   

   Country_   Alemanha  Country_   Angola  Country_   Argentina  \
0                    0                  0                     0   
1                    0                  1                     0   
2                    0                  0                     0   
3                    0                  0                     0   
4                    0                  0                     0   

   Country_   Arábia Saudita  Country_   Austrália  Country_   Bolívia  ...  \
0                          0                     0                   0  ...   
1                          0                     0                   0  ...   
2                          0                     0                   0  ...   
3                          0                     0                   0  ...   
4                          0                     0                   0  ...   

   Region_Europa  Region_Oceania  Region_Oriente Médio  Region_África  \
0              0               0                     0              1   
1              0               0                     0              1   
2              0               0                     0              1   
3              0               0                     0              1   
4              0               0                     0              1   

   Region_Ásia  Continent_América  Continent_Europa  Continent_Oceania  \
0            0                  0                 0                  0   
1            0                  0                 0                  0   
2            0                  0                 0                  0   
3            0                  0                 0                  0   
4            0                  0                 0                  0   

   Continent_África  Continent_Ásia  
0                 1               0  
1                 1               0  
2                 1               0  
3                 1               0  
4                 1               0  

[5 rows x 160 columns]
```

</div>

</div>

<div class="cell markdown">

We will be predicting only one column i.e., Total\_Arrived\_by\_air

So,

  - Dependent Variable (Y): Total\_Arrived\_by\_air
  - Independent Variable (X1, X2, ...): Rest of the columns

</div>

<div class="cell markdown">

Train-Test Split

</div>

<div class="cell code" data-execution_count="87">

``` python
X = df_onehot.drop(columns=['Total', 'Total_Arrived_by_air'])
y = df_onehot[['Total_Arrived_by_air']]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
```

</div>

<div class="cell markdown">

Models Implementations

</div>

<div class="cell code" data-execution_count="88">

``` python
def fitting_models_CV():
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    lr=LinearRegression()
    ls = Lasso(alpha=10.5)
    rg = Ridge(alpha=1.5)
    dt = DecisionTreeRegressor(max_depth=50)
    rfr = RandomForestRegressor()
    gbr = GradientBoostingRegressor()
    knr = KNeighborsRegressor(n_neighbors=20)
    rgs = [('Linear Regression', lr),
        ('Lasso', ls),
        ('Ridge', rg),
        ('Decision Tree', dt),
        ('Random Forest', rfr),
        ('Gradient Boosting', gbr),
        ('KNearest Neighbor',knr)       
    ]
    for name,rg in rgs:
        scores = cross_val_score(rg, X, y, cv=cv) 
        #rg.fit(X_train,y_train)
        #pred = rg.predict(X_test)
        score = format(mean(scores), '.4f')
        print("{} : {}".format(name,score))
```

</div>

<div class="cell code" data-execution_count="89">

``` python
fitting_models_CV()
```

<div class="output stream stdout">

    Linear Regression : -19645401907366692.0000
    Lasso : 0.7696
    Ridge : 0.7981
    Decision Tree : 0.9099
    Random Forest : 0.9253
    Gradient Boosting : 0.9191
    KNearest Neighbor : 0.5483

</div>

</div>

<div class="cell markdown">

`Conclusion`

  - Random Forest Regressor or Gradient Boosting Regressor can be a good
    choice.

  - Using grid search and different approaches, We can further improve
    the accuracy.

</div>

<div class="cell markdown">

Looping to check the gradient boosting Regressor on different train test
split

</div>

<div class="cell code" data-execution_count="90">

``` python
X = df_onehot.drop(columns=['Total', 'Total_Arrived_by_air'])
y = df_onehot[['Total_Arrived_by_air']]
scores = []

for each_outer in range(0,10):
    for each_inner in range(0,10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
        reg = GradientBoostingRegressor(max_depth=5, n_estimators=100)
        reg.fit(X_train, y_train)

        reg.predict(X_test)
        accuracy = reg.score(X_test, y_test)
        scores.append(accuracy)
    
    print(str(each_outer) + ' - ' + str(mean(scores)))
    
```

<div class="output stream stdout">

    0 - 0.9016216232564723
    1 - 0.9099734265477515
    2 - 0.9187502638986772
    3 - 0.9190877249796394
    4 - 0.9117061650294278
    5 - 0.9128271392412147
    6 - 0.9090598811674316
    7 - 0.9122879724603861
    8 - 0.9128118306346205
    9 - 0.9131280127928215

</div>

</div>

<div class="cell markdown">

So the accuracy remains between 87%-92%

</div>

<div class="cell markdown">

Looping to check the Random Forest Regressor on different train test
split

</div>

<div class="cell code" data-execution_count="91">

``` python
X = df_onehot.drop(columns=['Total', 'Total_Arrived_by_air'])
y = df_onehot[['Total_Arrived_by_air']]
scores = []

for each_outer in range(0,10):
    for each_inner in range(0,10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
        reg = RandomForestRegressor(max_depth=5, n_estimators=50)
        reg.fit(X_train, y_train)

        reg.predict(X_test)
        accuracy = reg.score(X_test, y_test)
        scores.append(accuracy)
    
    print(str(each_outer) + ' - ' + str(mean(scores)))
    
```

<div class="output stream stdout">

    0 - 0.8420306264268463
    1 - 0.8596938639889329
    2 - 0.8652225529131906
    3 - 0.8615512073244013
    4 - 0.8669171167472948
    5 - 0.8701198807036639
    6 - 0.8718756478845096
    7 - 0.8755695222523376
    8 - 0.8713422726127279
    9 - 0.8700654390512358

</div>

</div>

<div class="cell markdown">

So the accuracy remains between 84%-90%

</div>

<div class="cell markdown">

Grid Search Code for finding best Gradient Boosting Algorithm parameters

</div>

<div class="cell code" data-execution_count="92">

``` python
# cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)
# regRF = GradientBoostingRegressor(max_depth=5, random_state=0, n_estimators=50)
# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [5, 10, 15],
#     'max_features': [2, 3, 4],    
#     'min_samples_split': [8, 10, 12],
#     'n_estimators': [100, 200, 300] 
# }
# grid_search = GridSearchCV(estimator = regRF, param_grid=param_grid, cv = cv, n_jobs = -1, verbose = 2)
# grid_search.fit(X, y)
# best_grid = grid_search.best_estimator_
# print(best_grid)
print('Can\'t run the code here, it takes too much time!')
```

<div class="output stream stdout">

    Can't run the code here, it takes too much time!

</div>

</div>

<div class="cell markdown">

Bagging Regressor

</div>

<div class="cell code" data-execution_count="93">

``` python
cv = RepeatedKFold(n_splits=10, n_repeats=1)#, random_state=1)
reg_bg = BaggingRegressor(base_estimator=GradientBoostingRegressor(max_depth=5, n_estimators=100),
                        n_estimators=20, random_state=0)
scores = cross_val_score(reg_bg, X, y, cv=cv)
score = format(mean(scores), '.4f')
print(score)
```

<div class="output stream stdout">

    0.9004

</div>

</div>

<div class="cell markdown">

Stacking Regressor best three models (Decision Tree, Random Forest,
Gradient Boosting)

</div>

<div class="cell code" data-execution_count="94">

``` python
cv = RepeatedKFold(n_splits=10, n_repeats=1)#, random_state=1)
estimators = [
('dt', DecisionTreeRegressor(max_depth=50)),
('rf', RandomForestRegressor(max_depth=5, n_estimators=50))
]

reg_sr = StackingRegressor(estimators=estimators, final_estimator=GradientBoostingRegressor(max_depth=5, n_estimators=100, random_state=42))
scores = cross_val_score(reg_sr, X, y, cv=cv)
score = format(mean(scores), '.4f')
print(score)
```

<div class="output stream stdout">

    0.7380

</div>

</div>

<div class="cell markdown">

Voting Regressor with best models

</div>

<div class="cell code" data-execution_count="95">

``` python
cv = RepeatedKFold(n_splits=10, n_repeats=1)#, random_state=1)
r1 = DecisionTreeRegressor(max_depth=50)
r2 = RandomForestRegressor(max_depth=5,n_estimators=50)
r3 = GradientBoostingRegressor(max_depth=5,n_estimators=100)

reg_vr = VotingRegressor([('dt', r1), ('rf', r2),('gb', r3)])
scores = cross_val_score(reg_vr, X, y, cv=cv)
score = format(mean(scores), '.4f')
print(score)
```

<div class="output stream stdout">

    0.9207

</div>

</div>

<div class="cell markdown">

Log transformation

</div>

<div class="cell code" data-execution_count="96">

``` python
numeric_train_columns = pd.DataFrame(df.select_dtypes(np.number).columns, columns=['Feature'])
numeric_train_columns['Skew'] = numeric_train_columns['Feature'].apply(lambda feature: scipy.stats.skew(df[feature]))
numeric_train_columns['Abs Skew'] = numeric_train_columns['Skew'].apply(abs)
numeric_train_columns['Skewed'] = numeric_train_columns['Abs Skew'].apply(lambda x: True if x > 0.5 else False)
numeric_train_columns
```

<div class="output execute_result" data-execution_count="96">

``` 
                Feature       Skew   Abs Skew  Skewed
0                 Total   5.676921   5.676921    True
1  Total_Arrived_by_air   5.528274   5.528274    True
2  Total_Arrived_by_sea  10.656239  10.656239    True
3                  Year  -0.263379   0.263379   False
```

</div>

</div>

<div class="cell code" data-execution_count="99">

``` python
for column in numeric_train_columns.query('Skewed == True')['Feature'].values:
    df[column] = np.log1p(df[column])
```

</div>

<div class="cell code" data-execution_count="101">

``` python
df.head()
```

<div class="output execute_result" data-execution_count="101">

``` 
            Country     Total  Total_Arrived_by_air  Total_Arrived_by_sea  \
0     África do Sul  8.173575              8.010692              6.280396   
1            Angola  9.983361              9.980773              4.043051   
2        Cabo Verde  7.786552              7.786552              0.000000   
3           Nigéria  5.476464              5.455321              1.791759   
4            Outros  7.550135              7.486613              4.770685   

   Region Continent  Year  
0  África    África  2006  
1  África    África  2006  
2  África    África  2006  
3  África    África  2006  
4  África    África  2006  
```

</div>

</div>

<div class="cell code" data-execution_count="102">

``` python
df_onehot = pd.get_dummies(df)
```

</div>

<div class="cell code" data-execution_count="104">

``` python
df_onehot.head(1)
```

<div class="output execute_result" data-execution_count="104">

``` 
      Total  Total_Arrived_by_air  Total_Arrived_by_sea  Year  \
0  8.173575              8.010692              6.280396  2006   

   Country_   Alemanha  Country_   Angola  Country_   Argentina  \
0                    0                  0                     0   

   Country_   Arábia Saudita  Country_   Austrália  Country_   Bolívia  ...  \
0                          0                     0                   0  ...   

   Region_Europa  Region_Oceania  Region_Oriente Médio  Region_África  \
0              0               0                     0              1   

   Region_Ásia  Continent_América  Continent_Europa  Continent_Oceania  \
0            0                  0                 0                  0   

   Continent_África  Continent_Ásia  
0                 1               0  

[1 rows x 160 columns]
```

</div>

</div>

<div class="cell code" data-execution_count="105">

``` python
X = df_onehot.drop(columns=['Total', 'Total_Arrived_by_air'])
y = df_onehot[['Total_Arrived_by_air']]
scores = []

for each_outer in range(0,10):
    for each_inner in range(0,10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
        reg = GradientBoostingRegressor(max_depth=5, n_estimators=100)
        reg.fit(X_train, y_train)

        reg.predict(X_test)
        accuracy = reg.score(X_test, y_test)
        scores.append(accuracy)
    
    print(str(each_outer) + ' - ' + str(mean(scores)))
    
```

<div class="output stream stdout">

    0 - 0.8612242266823978
    1 - 0.8539696739614522
    2 - 0.8494167830762074
    3 - 0.8491289143110732
    4 - 0.8547752141900256
    5 - 0.8538825642012021
    6 - 0.8539761601932411
    7 - 0.8548899608817893
    8 - 0.855318166066103
    9 - 0.8558929888058384

</div>

</div>

<div class="cell markdown">

`Conclusion`

  - Accuracy decreases on taking log transformation of data

</div>

<div class="cell markdown">

-----

</div>

<div class="cell markdown">

### Winner Model - Gradient Boosting Regressor (Without Log transformation)

</div>

<div class="cell code" data-execution_count="113">

``` python
df_onehot = pd.get_dummies(df)
X = df_onehot.drop(columns=['Total', 'Total_Arrived_by_air'])
y = df_onehot[['Total_Arrived_by_air']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


reg = GradientBoostingRegressor(max_depth=5, n_estimators=100)
reg.fit(X_train, y_train)

reg.predict(X_test)
reg.score(X_test, y_test)
```

<div class="output execute_result" data-execution_count="113">

    0.9350594797238077

</div>

</div>
