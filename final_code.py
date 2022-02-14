#import necessary packages
import pandas as pd
import numpy as np 
import os
import glob
from pyomo.environ import *

#compile raw data
path = './raw_1yr/'
allFiles = sorted(glob.glob(path+'*.csv'))
list_= []
for file_ in allFiles:
    df = pd.read_csv(file_, index_col=None)
    list_.append(df)
frame = pd.concat(list_, axis=1, ignore_index = False)

#retrieve stock abbre. from file name
file_name = os.listdir(path) #get all file names from directory
file_name.sort() #sort alphabetically
file_name.remove('.DS_Store') #remove DS_Store
stock_name = [s.replace('.csv', '') for s in file_name] #remove'.csv'

#retrieve date
date = list(frame.iloc[:,0])

#create main frame
framea = frame['Close']
framea.columns = stock_name
framea.index = date

#parameter (please change the value here)
#notice that the return is calculated on a daily basis, so if one would like to get a 5% return annually, 
#the RoR should be 0.002 since (1+0.002)^253 = 0.05 (10% annually: 0.004, 15% annually: 0.0055)
UIC = 100000     #Upper intial capital
LIC = 50000        #Lower intial capital
T   = 126         #Time period (days)
RoR = 0.002      #Required rate of return
fc  = 50          #Fixed transactional cost
pc  = 0.01        #Proportional transactional cost
proportion = 0.3

#range default
stock_j = range(1,102)
time_t  = range(1,T+1)

#create stock price dictionary
d = {}
for i in stock_j:
    d['p%s' % i] = np.array(framea.iloc[:,i-1])

#derive return in given T
return_t = {}
return_mean = []
for j in stock_j:  
    returns = []
    for t in time_t: 
        returns.append( (d['p%s' % j][t]-d['p%s' % j][t-1]) / d['p%s' % j][t-1] ) #d[x][0] is 12/31
    return_t[j] = returns
    return_mean.append( sum(returns)/T )
risk_diff = {}
for j in stock_j:
    risks = []
    for t in time_t: 
        risks.append(return_t[j][t-1] - return_mean[j-1] ) #return_mean is a list, so location should minus 1
    risk_diff[j] = risks


# define variables
model = ConcreteModel()
model.x = Var(stock_j, within = NonNegativeIntegers) #101 variables
model.R = Var(time_t, within = NonNegativeReals)
model.CC = Var(within=NonNegativeReals)
model.zf = Var(within=Binary)
model.zp = Var(within=Binary)

# define constraints
def risk1(model, t):
    return -sum( model.x[j] * risk_diff[j][t-1] for j in stock_j) <= model.R[t]
def risk2(model, t):
    return model.R[t] >= 0
def trans1(model):
    return sum( pc * d['p%s' % j][1] * model.x[j] for j in stock_j) >= fc * (1 - model.zf )
def trans2(model):
    return sum( pc * d['p%s' % j][1] * model.x[j] for j in stock_j) * model.zf <= fc 
def exclusive(model):
    return model.zf == 1 - model.zp
def capital(model):
    return sum( (1+ pc * model.zp ) * d['p%s' % j][1] * model.x[j] for j in stock_j) + fc * model.zf == model.CC
def capital_LIC(model):
    return sum( (1+ pc * model.zp ) * d['p%s' % j][1] * model.x[j] for j in stock_j) + fc * model.zf >= LIC
def capital_UIC(model):
    return sum( (1+ pc * model.zp ) * d['p%s' % j][1] * model.x[j] for j in stock_j) + fc * model.zf <= UIC
def require_of_return (model):
    return sum(  return_mean[j-1] * d['p%s' % j][1] * model.x[j] for j in stock_j )  >= RoR * model.CC
def hold(model, j):
    return d['p%s' % j][1] * model.x[j] <= proportion * (model.CC - fc * model.zf)/(1+pc)

model.c1 = Constraint(time_t, rule = risk1)
model.c2 = Constraint(time_t, rule = risk2)
model.c3 = Constraint(rule = trans1)
model.c4 = Constraint(rule = trans2)
model.c5 = Constraint(rule = exclusive)
model.c6 = Constraint(rule = capital)
model.c7 = Constraint(rule = capital_LIC)
model.c8 = Constraint(rule = capital_UIC)
model.c9 = Constraint(rule = require_of_return)
model.c10 = Constraint(stock_j, rule = hold)

#run and select solver
def objrule(model):
    return sum( model.R[t] for t in time_t) /T
model.obj = Objective(rule = objrule, sense = minimize)
solver = SolverFactory('ipopt')
solver.solve(model)

#display result
print("Risk : ", round(model.obj(),3))
print("")
print("Portfolio:")
for j in stock_j:
    if model.x[j].value >= 1.0:
        print(stock_name[j-1],'=', round(model.x[j].value,2))
