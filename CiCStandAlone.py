# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 11:26:38 2018

@author: kmayr
"""

##################################
########     CIC Code     ########
##   Based on Athey and Imbens  ##
########### (2006) ###############
##################################

######################################################
############## Generated Sample Version ##############
######################################################

#######################
### IMPORT PACKAGES ###
#######################
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

###############################
### Choose Model Parameters ###
###############################
wageNTreat = 8
wageTreat = 9
meanerrorNTreat = 2
meanerrorTreat = 1
timeeffect = 1.5
samplesz = 2000
treateffect = 1

# No of quantiles to analyze
noquantiles = 10

# Add noise to the treatment effect?
randomtreateffect = False

# Graph parameters
_graph = True
_graphszx = 10
_graphszy = 5

# How many decimals?
_decimals = 5
_timedecimals = 2

# Set your favorite seed
_seed = 4649

###########################
######## SET TIMER ########
######## SET SEED  ########
###########################
np.random.seed(_seed)
_t0 = time.time()
print('\n Off we go! \n')

###################################
########   GENERATE DATA   ########
###################################
NTreatObsT0 = [wageNTreat]*samplesz
NTreatObsT0 = list(map(lambda x: x+np.random.normal(meanerrorNTreat), NTreatObsT0))
NTreatObsT1 = list(map(lambda x: x+timeeffect, NTreatObsT0))

NTreatObsT0 = pd.DataFrame(NTreatObsT0, columns =['Income'])
NTreatObsT1 = pd.DataFrame(NTreatObsT1, columns =['Income'])

TreatObsT0 = [wageTreat]*samplesz
TreatObsT0 = list(map(lambda x: x+np.random.normal(meanerrorTreat), TreatObsT0))

if randomtreateffect == True:
    TreatObsT1 = list(map(lambda x: x+timeeffect+np.random.normal(treateffect), TreatObsT0))
else:
    TreatObsT1 = list(map(lambda x: x+timeeffect+treateffect, TreatObsT0))

TreatObsT1 = pd.DataFrame(TreatObsT1, columns =['Income'])
TreatObsT0 = pd.DataFrame(TreatObsT0, columns =['Income'])

#%%
################################
####### BEGIN ESTIMATION #######
################################

### Sort data by income (ascending) and add quantiles ###
NTreatObsT0 = NTreatObsT0.sort_values(['Income'])
NTreatObsT0.loc[:, 'qtile'] = NTreatObsT0['Income'].rank(pct=True)

TreatObsT0 = TreatObsT0.sort_values(['Income'])
TreatObsT0.loc[:, 'qtile'] = TreatObsT0['Income'].rank(pct=True)

NTreatObsT1 = NTreatObsT1.sort_values(['Income'])
NTreatObsT1.loc[:, 'qtile'] = NTreatObsT1['Income'].rank(pct=True)

TreatObsT1 = TreatObsT1.sort_values(['Income'])
TreatObsT1.loc[:, 'qtile'] = TreatObsT1['Income'].rank(pct=True)

### Initialize storage lists ###
IncHist = []
QuantInc01Hist = []
Inc00QuantInc01Hist = []
QuantInc00QuantInc01Hist = []

### Generate counterfactual distribution following page 440 equation (9) ###
for _i in np.nditer(NTreatObsT1['Income']):
    ### Find the quantile for each income level in the non-treated group at t=1 ###
    _temp = NTreatObsT1['qtile'].where(NTreatObsT1['Income']==_i)
    ### Remove NaN values ###
    _temp = _temp[~np.isnan(_temp)]
    ### Keep only one value in case of duplicates ###
    _temp = _temp.iloc[0]
    ### Using the quantile from step 1, find the income in the non-treated group at t=0 with the quantile closest to the previously estimated quantile ###
    _temp2 = NTreatObsT0['Income'].where(NTreatObsT0['qtile'] == NTreatObsT0['qtile'][np.abs(NTreatObsT0['qtile']-_temp).idxmin()])
    _temp2 = _temp2[~np.isnan(_temp2)]
    _temp2 = _temp2.iloc[0]
    ### Use the estimated income from step 2 to find the counterfactual cdf for the treatment group at t=1 ###
    _temp3 = TreatObsT0['qtile'].where(TreatObsT0['Income'] == TreatObsT0['Income'][np.abs(TreatObsT0['Income']-_temp2).idxmin()])
    _temp3 = _temp3[~np.isnan(_temp3)]
    _temp3 = _temp3.iloc[0]
    ### Track the different values ###
    IncHist.append(_i)
    QuantInc01Hist.append(_temp)
    Inc00QuantInc01Hist.append(_temp2)
    QuantInc00QuantInc01Hist.append(_temp3)

### Plot the counterfactual distribution vs. the actual distribution ###
if _graph == True:
    plt.rcParams["figure.figsize"] = [_graphszx, _graphszy]
    plt.plot(IncHist, QuantInc00QuantInc01Hist, '--k', TreatObsT1['Income'], TreatObsT1['qtile'])
    plt.xlabel('Income')
    plt.ylabel('F(Income)')
    plt.title('True CDF & Counterfactual CDF (No treatment) for Treatment Group (treatment size: '+str(treateffect) + ')')
    plt.legend(['Counterfactual CDF', 'Empirical CDF'])
    plt.grid(True)

### Estimate average and quantile treatment effects ###
khist = []

### Average treatment effect (See p.441 (16) ###
for _i in np.nditer(TreatObsT0['Income']):
    ### Find the non-treated t=0 quantile for each income level in the treated group at t=0 ###
    _temp4 = NTreatObsT0['qtile'].where(NTreatObsT0['Income']==NTreatObsT0['Income'][np.abs(NTreatObsT0['Income']-_i).idxmin()])
    ### Remove NaN values ###
    _temp4 = _temp4[~np.isnan(_temp4)]
    ### Keep only one value in case of duplicates ###
    _temp4 = _temp4.iloc[0]
    ### Find the income level of non-treated at t=0 that corresponds to the quantile in part 1 ###
    _temp5 = NTreatObsT1['Income'].where(NTreatObsT1['qtile'] == NTreatObsT1['qtile'][np.abs(NTreatObsT1['qtile']-_temp4).idxmin()])
    _temp5 = _temp5[~np.isnan(_temp5)]
    _temp5 = _temp5.iloc[0]
    ### Track the different values ###
    khist.append(_temp5)

### Calculate the Average Effect ###
TauCiC = np.mean(TreatObsT1['Income'])-np.mean(khist)

### Estimate the quantile effect (See p.443 (18)) ###
qtls = []
qtlsCF = []
for _i in range(1, noquantiles):
    ### Find the income level for treated at t=0 for each quantile ###
    _temp6 = TreatObsT0['Income'].where(TreatObsT0['qtile']==TreatObsT0['qtile'][np.abs(TreatObsT0['qtile']-_i/noquantiles).idxmin()])
    ### Remove NaN values ###
    _temp6 = _temp6[~np.isnan(_temp6)]
    ### Keep only one value in case of duplicates ###
    _temp6 = _temp6.iloc[0]
    ### Using the income from step 1, find the quantile in the non-treated group at t=0 with income closest to the previously estimated income ###
    _temp7 = NTreatObsT0['qtile'].where(NTreatObsT0['Income'] == NTreatObsT0['Income'][np.abs(NTreatObsT0['Income']-_temp6).idxmin()])
    _temp7 = _temp7[~np.isnan(_temp7)]
    _temp7 = _temp7.iloc[0]
    ### Use the estimated quantile from step 2 to find the counterfactual income for the non-treatment group at t=1 ###
    _temp8 = NTreatObsT1['Income'].where(NTreatObsT1['qtile'] == NTreatObsT1['qtile'][np.abs(NTreatObsT1['qtile']-_temp7).idxmin()])
    _temp8 = _temp8[~np.isnan(_temp8)]
    _temp8 = _temp8.iloc[0]
    ### Track the different values ###
    qtls.append(_i/10)
    qtlsCF.append(_temp8)

TauCiCqt = []
for _i in range(1, noquantiles):
    _temp9 = TreatObsT1['Income'].where(TreatObsT1['qtile']==TreatObsT1['qtile'][np.abs(TreatObsT1['qtile']-_i/noquantiles).idxmin()])
    _temp9 = _temp9[~np.isnan(_temp9)]
    _temp9 = _temp9.iloc[0]
    _temp10 = _temp9 - qtlsCF[_i-1]
    TauCiCqt.append(_temp10)

### Print final time + results ###
_t1 = time.time()
_tdel = _t1-_t0
print('\n We are done! \n It took us: '+ str(round(_tdel, _timedecimals)) +' seconds \n' + ' The average treatment effect is: ' + str(round(TauCiC, _decimals)))
for _i in range(0, noquantiles-1):
    print(' The ' +str((_i+1)/noquantiles)+' quantile CiC effect is: ' + str(round(TauCiCqt[_i], _decimals)))
