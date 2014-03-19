import csv
import os
import datetime
import numpy as np
import pandas as p
from pandas import Series
from SecretSauceRandomForest import SecretSauceRandomForest

basePath = '/user/saif/Documents/uwdmc2014_files/'

def genTimeSeries(frequency = '3M',loadValidation = False):
    if loadValidation:
        path = basePath + 'Validation Customer Data/'
    else:
        path = basePath + 'Customer Data/'
    customers = [f[:-4] for f in os.listdir(path) if f[0] != '.']
    allDates = p.date_range(datetime.datetime(2010,6,1),datetime.datetime(2011,7,1),freq=frequency)  #rrule(MONTHLY,dtstart=datetime.datetime(2010,6,1),until=datetime.datetime(2011,10,1)) 
    timeSeries = {cust:0 for cust in customers}
    for f in os.listdir(path):
        if f[0] != '.':
            cust = f[:-4]
            timeSeries[cust] = Series(np.zeros(len(allDates)),index=allDates)
            with open(path + f) as fin:
                reader = csv.reader(fin)
                reader.next()
                for row in reader:
                    if row[1] == 'M216':
                        trxnDate = datetime.datetime.strptime(row[2].split()[0],'%Y-%m-%d')
                        cutoff = min([d for d in allDates if d >= trxnDate])
                        timeSeries[cust][cutoff] += 1
    return timeSeries

def genFeatures():
    # Read in returning/nonreturning customers
    with open(basePath + 'returningCustomers.csv') as fin:
        retCust = [row[0] for row in csv.reader(fin)]
    
    with open(basePath + 'nonreturningCustomers.csv') as fin:
        nonretCust = [row[0] for row in csv.reader(fin)]
    
    # Read in merchant data
    with open(basePath + 'merchant_metrics.csv') as filein: 
        reader = csv.reader(filein) 
        titles = reader.next()
        merchData = {row[12]:{titles[i]:row[i] for i in xrange(12)} for row in reader if row[0] != 'total_trxn_amount'}
     
    featInds = np.unique([merchData[m]['Industry_Name'] for m in merchData.keys()])
    
    # Generate list of zip codes where people have shopped at M216
    zipList = []
    for f in os.listdir(basePath + 'Customer Data/'):
        if f[0] != '.':
            with open(basePath + 'Customer Data/' + f) as fin:
                reader = csv.reader(fin)
                for row in reader: 
                    if row[1] == 'M216' and row[4] == '0.0':
                        zipList += [row[3]]
    zipList = np.unique(zipList)
    
    # Read in customer zip codes
    with open(basePath + 'customer_zip.csv') as fin:
        reader = csv.reader(fin)
        reader.next()
        custZip = {(row[0],row[1]):datetime.datetime.strptime(row[2],"%d%b%Y") for row in reader}
    print("Starting training features!")
    trainingFeatures = buildFeatures(retCust + nonretCust,merchData,featInds,zipList,custZip,loadValid = False)
    print("Done training features!")
    validationCustomers = [f[:-4] for f in os.listdir(basePath + 'Validation Customer Data/') if f[0] != '.']
    validationFeatures = buildFeatures(validationCustomers,merchData,featInds,zipList,custZip,loadValid = True)
    
    
    return trainingFeatures, validationFeatures, retCust, nonretCust

def buildFeatures(customers,merchData,featInds,zipList,custZip,loadValid = False):
    featureNames = ['total value of transactions','total number of transactions','number of internet transactions','total value of internet transactions','number of transactions at competitors','value of transactions at competitors','time since last transaction at M216','number of transactions at M216','M216 total value','zip code shared with M216','number of transactions in same zip as M216'] \
     + [a + ' total value' for a in featInds] + ['number of transactions at ' + a for a in featInds] 
    features = {cust:{ind:0 for ind in featureNames} for cust in customers} # Add time dependent random forest/neural network!
    
    if loadValid:
        path = basePath + 'Validation Customer Data/'
    else:
        path = basePath + 'Customer Data/'
        
    for f in os.listdir(path):
        if f[0] != '.':
            cust = f[:-4]
            unlabelledTrxns = 0
            valueOfUnlabelledTrxns = 0
            
            with open(path + f) as fin:
                reader = csv.reader(fin)
                reader.next()
                lastTransaction = datetime.datetime.strptime("2010-07-01","%Y-%m-%d")
                for row in reader:
                    if row[1] != '' and row[1] != 'N/A' and row[1] in merchData.keys()  :
                        ind =  merchData[row[1]]['Industry_Name']
                    else:
                        unlabelledTrxns += 1
                        valueOfUnlabelledTrxns += float(row[0])
                        ind = ''
                    # Get number, total value of transactions by industry
                    if ind in featInds:
                        features[cust][ind + ' total value'] += float(row[0])
                        features[cust]['number of transactions at ' + ind] += 1
                    if row[1] == 'M216':
                        features[cust]['number of transactions at M216'] += 1
                        features[cust]['M216 total value'] += float(row[0])
                        lastTransaction = min(lastTransaction,datetime.datetime.strptime(row[2].split()[0],"%Y-%m-%d"))
    #                 if row[1] in competitors:
    #                     features[cust]['number of transactions at competitors'] += 1
    #                     features[cust]['value of transactions at competitors'] += float(row[0])
                    if row[3] in zipList:
                        features[cust]['number of transactions in same zip as M216'] += 1
                    if row[4] == '1.0':
                        features[cust]['number of internet transactions'] += 1
                        features[cust]['total value of internet transactions'] += float(row[0])
                    features[cust]['total number of transactions'] += 1
                    features[cust]['total value of transactions'] += float(row[0])
            mostRecentZip = [key[1] for key,value in custZip.iteritems() if key[0] == cust
                              and value == max([value for key,value in custZip.iteritems() if key[0] == cust])][0]
            features[cust]['zip code shared with M216'] = mostRecentZip in zipList
            features[cust]['time since last transaction at M216'] = (datetime.datetime(2011,6,30) - lastTransaction).days
            for ind in featInds:
                if unlabelledTrxns != features[cust]['total number of transactions']:
                    features[cust][ind + ' total value'] /= features[cust]['total value of transactions'] - valueOfUnlabelledTrxns
                    features[cust]['number of transactions at ' + ind] /= features[cust]['total number of transactions'] - unlabelledTrxns
            features[cust]['number of internet transactions'] /= features[cust]['total number of transactions']
            features[cust]['total value of internet transactions'] /= features[cust]['total value of transactions']
            features[cust]['number of transactions in same zip as M216'] /= features[cust]['total number of transactions']
    
    timeSeries = genTimeSeries(loadValidation = loadValid)
    for cust in timeSeries.keys():
        for d in timeSeries[cust].keys():
            features[cust][d] = timeSeries[cust][d]
    return(features)

def printResultsToCsv(validationFeatures,features,retCust,nonretCust):
    ssrf = SecretSauceRandomForest(n_estimators=100)
    X = np.array([features[cust].values() for cust in retCust + nonretCust])
    y = np.array([1 for cust in retCust] + [0 for cust in nonretCust])
    ssrf.fit(X,y)
    with open(basePath + 'finalResults.csv','wb') as fout:
        writer = csv.writer(fout)
        writer.writerow(['ACCT_CODE_ID','OFFER'])
        for f in os.listdir(basePath + 'Validation Customer Data/'):
            if f[0] != '.':
                cust = f[:-4]
                phat = ssrf.predict_proba(validationFeatures[cust].values())
                if phat[0][1] > 0.25:
                    writer.writerow([cust,1])
                else:
                    writer.writerow([cust,0])

if __name__ == '__main__':
    trainingFeatures, validationFeatures, retCust, nonretCust = genFeatures()
    print("Printing to CSV!")
    printResultsToCsv(validationFeatures,trainingFeatures,retCust,nonretCust)