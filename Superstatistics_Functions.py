#!/usr/bin/env python
# coding: utf-8

# # Define Superstatistical functions to carry out analysis

# In[1]:
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kurtosis
from scipy.integrate import odeint
import statistics
from scipy.stats import lognorm
import math
import string
import sdeint
from joblib import Parallel, delayed 
import multiprocessing
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns
import pandas as pd
from datetime import datetime
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from scipy.stats import ttest_ind
from datetime import timedelta
import scipy.signal
import scipy.stats as st
from scipy.special import gamma, digamma
from scipy.optimize import minimize

#define custom pdf for the q-Gaussian distribution
def q_Gauss_pdf(x,q,l,mu):
    constant=np.sqrt(np.pi)*gamma((3-q)/(2*(q-1)))/(np.sqrt(q-1)*gamma(1/(q-1)))
    pdf=np.sqrt(l)/constant*(1+(1-q)*(-l*(x-mu)**2))**(1/(1-q))
    return pdf
class q_Gauss_custom(st.rv_continuous):
    def _pdf(self, x, q, l):
        "Custom q-Gauss distribution"
        #q=self.q
        #l=self.l
        mu=0
        pdf =q_Gauss_pdf(x,q,l,mu)
        return pdf
    def _stats(self, q, l):
        return [self.q,self.l,0,0]
    #fitstart provides a starting point for any MLE fit
    def _fitstart(self,data):
        return (1.1,1.1)
    def _argcheck(self, q,l):
        #define an arbitrary number of conditions on the arguments, such as psotivitiy or a certain range
        largeQ = q > 1
        positiveScale=l>0
        all_bool = largeQ&positiveScale
        return all_bool
qGauss_custom_inst = q_Gauss_custom(name='qGauss_custom',a=0)

#define custom pdf for the q-exponential distribution
def q_Exp_pdf(x,q,l,mu):
    #pdf=(2-q)*l*(1-l*(1-q)*(x-mu))**(1/(1-q))
    pdf=(2 - q) * l * np.sign(1 + (q- 1) * l * (x-mu))* (np.abs(1 + (q- 1) * l * (x-mu)))**( 1/(1 - q))
    return pdf
class q_Exp_custom(st.rv_continuous):
    def _pdf(self, x, q, l):
        "Custom q-Exp distribution"
        #q=self.q
        #l=self.l
        mu=0
        pdf =q_Exp_pdf(x,q,l,mu)
        return pdf
    def _stats(self, q, l):
        return [self.q,self.l,0,0]
    #fitstart provides a starting point for any MLE fit
    def _fitstart(self,data):
        return (1.1,1.1)
    def _argcheck(self, q,l):
        #define an arbitrary number of conditions on the arguments, such as psotivitiy or a certain range
        largeQ = q > 1
        positiveScale=l>0
        all_bool = largeQ&positiveScale
        return all_bool
qExp_custom_inst = q_Exp_custom(name='qExp_custom',a=0)

def plot_fluctuation_histo(data,targetKurtosis,xlabel,exportName):
    plot=sns.distplot(data)
    #extract distplot range
    (xvalues_hist,yvalues_hist)=plot.get_lines()[0].get_data()
    if targetKurtosis==3:
        mu=np.mean(data)
        qParameters=qGauss_custom_inst.fit(data-mu,1.2,1, floc=0, fscale=1)
        fitValues=q_Gauss_pdf(xvalues_hist,qParameters[0],qParameters[1],mu)
        plt.legend(['q-Gaussian','Data'])
    if targetKurtosis==9:
        mu=min(data)
        qParameters=qExp_custom_inst.fit(data-mu,1.2,0.001, floc=0, fscale=1)
        fitValues=q_Exp_pdf(xvalues_hist,qParameters[0],qParameters[1],mu)
        plt.legend(['q-Exp','Data'])
    plt.plot(xvalues_hist,fitValues)
    if targetKurtosis==3:
        plt.legend(['q-Gaussian','Data'])
    if targetKurtosis==9:
        plt.legend(['q-Exp','Data'])
    plt.yscale('log')
    plt.ylim(0.1*min(fitValues),10*max(fitValues))
    plt.xlabel(xlabel)
    plt.ylabel('PDF')
    plt.title('q='+str(round(qParameters[0],3)))
    plt.savefig(exportName+'_Hist.pdf')
    plt.show()

#Superstatistics functions

#Determine average kurtosis
def averageKappa(data,DeltaT):
    #make sure that negative calls return still a number
    if DeltaT<1:
        return 0
    meanData=np.mean(data);
    tMax=len(data);
    nominator=sum((data[0:DeltaT]-meanData)**4)
    denominator=sum((data[0:DeltaT]-meanData)**2)
    sumOfFractions=nominator/(denominator**2);

    for i in range(DeltaT,tMax):
        nominator=nominator+(data[i]-meanData)**4-(data[i-DeltaT]-meanData)**4;
        denominator=denominator+(data[i]-meanData)**2-(data[i-DeltaT]-meanData)**2;
        sumOfFractions = sumOfFractions + nominator/(denominator**2);
    return sumOfFractions/(tMax-DeltaT)*DeltaT

#define a membership function to avoid including indices that have been removed as NaN
def testMembership(item,list):
    if any(item == c for c in list):
        return True
    else:
        return False

def betaList(data,T):
    uSquareMean=sum(data[0:T]**2)/T;
    uMean=sum(data[0:T])/T;
    betaValues=[1/(uSquareMean-uMean**2)]
    tMax=len(data)
    for i in range(T+1,tMax):
        uSquareMean=uSquareMean+(data[i]**2 - data[i-T]**2)/T  
        uMean=uMean+(data[i]-data[i-T])/T;
        betaValues.append(1/(uSquareMean-uMean**2))
    return betaValues

#compute and plot the average kurtosis as a function of time
def plotLongTimeScale(data,startTime,EndTime,TimeStep,targetKurtosis,xlabel,exportName, timeUnit,timeUnitName):
    kurtosisList=[]
    timeList=range(startTime*timeUnit, EndTime*timeUnit, TimeStep*timeUnit)
    plotTimeList=range(startTime,EndTime,TimeStep)
    for time in timeList:
        kurtosisList.append(averageKappa(data,time))
    plt.plot(plotTimeList,kurtosisList, linewidth=4.0)
    plt.plot(plotTimeList, targetKurtosis*np.ones(len(kurtosisList)),linewidth=4.0)
    plt.xlabel('Time lag $\Delta t$ ['+timeUnitName+']')
    plt.ylabel('Average kurtosis $\overline{\kappa}$')
    #plt.title(stationName)
    if targetKurtosis==3:
        plt.legend([exportName,'Gaussian'])
    else:
        if targetKurtosis==9:
            plt.legend([exportName,'Exponential'])
        else:
            plt.legend([exportName,'Target $\kappa$'])
    plt.savefig('Long_time_scales_'+exportName+'.pdf')
    plt.show()

#plot a low and high-variance snapshot
def plotExtremeSnapshots(data,longTimeScale,xlabel,exportName):
    longTimeScaleApplied=round(longTimeScale)
    #compute the variance of the different time windows
    varianceList=[]
    for tIndex in range(0, int(len(data)/longTimeScaleApplied)):
        varianceList.append(np.std(data[tIndex*longTimeScaleApplied:(tIndex+1)*longTimeScaleApplied]))
    minposition=[i for i,x in enumerate(varianceList) if x == min(varianceList)]
    maxposition=[i for i,x in enumerate(varianceList) if x == max(varianceList)]
    #display two different histograms using only data within one "homogeneous" time window
    startingIndex1=minposition[0]
    startingIndex2=maxposition[0]
    #set up subplots
    f, (ax1, ax2) = plt.subplots(1,2,figsize=(12,5))
    #plot log-scale plot
    sns.distplot(data[startingIndex1*longTimeScaleApplied:(startingIndex1+1)*longTimeScaleApplied],ax=ax1)
    ax1.set_yscale('log')
    #ax1.hist(data[startingIndex1*longTimeScaleApplied:(startingIndex1+1)*longTimeScaleApplied], density=True,log=True,bins=10)
    ax1.set_ylabel('PDF');
    ax1.set_xlabel(xlabel);
    ax1.set_title('Low variance')
    ax1.text(0.05, 0.9, string.ascii_lowercase[0], transform=ax1.transAxes, 
                size=20, weight='bold')
    #plot linear-scale plot
    #ax2.hist(data[startingIndex2*longTimeScaleApplied:(startingIndex2+1)*longTimeScaleApplied], density=True,log=True,bins=10)
    sns.distplot(data[startingIndex2*longTimeScaleApplied:(startingIndex2+1)*longTimeScaleApplied],ax=ax2)
    ax2.set_yscale('log')
    ax2.set_ylabel('PDF');
    ax2.set_xlabel(xlabel);
    ax2.set_title('High variance')
    ax2.text(0.05, 0.9, string.ascii_lowercase[1], transform=ax2.transAxes, 
                size=20, weight='bold')
    #combine and show both plots
    f.subplots_adjust(wspace =0.2)
    plt.savefig('ExtremeSnapshots_'+exportName+'.pdf')
    plt.show()

#function that returns the next guess, assuming the kurtosis is a linear function
def nextGuessingTime(x1,y1,x2,y2,targetKurtosis):
    slope=(y2-y1)/(x2-x1)
    intercept=y1-slope*x1
    predictedTime=(targetKurtosis-intercept)/slope
    return int(predictedTime)

# Function to systemnatically determine the long time scale, given an initial value and a certain tolerance
def determineLongTimeScale(data,initialTimeGuess,kurtosisTolerance, targetKurtosis,initalGuessIncrement=1,method='Manual Newton'):
    if method=='Manual Newton':
        kurtosisList=[]
        timeList=[]
        #we simplify the taks by assuming that the long time scale is approxiamtely linear with delta t
        #inital 2 runs:
        averageKurtosis=averageKappa(data,initialTimeGuess)
        kurtosisList.append(averageKurtosis)
        timeList.append(initialTimeGuess)
        if averageKurtosis>targetKurtosis:
            newTime=initialTimeGuess-initalGuessIncrement
        else:
            newTime=initialTimeGuess+initalGuessIncrement
        timeList.append(newTime)
        averageKurtosis=averageKappa(data,newTime)
        kurtosisList.append(averageKurtosis)
        #initiate a counter to prevent endless loops
        loopCounter=0
        while abs(averageKurtosis-targetKurtosis)>kurtosisTolerance:
            #only repeat the loop 100 times
            if loopCounter==100:
                return 'No convergence'
            newTime=nextGuessingTime(timeList[-1],kurtosisList[-1],timeList[-2],kurtosisList[-2],targetKurtosis)
            if newTime<0:
                return 'No convergence'
            timeList.append(newTime)
            averageKurtosis=averageKappa(data,newTime)
            #abort if average kurtosis becomes negative
            if averageKurtosis<0:
                return 'No convergence'
            kurtosisList.append(averageKurtosis)
            loopCounter=loopCounter+1
        return (timeList[-1],'Converged to kurtosis='+str(averageKurtosis),'Iterations needed='+str(loopCounter))
    if method=='Nelder-Mead':
        #we define a function of the average kurtosis, purely as a function of teh long time scale
        def averageKurtosis_func(deltaT):
            #round the applied long time scale to an integer
            appliedDeltaT=int(deltaT)
            return abs(averageKappa(data,appliedDeltaT)-targetKurtosis)
        #next, we employ the nelder mead optimizer to find the optimal long time scale 
        result=minimize(averageKurtosis_func, initialTimeGuess, method='Nelder-Mead', tol=kurtosisTolerance)#,bounds=bnds
        return (result['x'][0],'Converged to kurtosis='+str(targetKurtosis+result['fun']),'Iterations needed='+str(result['nfev']))

def fit_and_plot_betaDist(data,longTimeScale,xlabel,exportName,timeUnit,timeUnitName):
    betaDis=betaList(data,round(longTimeScale))
    meanBeta=np.mean(betaDis)
    meanB=meanBeta
    #define custom pdf, using the mean of beta for inverse chi^2 distributions but leaving it as a fitting parameter for chi^2 distributions (leads to stabler fits)
    #define the chi-square custom pdf, note that we also fit the mean beta value to allow the algorithm to converge
    #define custom pdf
    class chiSquare_custom(st.rv_continuous):
        def _pdf(self, x, degreesN, meanB):
            "Custom Chi-square distribution"
            pdf =1/(gamma(degreesN/2))*(degreesN/(2*meanB))**(degreesN/2)*x**(degreesN/2-1)*np.exp(-(degreesN*x)/(2*meanB))
            return pdf
        def _stats(self, degreesN, meanB):
            return [self.degreesN,self.meanB,0,0]
        #fitstart provides a starting point for any MLE fit
        def _fitstart(self,data):
            return (1.1,1.1)
        def _argcheck(self, degreesN,meanB):
            #define an arbitrary number of conditions on the arguments, such as psotivitiy or a certain range
            positiveDegrees = degreesN > 0
            positiveMean=meanB>0
            all_bool = positiveDegrees&positiveMean
            return all_bool

    class invChiSquare_custom(st.rv_continuous):
        def _pdf(self, x, degreesN):
            "Custom inverse Chi-square distribution"
            pdf =1/(gamma(degreesN/2))*meanB*(degreesN*meanB/2)**(degreesN/2)*x**(-degreesN/2-2)*np.exp(-(degreesN*meanB)/(2*x))
            return pdf
        def _stats(self, degreesN):
            return [self.degreesN,0,0]
        #fitstart provides a starting point for any MLE fit
        def _fitstart(self,data):
            return (1.1,1.1)
        def _argcheck(self, degreesN):
            #define an arbitrary number of conditions on the arguments, such as psotivitiy or a certain range
            positiveDegrees = degreesN > 0
            all_bool = positiveDegrees
            return all_bool
    #initiate distribution instances with support starting at 0
    chiSquare_custom_inst = chiSquare_custom(name='chiSquare_custom',a=0)
    invchiSquare_custom_inst = invChiSquare_custom(name='invchiSquare_custom',a=0)

    logNormalFit=stats.lognorm.fit(betaDis, scale=np.exp(5),loc=0)
    chiSquareFit=chiSquare_custom_inst.fit(betaDis,3,meanBeta,floc=0,fscale=1)
    degrees_chi=chiSquareFit[0]
    inv_chiSquareFit=invchiSquare_custom_inst.fit(betaDis,1,floc=0,fscale=1)
    degrees_inv_chi=inv_chiSquareFit[0]
    #Compare the fits with the data:
    xrange=np.arange(0,max(betaDis),max(betaDis)/100)
    pdfVlaues_chi=chiSquare_custom_inst.pdf(xrange,degreesN=degrees_chi,meanB=meanBeta, loc=0, scale=1)
    pdfVlaues_inv_chi=invchiSquare_custom_inst.pdf(xrange,degreesN=degrees_inv_chi, loc=0, scale=1)
    pdfVlaues_logNorm=stats.lognorm.pdf(xrange,*logNormalFit)
    plot=sns.distplot(betaDis)
    #plt.hist(betaDis, density=True)
    plt.plot(xrange,pdfVlaues_chi, linewidth=4.0)
    plt.plot(xrange,pdfVlaues_inv_chi, linewidth=4.0)
    plt.plot(xrange,pdfVlaues_logNorm, linewidth=4.0)
    #plt.yscale('log')
    
    displayedTimeScale=longTimeScale/timeUnit
    plt.title('T='+str(round(displayedTimeScale))+timeUnitName+', $n_{\chi^2}$= '+str(round(degrees_chi,3))+ ', $n_{inv. \chi^2}$= '+str(round(degrees_inv_chi,3)))
    plt.xlabel(r'$ \beta $ ('+exportName+')')
    plt.ylabel("PDF")
    #extract distplot range
    (xvalues_hist,yvalues_hist)=plot.get_lines()[0].get_data()
    plt.ylim(min(yvalues_hist),max(yvalues_hist)*1.1)
    plt.xlim(0)
    plt.legend(['$\chi^2$','inv. $\chi^2$','log-norm.',r'$\beta$ values'])
    plt.savefig('BetaDistribution_'+exportName+'.pdf')
    plt.show()


# In[ ]:




