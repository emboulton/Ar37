import sys
import os
import h5py as h5
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats.distributions import t
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from sympy.solvers import solve
from sympy.abc import x, y
from sympy import Symbol, Function, Eq, N
from sympy.functions import log
from scipy.optimize import fsolve, broyden1, minimize
from scipy.misc import factorial
import math as math
from scipy import stats
import scipy.interpolate as ter
import scipy.integrate as integrate
from scipy.signal import savgol_filter
#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

sys.path.insert(2, '/Users/mac/Documents/McKinsey_Research/Ar37')
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'
import aLib

#Set font and other things for pretty/uniform plots
#mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman'] #for paper
#mpl.rcParams['font.serif'] = ['Cambria'] #for presentation

#LOADING DATA SETS
Ar1 = 'data_skimmedFiles/Ar37_100mVns_C0.5_A2.5_newFilter_botT23mV_sumVeto1000mV_150407-1553_v3a_lowS2_proc.mat'
Ar2 = 'data_skimmedFiles/Ar37_100mVns_C0.5_A2.5_newFilter_botT23mV_sumVeto1000mV_150407-1553_v3a_lowS2_proc.mat'
Ar3 = 'data_skimmedFiles/Ar37_100mVns_C0.25_A2.5_newFilter_botT23mV_sumVeto1000mV_150429-1136_v3a_lowS2_proc.mat'
Ar4 = 'data_skimmedFiles/Ar37_100mVns_C1_A2.5_newFilter_botT10mV_sumVeto1000mV_150429-0907_v3a_lowS2_proc.mat'
Ar5 = 'data_skimmedFiles/Ar37_100mVns_C1_A2.5_newFilter_botT15mV_sumVeto1000mV_150428-1148_v3a_lowS2_proc.mat'
Ar6 = 'data_skimmedFiles/Ar37_100mVns_C1_A2.5_newFilter_botT23mV_sumVeto1000mV_150406-1512_v3a_lowS2_proc.mat'
Ar7 = 'data_skimmedFiles/Ar37_100mVns_C1_A2.5_newFilter_botT15mV_sumVeto1000mV_150428-1148_v3a_lowS2_proc.mat'
Ar8 = 'data_skimmedFiles/Ar37_100mVns_C2_A2.5_newFilter_botT23mV_sumVeto1000mV_150408-0802_v3a_lowS2_proc.mat'
Ar9 = 'data_skimmedFiles/Ar37_100mVns_C2_A2.5_newFilter_botT23mV_sumVeto1000mV_150408-0940_v3a_lowS2_proc.mat'
Ar10 = 'data_skimmedFiles/Ar37_100mVns_C3.5_A2.5_newFilter_botT23mV_sumVeto1000mV_150408-1538_v3a_lowS2_proc.mat'
Ar11 = 'data_skimmedFiles/Ar37_100mVns_C5_A2.5_newFilter_botT23mV_sumVeto1000mV_150409-1117_v3a_lowS2_proc.mat'
Ar12 = 'data_skimmedFiles/Ar37_100mVns_C10_A2.5_newFilter_botT23mV_sumVeto1000mV_150409-2041_v3a_lowS2_proc.mat'

data=[Ar1, Ar6, Ar9, Ar10, Ar11, Ar12]

field=np.array([99,198,396,693,990,1980])/1000
field_approx = [100 , 200 , 400 , 700 , 1000 , 2000]
SE1 = np.array([31.8888, 31.4924, 31.2577, 31.5749, 30.7863, 31.0716])*100/85
SE2 = np.array([31.8985, 31.7162, 31.2832, 31.0180, 30.9145, 31.1543])*100/85
SE3 = np.array([31.5167, 31.0791, 31.2970, 30.8198, 31.0224, 30.9267])*100/85
SE4 = np.array([31.7967, 31.2354, 31.0878, 31.2203, 30.9274, 30.9801])*100/85
SE5 = np.array([31.5737, 31.2244, 31.4427, 31.0895, 31.0435, 31.4262])*100/85

#CREATING SE_avg LIST AND SE_std LIST
SE_avg=(SE1+SE2+SE3+SE4+SE5)/5

arrays=[SE1,SE2,SE3,SE4,SE5]

i=0
SE_std=np.zeros(len(field))
while i<len(field):
	j=0
	x = np.array([])
	while j<5:
		x = np.append(x,arrays[j][i])
		j=j+1
	SE_std[i]=np.std(x)
	i=i+1	

index=0
SE_std=np.array([])
while index<6:
	x=np.array([])
	for array in arrays:
		x=np.append(x,array[index])
	std=np.std(x)
	SE_std=np.append(SE_std,std)
	index=index+1

w_fnc=13.7 #work function for Xe
EE=.78 #extraction efficiency
EE_err=0.05 #extraction efficiency *100= %error
g1 = 0.097 #s1 light collection efficiency
g1_err=.07 #*100= % error
g2=SE_avg*EE;
g2_err=np.sqrt((SE_std/SE_avg)**2+(EE_err)**2) #percent error


#FUNCTIONS TO NAVIGATE HDF5 FILES
def printname(name):
	print (name)    
#prints names of keys or groups in file#
def keys(data):
	return [key for key in data.keys()]
#prints values or subgroups associated with each key as an object#
def values(data):
	return [value for value in data.values()]
def sets(data):
	return [sets for sets in data.items()]

#OTHER MISC. FUNCTIONS
def compound_and(lst,length):
	total=(len(lst)-1)
	index=0
	already_checked=np.ones(length)
	while index<=total:
		condition= lst[index]
		already_checked=np.logical_and(condition,already_checked)
		index=index+1
	return(already_checked)
def gaussian(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))
def gaussian_yoff(x,a,x0,sigma,b):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))+b
def TIB_fit(x,A,delta):
	return A*(np.array(x)**(-delta))
def bin_to_x(bin):
	length= len(bin)
	value=0
	x=np.array([])
	while value<(length-1):
		new_x=(bin[value]+bin[value+1])/2
		x=np.append(x,new_x)
		value=value+1    
	return x
def x_to_bin(x):
	length= len(x)
	value=0
	bin=np.array([])
	bin1=x[0]-(x[1]-x[0])/2
	if bin1 < .00001:
		bin1=0
	bin=np.append(bin,bin1)
	while value<(length):
		new_bin=2*x[value]-bin[value]
		bin=np.append(bin,new_bin)
		value=value+1    
	return bin
#def poisson(k,lamb):
#	return (lamb**(k)/factorial(k))*np.exp(-lamb)
def poisson(bins,lamb):
	k=np.arange(len(bins))
	p0=(lamb**(k)/factorial(k))*np.exp(-lamb)
	binse=bins+(bins[1]-bins[0])/2
	f=ter.interp1d(k,p0,'quadratic')
	p=f(binse)
	return p
def logistic(x,L,k1,x0):
	return L/(1+np.exp(-k1*(x-x0)))
def gauss_AreaOfPoisson(x,sigma,lamb): #x[0]->x, x[1]->k
	x1=np.array(x[1])
	x1=x1.reshape(len(x1),1)
	z=np.ones((len(x[1]),len(x[0])))*x1
	return (lamb**z/(factorial(z)*sigma*np.sqrt(2*np.pi)))*np.exp(-lamb-((x[0]-z)**2)/2*sigma**2)
#def gauss_AreaOfPoisson(x,k,sigma,lamb): 
#	return (lamb**k/(factorial(k)*sigma*np.sqrt(2*np.pi)))*np.exp(-lamb-((x-k)**2)/2*sigma**2)
def GaussSum(x,*p):
	if len(p) == 1:
		p=p[0]
	n=int(len(p)/3)
	A=p[:n]
	simga=p[n:2*n]
	x0=p[2*n:3*n]
	y=sum([A[i]*np.exp(-1*(x-x0[i])**2/(2*(simga[i]**2))) for i in range(n)])
	return y
def integer_from_list(x_values):
	return x_values[np.mod(x_values,1)==0]
def SmearedPoisson(x_values,scl,sigma,lamb): # scl=fraction of histogram area included in fit call
	if lamb<15:
		k_values=np.arange(3*lamb)
	else:
		k_values=np.arange(2*lamb)
	A_values=lamb**k_values*np.exp(-lamb)/(np.sqrt(2*np.pi)*factorial(k_values)*sigma)
	sigma_values=np.ones(len(k_values))*sigma
	p=np.append(np.append(A_values,sigma_values),k_values)
	y=GaussSum(x_values,p)
	db=x_values[1]-x_values[0]
	sumy=0
	for i in np.arange(len(y)):
		sumy=sumy+y[i]*db
	scl_fctr=scl/sumy
	return scl_fctr*y
def SmearedPoisson_Unit(x_values,sigma,lamb): # CAN ONLY USE IF SCL=1
	if lamb<15:
		k_values=np.arange(3*lamb)
	else:
		k_values=np.arange(2*lamb)
	A_values=lamb**k_values*np.exp(-lamb)/(np.sqrt(2*np.pi)*factorial(k_values)*sigma)
	sigma_values=np.ones(len(k_values))*sigma
	p=np.append(np.append(A_values,sigma_values),k_values)
	return GaussSum(x_values,p)
def SmearedPoisson_NewRange(x_values, new_scl, scl, sigma, lamb):
	#new_scl = fraction of histogram area included in fit call
	#scl=scl from when the SP was originally calculated, not used - it's included anyway so that you can call SmearedPoisson_NewRange(x_values, new_scl, *popt)
	return SmearedPoisson(x_values, new_scl, sigma, lamb)
##OLD SMEAREDPOISSON DEFINITIONS
#def SmearedPoisson(x_values,sigma,lamb):
#	k_values=integer_from_list(x_values)
#	z=[x_values,k_values]
#	return sum(gauss_AreaOfPoisson(z,sigma,lamb))
#def SmearedPoisson(x_values,sigma,lamb):
#	k_values=integer_from_list(x_values)
#	sum_gauss=np.zeros(len(x_values))
#	for k in k_values:
#		sum_gauss=sum_gauss+gauss_AreaOfPoisson(x_values,k,sigma,lamb)
#	return sum_gauss
#def IntVarVal(x,k,sigma,lamb):
	#ExpVal=lamb/(sigma*np.sqrt(2*np.pi))
#	return (x)**2*np.exp(-(x-k)**2/(2*sigma**2))
def IntVarVal(x,sigma,lamb):
	return x**2*SmearedPoisson_Unit(x,sigma,lamb)
def GenExp(x,a,b,c):
	return a*b**x+c

#sigmatemp=2.68
#lambtemp=8.41
#ktemp=0
#plt.figure(0)
#plt.clf()
#xt=np.arange(-10,50,.1)
#plt.plot(xt,(xt-1.25)**2,color='r')
#plt.plot(xt,np.exp(-(xt-ktemp)**2)/(2*sigmatemp),color='g')
#plt.plot(xt,IntVarVal(xt,ktemp,sigmatemp,lambtemp),color='b')
#plt.xlim([-10,50])
#plt.ylim([0,30])
#plt.show()



#set up variables for the for-loop
p=1
i=0
j=0


#FIGURE1: show s1 and s2 spectrums for three fields
#dataF1=[Ar1, Ar10, Ar12] #100, 700, 2000 V
dataF1=[Ar1, Ar9, Ar12] #100, 400, 2000 V
colors_hist=['#ed665d','#729ece','#67bf5c'] #for paper
#colors_hist=['#17478D', '#4FB0D6', '#F4B23C'] #for presentation
#colors_gauss=['#d62728', '#1f77b4', '#2ca02c']
#colors_gauss=['darkred','darkblue','darkgreen'] #for paper
colors_gauss=['#d62728', '#1f77b4', '#2ca02c']
#colors_gauss=['#0B2345', '#214959', '#6E501B']
g2_new=np.zeros(6)
g1_new=np.zeros(6)
S2histHighLim_el=175
S1histHighLim=40
binsperphe=5
s1max=14
ParamsEnergyMean=[[462.88,1.1351,2567.9],[471.07,1.1384,2585.7],[480.66,1.139,2600.4],[483.67,1.1402,2590],[484.32,1.1421,2600.7],[497.42,1.1401,2579.3]]
s1area_nocut=[]
s2area_nocut=[]
rad_nocut=[]
s1CutEvents=[]
dTimeCutEvents=[]
minS1area=[]
minS2area=[]
y_s2=[]
bin_s2=[]
gauss1x=[]
gauss2x=[]
gauss1_popt=[]
gauss1_poptOff=[]
gauss2_popt=[]
gauss2_poptOff=[]
err2800_s2_A=[]
err2800_s2_sig=[]
err0270_s2_A=[]
err0270_s2_sig=[]
handles=[]
lables=[]
text=[]
yy=[]
y_s1=[]
x_s1=[]
y_s1_norm=[]
bin_s1=[]
binss=[]
y_CE=[]
ThreshFitParams=[]
ThreshFitErrors=[]
poissonParams=[]
totQuant_popt=[]
totQuant_y=[]
totQuant_bins=[]
ratioArray=[]
resArray=[]
meanArray=[]
EnergyParams=[]
MeanS2_2800=[]
MeanS2_270=[]
MeanS1_2800=[]
MeanS2_2800_el=[]
MeanS2_270_el=[]
MeanS2_2800_Err=[]
MeanS2_270_Err=[]
MeanS2_2800_el_Err=[]
MeanS2_270_el_Err=[]
MeanCE=[]
MeanCE_Err=[]
SigmaS1_2800=[]
SigmaS2_2800=[]
SigmaS2_2800_el=[]
Integ_Err=[]
SigmaS2_2800_Err=[]
SigmaS2_2800_el_Err=[]
SigmaCE=[]
SigmaCE_Err=[]




for dataset in data:
	f = h5.File(dataset, 'r')
	name= f.filename
	print (name)
	#print (sets(f))

	#reading dvt22 data from file
	dvt22=f['dvt22']
	#dvt22.visit(printname)
	
	#reading s2area data from file
	dset=dvt22['s2area']
	s2area=dset[0]
	s2Area=np.array(s2area)

	#reading seArea_phe data from file
	dset2=dvt22['seArea_phe']
	seArea_phe=dset2[0]

	#processing seArea_phe variable
	seArea_phe=np.ones((len(seArea_phe)))
	seArea_phe=SE_avg[i]*seArea_phe

	#reading total_pulse_area from file
	dset3=dvt22['totalPulsesArea']
	totalPulsesArea=dset3[0]
	TPArea=np.array(totalPulsesArea)

	#reading s1Area from file
	dset4=dvt22['s1area']
	s1area=dset4[0]
	s1Area=np.array(s1area)

	#reading s2tau from file
	dset5=dvt22['s2tau']
	s2tau=dset5[0]
	s2Tau=np.array(s2tau)

	#reading radius from file
	dset6=dvt22['radius']
	radius=dset6[0]
	Radius=np.array(radius)
	
	#reading dtime from file
	dset7=dvt22['dtime']
	dtime=dset7[0]
	dTime=np.array(dtime)

	S2histHighLim=175*SE_avg[i]

	#print(max(dTime[np.logical_not(np.isnan(dTime))]/250))
	#cutting data
	#cut_data=(compound_and([s2Area/TPArea>0.8,Radius<35,s2Tau/250>0.6,s2Tau/250<2.5],len(s2Area)))
	cut_data=np.logical_or(compound_and([s2Area/TPArea>0.8,np.isnan(s1Area),Radius<35,s2Tau/250>0.6,s2Tau/250<2.5],len(s2Area)),\
		compound_and([(s2Area+s1Area)/TPArea>0.8,np.logical_not(np.isnan(s1Area)),Radius<35,s2Tau/250>0.6,s2Tau/250<1.4,dTime/250>3,dTime/250<30,np.log10(s2Area/s1Area)>2,np.log10(s2Area/s1Area)<3.4],len(s2Area)))
	cut_test=np.logical_or(compound_and([s2Area/TPArea>0.8,np.isnan(s1Area),Radius<35,s2Tau/250>0.6,s2Tau/250<2.5],len(s2Area)),\
		compound_and([(s2Area+s1Area)/TPArea>0.8,np.logical_not(np.isnan(s1Area)),Radius<35,s2Tau/250>0.6,s2Tau/250<1.4,dTime/250>3,dTime/250<30],len(s2Area)))
	cut_test2=np.logical_or(compound_and([s2Area/TPArea>0.8,np.isnan(s1Area),s2Tau/250>0.6,s2Tau/250<2.5],len(s2Area)),\
		compound_and([(s2Area+s1Area)/TPArea>0.8,np.logical_not(np.isnan(s1Area)),s2Tau/250>0.6,s2Tau/250<1.4,dTime/250>3,dTime/250<30,np.log10(s2Area/s1Area)>2,np.log10(s2Area/s1Area)<3.4],len(s2Area)))

	s2area_nocutTemp=(s2Area*cut_test)*2/85
	s2area_nocut.append(s2area_nocutTemp)
	s1area_nocutTemp=(s1Area*cut_test)*2/85
	s1area_nocut.append(s1area_nocutTemp)
	rad_nocut.append(Radius*cut_test2)

	#S2AREA IN PHE
	#processing data
	s2Area_cut=(s2Area*cut_data)
	pdata2=((s2Area_cut*2)/85) # IN PHE
	#pdata=(((s2Area_cut*2)/100)/seArea_phe)  #IN ELECTRONS
	pdata=pdata2[pdata2 !=0]
	pdata=pdata[pdata<S2histHighLim]
	#fit gaussians to peaks
	#first find the optimum number of bins
	numData = len(pdata)
	iqr = np.subtract(*np.percentile(pdata, [75, 25]))
	bw=2.0*iqr*(numData**(-1.0/3.0))
	numbins = math.floor((max(pdata)-min(pdata))/bw)
	y,bin=np.histogram(pdata, numbins,range=(0, S2histHighLim),density=True)
	numbins=100
	x=bin_to_x(bin)
	sumCD=sum(y)
	y_s2.append(y)
	bin_s2.append(bin)
	#splitting data into two peaks
	Gauss1x =x[x<1000]
	n=len(Gauss1x)
	Gauss1y=y[:n]
	Gauss2x=x[x>1000]
	Gauss2y=y[n:]
	gauss1x.append(Gauss1x)
	gauss2x.append(Gauss2x)
	#fitting Gauss1:
	mean1=sum(Gauss1x*Gauss1y)/np.sum(Gauss1y)
	sigma1=np.sqrt(sum(Gauss1y*(Gauss1x-mean1)**2)/np.sum(Gauss1y))
	popt1,pcov1=curve_fit(gaussian,Gauss1x,Gauss1y,p0=[1,mean1,sigma1])
	gauss1_popt.append(popt1)
	popt1_off,pcov1_off=curve_fit(gaussian_yoff,Gauss1x,Gauss1y,p0=[1,mean1,sigma1,0])
	gauss1_poptOff.append(popt1_off)
	#fitting Gauss2:
	mean2=sum(Gauss2x*Gauss2y)/np.sum(Gauss2y)
	sigma2=np.sqrt(sum(Gauss2y*(Gauss2x-mean2)**2)/np.sum(Gauss2y))
	popt2,pcov2=curve_fit(gaussian,Gauss2x,Gauss2y,p0=[1,mean2,sigma2])
	gauss2_popt.append(popt2)
	popt2_off,pcov2_off=curve_fit(gaussian_yoff,Gauss2x,Gauss2y,p0=[1,mean2,sigma2,0])
	gauss2_poptOff.append(popt2_off)
	#Plot the S2 spectrum for the three chosen voltages
	#if dataset in dataF1:
		#plt.subplot(212)
	#	plt.sca(axesS2[j])
	#	minS2area.append(min(bin[y!=0]))
	#	y_offset=y+offset_s2[j]
	#	aLib.stairs(bin,y_offset,color=colors_hist[j],linewidth=2.5)
	#	Gauss1y_offset=gaussian(Gauss1x,*popt1)+offset_s2[j]
	#	Gauss2y_offset=gaussian(Gauss2x,*popt2)+offset_s2[j]
	#	plt.plot(Gauss1x,Gauss1y_offset,color=colors_gauss[j], linewidth=1.2)
	#	plt.plot(Gauss2x,Gauss2y_offset,color=colors_gauss[j], linewidth=1.2)
	#calculating error of Gauss1 mean:
	conf_intTemp=stats.t.interval(.95, len(Gauss1y)-1, loc=popt1[1], scale=np.sqrt(np.diag((pcov1))[1]))
	conf_int270=popt1[1] - conf_intTemp[0]
	#calculating error of Gauss2 mean:
	conf_intTemp=stats.t.interval(.95, len(Gauss2y)-1, loc=popt2[1], scale=np.sqrt(np.diag((pcov2))[1]))
	conf_int2800=popt2[1] - conf_intTemp[0]
	#caluclating error of the Gauss2 sigma:
	conf_intSigmaTemp=stats.t.interval(.95, len(Gauss2y)-1, loc=popt2[2], scale=np.sqrt(np.diag((pcov2))[2]))
	conf_intSigma2800=popt2[2] - conf_intSigmaTemp[0]

	conf_intTemp=stats.t.interval(.95, len(Gauss2y)-1, loc=popt2_off[0], scale=np.sqrt(np.diag((pcov2_off))[0]))
	err2800_s2_A.append(popt2_off[0] - conf_intTemp[0])
	conf_intTemp=stats.t.interval(.95, len(Gauss2y)-1, loc=popt2_off[2], scale=np.sqrt(np.diag((pcov2_off))[2]))
	err2800_s2_sig.append(popt2_off[2]-conf_intTemp[0])
	conf_intTemp=stats.t.interval(.95, len(Gauss1y)-1, loc=popt1_off[0], scale=np.sqrt(np.diag((pcov1_off))[0]))
	err0270_s2_A.append(popt1_off[0]-conf_intTemp[0])
	conf_intTemp=stats.t.interval(.95, len(Gauss1y)-1, loc=popt1_off[2], scale=np.sqrt(np.diag((pcov1_off))[2]))
	err0270_s2_sig.append(popt1_off[2]-conf_intTemp[0])

	
	#S2AREA IN ELECTRONS
	pdata_el=(((s2Area_cut*2)/85)/seArea_phe)  #IN ELECTRONS
	pdata_el=pdata_el[pdata_el !=0]
	pdata_el=pdata_el[pdata_el<S2histHighLim_el]
	#fit gaussians to peaks
	numData = len(pdata_el)
	iqr = np.subtract(*np.percentile(pdata_el, [75, 25]))
	bw=2.0*iqr*(numData**(-1.0/3.0))
	numbins = math.floor((max(pdata_el)-min(pdata_el))/bw)
	y_el,bin_el=np.histogram(pdata_el, numbins, range=(0, S2histHighLim_el),density=True)
	x_el=bin_to_x(bin_el)
	sumCD=sum(y)
	#splitting data into two peaks
	Gauss1x_el =x_el[x_el<40]
	n_el=len(Gauss1x_el)
	Gauss1y_el=y_el[:n_el]
	Gauss2x_el=x_el[x_el>40]
	Gauss2y_el=y_el[n_el:]
	#fitting Gauss1:
	mean1_el=sum(Gauss1x_el*Gauss1y_el)/np.sum(Gauss1y_el)
	sigma1_el=np.sqrt(sum(Gauss1y_el*(Gauss1x_el-mean1_el)**2)/np.sum(Gauss1y_el))
	popt1_el,pcov1_el=curve_fit(gaussian,Gauss1x_el,Gauss1y_el,p0=[1,mean1_el,sigma1_el])
	#fitting Gauss2:
	mean2_el=sum(Gauss2x_el*Gauss2y_el)/np.sum(Gauss2y_el)
	sigma2_el=np.sqrt(sum(Gauss2y_el*(Gauss2x_el-mean2_el)**2)/np.sum(Gauss2y_el))
	popt2_el,pcov2_el=curve_fit(gaussian,Gauss2x_el,Gauss2y_el,p0=[1,mean2_el,sigma2_el])
	#calculating error of Gauss1 mean:
	conf_intTemp_el=stats.t.interval(.95, len(Gauss1y_el)-1, loc=popt1_el[1], scale=np.sqrt(np.diag((pcov1_el))[1]))
	conf_int270_el=popt1_el[1] - conf_intTemp_el[0]
	#calculating error of Gauss2 mean:
	conf_intTemp_el=stats.t.interval(.95, len(Gauss2y_el)-1, loc=popt2_el[1], scale=np.sqrt(np.diag((pcov2_el))[1]))
	conf_int2800_el=popt2_el[1] - conf_intTemp_el[0]
	#caluclating error of the Gauss2 sigma:
	conf_intSigmaTemp_el=stats.t.interval(.95, len(Gauss2y_el)-1, loc=popt2_el[2], scale=np.sqrt(np.diag((pcov2_el))[2]))
	conf_intSigma2800_el=popt2_el[2] - conf_intSigmaTemp_el[0]
	

	
	#plotting s1 histogram
	s1Area_cut=s1Area*cut_data
	pdata1=s1Area_cut*2/85
	pdata=pdata1[pdata1 !=0]
	pdata=pdata[np.logical_not(np.isnan(pdata))]
	pdata=pdata[(pdata>(-1/(binsperphe*2)))*(pdata<(S1histHighLim-1/(binsperphe*2)))]
	dTime_cut=dTime*cut_data
	dtdata=dTime_cut[pdata1 !=0]
	dtdata=dtdata[np.logical_not(np.isnan(pdata))]
	s1CutEvents.append(pdata)
	dTimeCutEvents.append(dtdata)
	numData = len(pdata)
	iqr = np.subtract(*np.percentile(pdata, [75, 25]))
	bw=2.0*iqr*(numData**(-1.0/3.0))
	numbins = math.floor((max(pdata)-min(pdata))/bw)
	#numbins=50
	y,bin=np.histogram(pdata, bins=binsperphe*S1histHighLim, range=(-1/(binsperphe*2),S1histHighLim-1/(binsperphe*2)), density=True)
	y_temp,bin=np.histogram(pdata, bins=binsperphe*S1histHighLim, range=(-1/(binsperphe*2),S1histHighLim-1/(binsperphe*2)))
	binWidth=bin[1]-bin[0]
	sumHist_fit=0
	sumHist_plot=0
	x=bin_to_x(bin)
	x_small=x[:binsperphe*5]
	n=len(x_small[y[:binsperphe*5]==0])
	x=x[n:]
	y=y[n:]
	y_temp=y_temp[n:]
	bin=bin[n:]
	hilim=15
	lolim=6
	fitx=x[(x>lolim)*(x<hilim)]
	fity=y[(x>lolim)*(x<hilim)]
	for m in np.arange(len(fity)):
		sumHist_fit=sumHist_fit+binWidth*fity[m]
	mean=sum(fitx*fity)/np.sum(fity)
	sigma=np.sqrt(sum(fity*(fitx-mean)**2)/np.sum(fity))
	popt,pcov=curve_fit(SmearedPoisson,fitx,fity,p0=[sumHist_fit,sigma,mean])
	poissonParams.append(popt)

	integ, intgErr=integrate.quad(IntVarVal, 0, 40, args=(popt[1],popt[2]))
	SqrtVarS1=np.sqrt(integ-popt[2]**2)
	#Calculate threshold and save variabled needed for threshold error calcs
	for m in np.arange(len(y)):
		sumHist_plot=sumHist_plot+binWidth*y[m]
	yy.append(SmearedPoisson_NewRange(x,sumHist_plot, *popt))
	minS1area.append(min(bin[y!=0]))
	thresh=y/yy[i];
	poptT,pcovT=curve_fit(logistic,x[(x>minS1area[i])*(x<s1max)],thresh[(x>minS1area[i])*(x<s1max)], p0=[1,1,minS1area[i]])
	#plt.plot(x_full,logistic(x_full,*popt),color=colors_gauss[i], marker='',linestyle='-')
	ThreshFitParams.append(poptT)
	conf_intTemp=stats.t.interval(.95, len(x[(x>minS1area[i])*(x<s1max)])-1, loc=poptT[2], scale=np.sqrt(np.diag((pcovT))[2]))
	ThreshFitErrors.append(poptT[2]-conf_intTemp[0])
	y,bin=np.histogram(pdata, bins=numbins, range=(-1/(binsperphe*2),S1histHighLim-1/(binsperphe*2)),density=True)
	x_temp=bin_to_x(bin)
	x_small=x_temp[x_temp<=5]
	n=len(x_small[y[x_temp<=5]==0])
	y=y[n:]
	bin=bin[n:]
	x_s1.append(x)
	y_s1_norm.append(y)
	bin_s1.append(bin)
	y_s1.append(y_temp)
	#plot s1 spectrum for chosen fields
	#if dataset in dataF1:
		#plt.subplot(211)
	#	plt.sca(axesS1[j])
	#	y,bin=np.histogram(pdata, bins=numbins, range=(-1/(binsperphe*2),S1histHighLim-1/(binsperphe*2)),density=True)
	#	x_temp=bin_to_x(bin)
	#	x_small=x_temp[x_temp<=5]
	#	n=len(x_small[y[x_temp<=5]==0])
	#	y=y[n:]
	#	bin=bin[n:]
	#	y_offset=y+offset_s1[j]
	#	h=aLib.stairs(bin,y_offset,color=colors_hist[j], linewidth=2.5)
	#	lable_str = str(field[i])+' V/cm'
	#	lables.append(lable_str)
	#	handles.append(h)
	#	yy_offset=yy[i]+offset_s1[j]
		#plt.plot(x,yy_offset,color=colors_gauss[j])
		#plt.plot((popt[1], popt[1]), (0, .05), color=colors_gauss[j], linestyle='--',linewidth=2)
	#	text_str='Mean 2.8 keV S1=%.3f'%popt[0]+' phe'
	#	text.append(text_str)
	
	#calculating and fitting combined energy
	pdata2=pdata2[pdata1 != 0] #S2 area in phe
	pdata1=pdata1[pdata1 != 0] #s1 area
	pdata2=pdata2[np.logical_not(np.isnan(pdata1))]
	pdata1=pdata1[np.logical_not(np.isnan(pdata1))]
	#fit total quanta first
	totQuant=pdata1/g1+pdata2/g2[i]
	yT,bintemp=np.histogram(totQuant, bins=50,range=(0,400),density=True)
	xT=bin_to_x(bintemp)
	meanT=sum(yT*xT)/np.sum(yT)
	sigmaT=np.sqrt(sum(yT*(xT-meanT)**2)/np.sum(yT))
	poptT,pcovT=curve_fit(gaussian,xT,yT,p0=[1,meanT,sigmaT])
	totQuant_popt.append(poptT)
	totQuant_y.append(yT)
	totQuant_bins.append(bintemp)
	xg2=np.arange(0,23,.1)
	#successively subtract .1 to g2 until sigma of Ecomb gaussian is smallest
	ratioTemp=[]
	resTemp=[]
	EMeanTemp=[]
	for m in np.arange(len(xg2)):
		if xg2[m]<10:
			hilim=7000
		elif 10<=xg2[m]<15:
			hilim=10000
		elif 15<=xg2[m]<20:
			hilim=40000
		else:
			hilim=200000
		Ecomb=w_fnc*(pdata1/g1+pdata2/(g2[i]-xg2[m]))
		yE,bintemp=np.histogram(Ecomb,bins=10000, range=(0,200000), density=True)
		xE=bin_to_x(bintemp)
		fitx=xE[(xE<hilim)]
		fity=yE[(xE<hilim)]
		meanE=sum(fitx*fity)/np.sum(fity)
		sigmaE=np.sqrt(sum(fity*(fitx-meanE)**2)/np.sum(fity))
		poptE,pcovE=curve_fit(gaussian,fitx,fity,p0=[1,meanE,sigmaE])
		ratioTemp=np.append(ratioTemp,(g2[i]-xg2[m])/g1)
		resTemp=np.append(resTemp,np.abs(poptE[2]/poptE[1]))
		EMeanTemp=np.append(EMeanTemp,poptE[1])
	ratioArray.append(np.array(ratioTemp))
	resArray.append(np.array(resTemp))
	meanArray.append(np.array(EMeanTemp))
	g2_new[i]=g1*ratioTemp[resTemp==min(resTemp)]
	ratio=g1/g2_new[i]
	meanTemp=[]
	xg2=np.arange(14,17,.1)
	for m in np.arange(len(xg2)):
		if xg2[m]<10:
			hilim=7000
		elif 10<=xg2[m]<15:
			hilim=10000
		elif 15<=xg2[m]<20:
			hilim=40000
		else:
			hilim=200000
		Ecomb=w_fnc*(pdata1/((g2_new[i]+xg2[m])*ratio)+pdata2/(g2_new[i]+xg2[m]))
		yE,bintemp=np.histogram(Ecomb,bins=10000, range=(0,20000), density=True)
		xE=bin_to_x(bintemp)
		fitx=xE[(xE<hilim)]
		fity=yE[(xE<hilim)]
		meanE=sum(fitx*fity)/np.sum(fity)
		sigmaE=np.sqrt(sum(fity*(fitx-meanE)**2)/np.sum(fity))
		poptE,pcovE=curve_fit(gaussian,fitx,fity,p0=[1,meanE,sigmaE])
		meanTemp=np.append(meanTemp,poptE[1])
	possibleValuesX=xg2[(meanTemp>2600)*(meanTemp<3000)]
	possibleValues=np.around(meanTemp[(meanTemp>2600)*(meanTemp<3000)],2)
	diff=[]
	diffSign=[]
	for m in np.arange(len(possibleValues)):
		diff=np.append(diff,possibleValues[m]-2800)
		if diff[m]<0:
			temp=True
		else:
			temp=False
		diffSign=np.append(diffSign,temp)
	diffSign=np.array(diffSign,dtype=bool)
	diff=np.abs(diff)
	NminDiff=np.where(diff==min(diff))
	diff[diffSign]=-1*diff[diffSign]
	g2_new[i]=g2_new[i]+possibleValuesX[possibleValues==diff[NminDiff]+2800]
	g1_new[i]=ratio*g2_new[i]
	Ecomb=w_fnc*(pdata1/g1_new[i]+pdata2/g2_new[i])
	Ecomb=Ecomb[Ecomb<4000]
	numData = len(Ecomb)
	iqr = np.subtract(*np.percentile(Ecomb, [75, 25]))
	bw=2.0*iqr*(numData**(-1.0/3.0))
	numbins = math.floor((max(Ecomb)-min(Ecomb))/bw)
	yE,bintemp=np.histogram(Ecomb,bins=numbins, range=(0,4000), density=True)
	xE=bin_to_x(bintemp)
	fitx=xE[(xE<hilim)]
	fity=yE[(xE<hilim)]
	meanE=sum(fitx*fity)/np.sum(fity)
	sigmaE=np.sqrt(sum(fity*(fitx-meanE)**2)/np.sum(fity))
	poptE,pcovE=curve_fit(gaussian,fitx,fity,p0=[1,meanE,sigmaE])
	y_CE.append(yE)
	binss.append(bintemp)
	EnergyParams.append(poptE)
	#error of energy mean
	conf_intTemp=stats.t.interval(.95, len(fity)-1, loc=poptE[1], scale=np.sqrt(np.diag((pcovE))[1]))
	conf_intMeanE=poptE[1] - conf_intTemp[0]
	#error of energy sigma
	conf_intTemp=stats.t.interval(.95, len(fity)-1, loc=poptE[2], scale=np.sqrt(np.diag((pcovE))[2]))
	conf_intSigmaE=poptE[2]- conf_intTemp[0]

	p=p+1
	if dataset in dataF1:
		j=j+1

	# recording mean of peak's fit and error for yield vs field plots
	MeanS2_2800.append(popt2[1])
	MeanS2_270.append(popt1[1])
	MeanS2_2800_el.append(popt2_el[1])
	MeanS2_270_el.append(popt1_el[1])
	MeanS1_2800.append(popt[2])
	MeanS2_2800_Err.append((((abs(conf_int2800)/popt2[1])**2+((SE_std[i]/popt2[1])**2))**0.5)*popt2[1])
	MeanS2_270_Err.append((((abs(conf_int270)/popt1[1])**2+((SE_std[i]/popt1[1])**2))**0.5)*popt1[1])
	MeanS2_2800_el_Err.append((((abs(conf_int2800_el)/popt2_el[1])**2+((SE_std[i]/SE_avg[i])**2))**0.5)*popt2_el[1])
	MeanS2_270_el_Err.append((((abs(conf_int270_el)/popt1_el[1])**2+((SE_std[i]/SE_avg[i])**2))**0.5)*popt1_el[1])
	MeanCE.append(poptE[1])
	MeanCE_Err.append(conf_intMeanE)
	# recording sigma for s1, s2, and Combined energy of the 2800 eV peak
	SigmaS2_2800.append(popt2[2])
	SigmaS2_2800_el.append(popt2_el[2])
	SigmaS1_2800.append(SqrtVarS1)
	SigmaS2_2800_Err.append(conf_intSigma2800)
	SigmaS2_2800_el_Err.append(conf_intSigma2800_el)
	SigmaCE.append(poptE[2])
	SigmaCE_Err.append(conf_intSigmaE)
	Integ_Err.append(intgErr)
	i=i+1


MeanS2_2800=np.array(MeanS2_2800)
MeanS2_270=np.array(MeanS2_270)
MeanS1_2800=np.array(MeanS1_2800)
MeanS2_2800_el=np.array(MeanS2_2800_el)
MeanS2_270_el=np.array(MeanS2_270_el)
errorbar2800_s2=np.array(MeanS2_2800_Err)
errorbar270_s2=np.array(MeanS2_270_Err)
errorbar2800_s2_el=np.array(MeanS2_2800_el_Err)
errorbar270_s2_el=np.array(MeanS2_270_el_Err)
MeanCE=np.array(MeanCE)
MeanCE_Err=np.array(MeanCE_Err)
SigmaS2_2800=np.array(SigmaS2_2800)
SigmaS2_2800_el=np.array(SigmaS2_2800_el)
SigmaS1_2800=np.array(SigmaS1_2800)
SigmaS2_2800_Err=np.array(SigmaS2_2800_Err)
SigmaS2_2800_el_Err=np.array(SigmaS2_2800_el_Err)
SigmaCE=np.array(SigmaCE)
SigmaCE_Err=np.array(SigmaCE_Err)
Integ_Err=np.array(Integ_Err)


#S1 ERROR APPROXIMATIONS
for i in np.arange(len(field)):
	print('50% x-value')
	print(ThreshFitParams[i][2])

AvgThreshParams=np.mean(ThreshFitParams,axis=0)
AvgThreshError=1/6*np.sqrt(np.sum(np.array(ThreshFitErrors)**2))
AvgThreshBiggerParams=np.array([AvgThreshParams[0],AvgThreshParams[1],AvgThreshParams[2]+AvgThreshError])
AvgThreshSmallerParams=np.array([AvgThreshParams[0],AvgThreshParams[1],AvgThreshParams[2]-AvgThreshError])
changex0=np.arange(-.1,.1,.001)
CO=15
x_GF=x[x<CO]
yy_GF=[]
Corrected_MeanS1_2800=[]
errorbar_s1_stat=[]
SysHi=[]
SysLo=[]
NewLambda=[]
for i in np.arange(len(field)):
	yy_GF.append(yy[i][x<CO])
for i in np.arange(len(field)):
	sumYS1=0
	for j in np.arange(len(yy[i])):
		sumYS1=sumYS1+binWidth*y_s1[i][j]
	data=y_s1[i][x<CO]
	initGuess=yy[i][x<CO]
	chisq=[]
	for l in np.arange(len(changex0)):
		yynew=(SmearedPoisson_NewRange(x, 1, poissonParams[i][0], poissonParams[i][1], poissonParams[i][2]-changex0[l]))
		model=yynew[x<CO]*logistic(x_GF, *AvgThreshParams)
		chisq=np.append(chisq,sum((data-sumYS1*model)**2/(sumYS1*model)))
	x0=changex0[chisq.argmin()]
	tempDiff=abs(chisq-(min(chisq)+1))
	x1=changex0[tempDiff.argmin()]
	np.put(tempDiff,tempDiff.argmin(),10)
	x2=changex0[tempDiff.argmin()]
	#ensure that x1<x2
	if x1<x2:
		xtemp=x1
		x1=x2
		x2=xtemp
	Corrected_MeanS1_2800=np.append(Corrected_MeanS1_2800,MeanS1_2800[i]+x0)
	errorbar_s1_stat=np.append(errorbar_s1_stat, x1-x0)
	NewLambda=np.append(NewLambda,poissonParams[i][2]-x0)

	chisq=[]
	for l in np.arange(len(changex0)):
		yynew=(SmearedPoisson_NewRange(x, 1, poissonParams[i][0], poissonParams[i][1], poissonParams[i][2]-changex0[l]))
		model=yynew[x<CO]*logistic(x_GF, *AvgThreshBiggerParams)
		chisq=np.append(chisq,sum((data-sumYS1*model)**2/(sumYS1*model)))
	x4=changex0[chisq.argmin()]
	SysHi=np.append(SysHi,x4-x0)

	chisq=[]
	for l in np.arange(len(changex0)):
		yynew=(SmearedPoisson_NewRange(x, 1, poissonParams[i][0], poissonParams[i][1], poissonParams[i][2]-changex0[l]))
		model=yynew[x<CO]*logistic(x_GF, *AvgThreshSmallerParams)
		chisq=np.append(chisq,sum((data-sumYS1*model)**2/(sumYS1*model)))
	x5=changex0[chisq.argmin()]
	SysLo=np.append(SysLo,x0-x5)
errorbar_s1_sys=[SysHi,SysLo]
errorbar_s1_total=[np.sqrt(SysHi**2+errorbar_s1_stat**2),np.sqrt(SysLo**2+errorbar_s1_stat**2)]


#Make a combined thresh+smeared poisson array
CombinedCurve=[]
for i in np.arange(len(field)):
	CombinedCurve.append(yy[i]*logistic(x,*AvgThreshParams))

#Putting everything in quanta (either electrons or photons)
s1_2800eV_corr=Corrected_MeanS1_2800/g1/1.175 #1.175 is the 2 photoelectron correction  NUMBER OF PRODUCED PHOTONS
errorbar_s1_corr=Corrected_MeanS1_2800/g1/1.175*np.sqrt((errorbar_s1_stat/Corrected_MeanS1_2800)**2+(g1_err)**2)
errorbar_s1_corr_stat=Corrected_MeanS1_2800/g1/1.175*(errorbar_s1_stat/Corrected_MeanS1_2800)
errorbar_s1_total_corr=Corrected_MeanS1_2800/g1/1.175*[np.sqrt((errorbar_s1_total[0]/Corrected_MeanS1_2800)**2+(g1_err)**2),np.sqrt((errorbar_s1_total[1]/Corrected_MeanS1_2800)**2+(g1_err)**2)]
s2_2800eV_corr=MeanS2_2800_el/EE #number of PRODUCED electrons
errorbar_s2_corr=MeanS2_2800_el/EE*np.sqrt((errorbar2800_s2_el/MeanS2_2800_el)**2+(EE_err/EE)**2)
errorbar_s2_corr_stat=MeanS2_2800_el/EE*(errorbar2800_s2_el/MeanS2_2800_el)
s2_270ev_corr=MeanS2_270_el/EE
errorbar_s2_270eVcorr=MeanS2_270_el/EE*np.sqrt((errorbar270_s2_el/MeanS2_270_el)**2+(EE_err/EE)**2)
errorbar_s2_270eVcorr_stat=MeanS2_270_el/EE*(errorbar270_s2_el/MeanS2_270_el)
totalQuanta_270eV=s2_270ev_corr*1.06
totalQuanta_270eV_Err=np.sqrt(errorbar_s2_270eVcorr**2+.02**2) #.02 is from the 2 percent discrepinacy between 4 and 6 and 6 and 8%
totalQuanta=np.array(s1_2800eV_corr+s2_2800eV_corr)
totalQuanta_Err=np.sqrt(errorbar_s1_corr**2+errorbar_s2_corr**2)
totalQuanta_Err_stat=np.sqrt(errorbar_s1_corr_stat**2+errorbar_s2_corr_stat**2)

#Is W higher in a significant way
print('W Significance')
for i in np.arange(len(field)):
	print(field[i])
	#find varience of 2.8 total quanta
	#varience of s1
	integ, intgErr=integrate.quad(IntVarVal, 0, 40, args=(poissonParams[i][1],NewLambda[i]))
	SqrtVarS1_2800=np.sqrt(integ-NewLambda[i]**2)/g1/1.175/2.8
	#varience of s2
	SqrtVarS2_2800=gauss2_popt[i][2]/g2[i]/2.8
	SqrtVarT_2800=np.sqrt(SqrtVarS1_2800**2+SqrtVarS2_2800**2)/2.8

	#varience of .27 s2
	SqrtVarS2_0270=gauss1_popt[i][2]/g2[i]/.27

	WSignificance=(totalQuanta[i]/2.8-s2_270ev_corr[i]/.27)/np.sqrt(SqrtVarT_2800**2+SqrtVarS2_0270**2)
	WSignificance_other=(totQuant_popt[i][1]/2.8-s2_270ev_corr[i]/.27)/np.sqrt((totQuant_popt[i][2]/2.8)**2+(SqrtVarS2_0270)**2)
	WSignificance_3=(totalQuanta[i]/2.8-s2_270ev_corr[i]/.27)/np.sqrt(totalQuanta_Err[i]**2+errorbar_s2_270eVcorr[i]**2)
	WSignificance06=(totalQuanta[i]/2.8-(s2_270ev_corr[i]*1.06)/.27)/np.sqrt(SqrtVarT_2800**2+SqrtVarS2_0270**2)
	WSignificance06_other=(totQuant_popt[i][1]/2.8-(s2_270ev_corr[i]*1.06)/.27)/np.sqrt((totQuant_popt[i][2]/2.8)**2+(SqrtVarS2_0270)**2)
	WSignificance06_3=(totalQuanta[i]/2.8-(s2_270ev_corr[i]*1.06)/.27)/np.sqrt(totalQuanta_Err[i]**2+errorbar_s2_270eVcorr[i]**2)
	print('Using s2 and s1 hists: '+str(WSignificance))
	print('Using total quanta hist: '+str(WSignificance_other))
	print('Using errors: '+str(WSignificance_3))
	print('Using s2 and s1 hists for tot .27: '+str(WSignificance06))
	print('Using total quanta hist for tot .27: '+str(WSignificance06_other))
	print('Using errors for tot .27: '+str(WSignificance06_3))


#CALCULATION OF TIB PARAMETER
alpha1=0.06
r_06=np.array([])
i=0
for value in s1_2800eV_corr:
	r_06 = np.append(r_06,(value/s2_2800eV_corr[i] - alpha1) / (value/s2_2800eV_corr[i]+1))
	i=i+1
r_06New=1-r_06
y=np.array([])
x=Symbol('x')
for value in r_06New:
	def TIBFunc(x):
		return value-1/x*log(x+1)
	temp = broyden1(TIBFunc,.2)
	y=np.append(y,temp) 
Ni_06=s2_2800eV_corr/(1-r_06)
TIB_06=4*y/Ni_06
r_06_Err=((1-alpha1)/(s1_2800eV_corr+s2_2800eV_corr)**2)*np.sqrt((s1_2800eV_corr*errorbar_s1_corr_stat)**2+(s2_2800eV_corr*errorbar_s2_corr_stat)**2)
Ni_06_Err=s2_2800eV_corr/(1-r_06)*np.sqrt((errorbar_s2_corr_stat/s2_2800eV_corr)**2+(r_06_Err/r_06)**2)
TIB_06_Err=4*y/Ni_06*(Ni_06_Err/Ni_06)


#use estimate of 0.27 keV s1 peak: 0.1-0.2phe
avgNphoton=.2/g1/1.175 #estimate of s1_270eV_corr
r_06_270est = (avgNphoton/s2_270ev_corr - alpha1) / (avgNphoton/s2_270ev_corr+1)
r_06_270est_Err=((1-alpha1)/(avgNphoton+s2_270ev_corr)**2)*np.sqrt((s2_270ev_corr*errorbar_s2_270eVcorr)**2)

Ni_06_270est=s2_270ev_corr/(1-r_06_270est)

print('fraction of recombination')
print(r_06)
print(np.mean(r_06))
print(np.std(r_06))
print(r_06_270est)
print(np.mean(r_06_270est))
print(np.std(r_06_270est))
print((np.sqrt(r_06_270est_Err[0]**2+r_06_270est_Err[1]**2+r_06_270est_Err[2]**2+r_06_270est_Err[3]**2+r_06_270est_Err[4]**2+r_06_270est_Err[5]**2)/6))

#Fitting TIB data to curve
poptT06,pcovT06=curve_fit(TIB_fit,field,TIB_06)
print(poptT06)
#poptT2,pcovT2=curve_fit(TIB_fit,field,TIB_2)
Aunct=stats.t.interval(.95, len(fity)-1, loc=poptT06[0], scale=np.sqrt(np.diag((pcovT06))[0]))
deltaUnct=stats.t.interval(.95, len(fity)-1, loc=poptT06[1], scale=np.sqrt(np.diag((pcovT06))[1]))


#Calculate the curve values for number of photons and number of electrons by using the A(field)^-delta=TIB fit
Ni_06_avg=np.mean(Ni_06)
changex0=np.arange(-1,1,.01)
TIB_06calc=TIB_fit(field,*poptT06)
chisquared=[]
for l in np.arange(len(changex0)):
	xi=(Ni_06_avg+changex0[l])*TIB_06calc/4
	r=1-1/xi*np.log(xi+1)
	nphotons=s2_2800eV_corr*(r+alpha1)/(1-r)
	nelectrons=s1_2800eV_corr*(1-r)/(r+alpha1)
	chisquared_ph=sum((s1_2800eV_corr-nphotons)**2/nphotons)
	chisquared_el=sum((s2_2800eV_corr-nelectrons)**2/nelectrons)
	chisquared.append(chisquared_el+chisquared_ph)
chisquaredMin=min(chisquared)
x0=changex0[chisquared==chisquaredMin]
Ni_best=Ni_06_avg+x0

Ni_06_270est_avg=np.mean(Ni_06_270est)
Ni_270est_best=Ni_06_270est_avg+.2

field_fine=np.arange(100,3900,10)/1000
TIB_06calc=TIB_fit(field_fine,*poptT06)
xi_06=Ni_best*TIB_06calc/4
r_06calc=1-1/xi_06*np.log(xi_06+1)
Nex=alpha1*Ni_best
nelectrons=Ni_best*(1-r_06calc)
nphotons=Nex+r_06calc*Ni_best
nphotons_06calc=nelectrons*(r_06calc+alpha1)/(1-r_06calc)
nelectrons_06calc=nphotons*(1-r_06calc)/(r_06calc+alpha1)
ntotal_06calc=nphotons_06calc+nelectrons_06calc

xi_270est_06=Ni_270est_best*TIB_06calc/4
r_270est_06calc=1-1/xi_270est_06*np.log(xi_270est_06+1)
Nex_270est=alpha1*Ni_270est_best
nelectrons=Ni_270est_best*(1-r_270est_06calc)
nphotons=Nex_270est+r_270est_06calc*Ni_270est_best
nphotons_270est_06calc=nelectrons*(r_270est_06calc+alpha1)/(1-r_270est_06calc)
nelectrons_270est_06calc=nphotons*(1-r_270est_06calc)/(r_270est_06calc+alpha1)
ntotal_270est_06calc=nphotons_270est_06calc+nelectrons_270est_06calc

#Branching Ratio
RatioTemp=[]
RatioErrTemp=[]
for i in np.arange(6):
	Peak027Area=gauss1_poptOff[i][0]*gauss1_poptOff[i][2]*np.sqrt(2*np.pi)
	Peak280Area=gauss2_poptOff[i][0]*gauss2_poptOff[i][2]*np.sqrt(2*np.pi)
	RatioTemp.append(Peak027Area/Peak280Area)
	ErrTemp=Peak027Area/Peak280Area*np.sqrt((err0270_s2_A[i]/gauss1_poptOff[i][0])**2+(err0270_s2_sig[i]/gauss1_poptOff[i][2])**2+(err2800_s2_A[i]/gauss2_poptOff[i][0])**2+(err2800_s2_sig[i]/gauss2_poptOff[i][2])**2)
	RatioErrTemp.append(ErrTemp)
Ratio=np.mean(RatioTemp)
print('Branching Ratio')
print(Ratio)
RatioErr=1/6*np.sqrt(np.sum(np.array(RatioErrTemp)**2))
RatioErr_other=1/(2*np.sqrt(len(RatioTemp)))*(max(RatioTemp)-min(RatioTemp))
print(RatioErr_other)

#TEST FIGURES
x_reg=np.arange(50)
plt.figure()
f,axarr=plt.subplots(nrows=2,ncols=3,sharex='all',sharey='all')
for i in np.arange(len(field)):
	if i<3:
		plt.axes(axarr[0,i])
	else:
		plt.axes(axarr[1,i-3])
	plt.plot(s1area_nocut[i],np.log10(s2area_nocut[i]/s1area_nocut[i]),marker='.',linestyle='',markersize=5,color='k')
	plt.plot(x_reg,np.ones(50)*2,marker='',linestyle='--', linewidth=2,color='r')
	plt.plot(x_reg,np.ones(50)*3.4,marker='',linestyle='--', linewidth=2,color='r')
	plt.xlim([0,50])
	plt.ylim([-1,4])
	plt.title(str(field[i])+' V/cm')

plt.savefig('DiscSpCut.png')
plt.show()

plt.figure()
f,axarr=plt.subplots(nrows=2,ncols=3,sharex='all',sharey='all')
for i in np.arange(len(field)):
	if i<3:
		plt.axes(axarr[0,i])
	else:
		plt.axes(axarr[1,i-3])
	radhist,binbin = np.histogram(rad_nocut[i],bins=50, range=(0,50), density=True)

	radhist_cum=np.cumsum(radhist)
	aLib.stairs(binbin,radhist_cum, color='k', linestyle='-', linewidth=2)
	plt.plot([35,35],[0,1],color='r',linestyle='--',linewidth=3)
	plt.xlim([0,50])
	plt.ylim([0,1])
	plt.title(str(field[i])+' V/cm')

plt.savefig('Radcut.png')
plt.show()


#FIGURE 1 plot asthetics
#s1Area plot parameters
plt.figure(figsize=(8,8))
gs1=gridspec.GridSpec(3,1)
gs1.update(bottom=.48,top=.95, left=.13, right=.95, hspace=.05)
gs2=gridspec.GridSpec(3,1)
gs2.update(bottom=.11,top=.37, left=.13, right=.95, hspace=.05)
ax1=plt.subplot(gs1[2])
ax2=plt.subplot(gs1[1])
ax3=plt.subplot(gs1[0])
axesS1=[ax1,ax2,ax3]
ax4=plt.subplot(gs2[2])
ax5=plt.subplot(gs2[1])
ax6=plt.subplot(gs2[0])
axesS2=[ax4,ax5,ax6]
offset_s1=np.array([0,.16,.32])
offset_s2=np.array([0,0.001,0.002])
ticks_y_s1=[str(0),str(.04),str(.08),str(.12),str(.16)]
ticks_y_s2=[str(0),str(.4),str(.8)]
ticks_4blank=['','','','']
s1thresh=np.mean(minS1area)
d=.007
kwargs=dict(transform=ax1.transAxes, color='k', clip_on=False)
chosenThree=[0,2,5]
for j in np.arange(3):
	a=chosenThree[j]
	plt.sca(axesS1[j])
	plt.plot((s1thresh, s1thresh), (0,.5), linestyle='--',color='k')
	bin=bin_s1[a]
	y=y_s1_norm[a]
	y_offset=y+offset_s1[j]
	#print(y_offset)
	aLib.stairs(bin,y_offset,color=colors_gauss[j],linewidth=3,alpha=.8)
	yy_offset=yy[a]+offset_s1[j]
	yyComb_offset=CombinedCurve[a]+offset_s1[j]
	plt.plot(x_s1[a],yy_offset,color='k', linestyle='--',linewidth=2)
	plt.plot(x_s1[a],yyComb_offset,color='k', linewidth=2)
	plt.xlim([0,25])
	kwargs.update(transform=axesS1[j].transAxes)
	plt.grid()
	if j==0:
		plt.xlabel('S1 area (photoelectrons)', fontsize=18)
		plt.yticks(np.arange(0,.16,.04),ticks_y_s1)
		plt.ylim([offset_s1[j],offset_s1[j]+.15])
		axesS1[j].spines['top'].set_visible(False)
		axesS1[j].tick_params(axis='x', which='both', bottom='on', labelbottom='on', direction='out',labelsize=14)
		axesS1[j].tick_params(axis='x', which='both', top='off', labeltop='off')
		axesS1[j].tick_params(axis='y',direction='out', right='off',labelsize=14)
		axesS1[j].plot((-d,+d),(1-2*d,1+2*d),**kwargs)
		axesS1[j].plot((1-d,1+d), (1-2*d,1+2*d), **kwargs)
	elif j==1:
		plt.yticks(np.arange(.16,.32,.04),ticks_y_s1)
		plt.ylim([offset_s1[j]-.01,offset_s1[j]+.15])
		axesS1[j].spines['top'].set_visible(False)
		axesS1[j].spines['bottom'].set_visible(False)
		axesS1[j].tick_params(axis='x', which='both',bottom='off', labelbottom='off')
		axesS1[j].tick_params(axis='x', which='both',top='off', labeltop='off')
		axesS1[j].tick_params(axis='y',direction='out', right='off',labelsize=14)
		axesS1[j].plot((-d,+d),(1-2*d,1+2*d),**kwargs)
		axesS1[j].plot((1-d,1+d), (1-2*d,1+2*d), **kwargs)
		axesS1[j].plot((-d,+d),(-2*d,+2*d),**kwargs)
		axesS1[j].plot((1-d,1+d), (-2*d,+2*d), **kwargs)
	else:
		plt.yticks(np.arange(.32,.48,.04),ticks_y_s1)
		plt.ylim([offset_s1[j]-.01,offset_s1[j]+.15])
		axesS1[j].spines['bottom'].set_visible(False)
		axesS1[j].tick_params(axis='x', which='both',bottom='off', labelbottom='off')
		axesS1[j].tick_params(axis='x', which='both',top='on', labeltop='off')
		axesS1[j].tick_params(axis='y',direction='out', right='off',labelsize=14)
		axesS1[j].plot((-d,+d),(-2*d,+2*d),**kwargs)
		axesS1[j].plot((1-d,1+d), (-2*d,+2*d), **kwargs)



#s2area plot parameters
for j in np.arange(3):
	a=chosenThree[j]
	plt.sca(axesS2[j])
	#plt.gca().set_xscale('log')
	plt.xlim([0,5000])
	kwargs.update(transform=axesS2[j].transAxes)
	plt.grid()
	bin=bin_s2[a]
	y=y_s2[a]
	s2thresh=min(bin[y!=0])
	Gauss1x=gauss1x[a]
	Gauss2x=gauss2x[a]
	popt1=gauss1_popt[a]
	popt2=gauss2_popt[a]
	y_offset=y+offset_s2[j]
	aLib.stairs(bin,y_offset,color=colors_gauss[j],linewidth=3,alpha=.8)
	Gauss1y_offset=gaussian(Gauss1x,*popt1)+offset_s2[j]
	Gauss2y_offset=gaussian(Gauss2x,*popt2)+offset_s2[j]
	plt.plot(Gauss1x,Gauss1y_offset,color='k', linewidth=2)
	plt.plot(Gauss2x,Gauss2y_offset,color='k', linewidth=2)
	plt.xscale('log')
	if j==0:
		plt.xlabel('S2 area (photoelectrons)', fontsize=18)
		plt.xlim([250,5000])
		#ticks_x=[str(0),str(1000), str(2000),str(3000),str(4000)]
		ticks_x=[str(300),str(''),str(''),str(''),str(''),str(''),str(''),str(1000), str(''),str(3000),str(''),str('')]
		#plt.xticks([0,1000,2000,3000,4000,5000],ticks_x)
		plt.xticks([300,400,500,600,700,800,900,1000,2000,3000,4000,5000],ticks_x)
		plt.yticks(np.arange(0,.001,.0004),ticks_y_s2)
		plt.ylim([offset_s2[j],offset_s2[j]+.001])
		axesS2[j].spines['top'].set_visible(False)
		axesS2[j].tick_params(axis='x', which='both', bottom='on', labelbottom='on', direction='out',labelsize=14)
		axesS2[j].tick_params(axis='x', which='both', top='off', labeltop='off')
		axesS2[j].tick_params(axis='y',direction='out', right='off',labelsize=14)
		axesS2[j].plot((-d,+d),(1-2*d,1+2*d),**kwargs)
		axesS2[j].plot((1-d,1+d), (1-2*d,1+2*d), **kwargs)
	elif j==1:
		plt.yticks(np.arange(.001,.002,.0004),ticks_y_s2)
		plt.ylim([offset_s2[j]-.0001,offset_s2[j]+.001])
		plt.xlim([250,5000])
		#plt.xticks([0,1000,2000,3000,4000,5000])
		plt.xticks([300,400,500,600,700,800,900,1000,2000,3000,4000,5000])
		axesS2[j].spines['top'].set_visible(False)
		axesS2[j].spines['bottom'].set_visible(False)
		axesS2[j].tick_params(axis='x', which='both',bottom='off', labelbottom='off')
		axesS2[j].tick_params(axis='x', which='both',top='off', labeltop='off')
		axesS2[j].tick_params(axis='y',direction='out', right='off',labelsize=14)
		axesS2[j].plot((-d,+d),(1-2*d,1+2*d),**kwargs)
		axesS2[j].plot((1-d,1+d), (1-2*d,1+2*d), **kwargs)
		axesS2[j].plot((-d,+d),(-2*d,+2*d),**kwargs)
		axesS2[j].plot((1-d,1+d), (-2*d,+2*d), **kwargs)
	else:
		plt.yticks(np.arange(.002,.003,.0004),ticks_y_s2)
		plt.ylim([offset_s2[j]-.0001,offset_s2[j]+.001])
		plt.xlim([250,5000])
		#plt.xticks([0,1000,2000,3000,4000,5000])
		plt.xticks([300,400,500,600,700,800,900,1000,2000,3000,4000,5000])
		axesS2[j].spines['bottom'].set_visible(False)
		axesS2[j].tick_params(axis='x', which='both',bottom='off', labelbottom='off')
		axesS2[j].tick_params(axis='x', which='both',top='on', labeltop='off')
		axesS2[j].tick_params(axis='y',direction='out', right='off',labelsize=14)
		axesS2[j].plot((-d,+d),(-2*d,+2*d),**kwargs)
		axesS2[j].plot((1-d,1+d), (-2*d,+2*d), **kwargs)
ytext='Normalized counts/bin'
s1ylimtext=r'X10$^{0}$'
s2ylimtext=r'X10$^{-3}$'
plt.figtext(.03,.23,ytext,fontsize=18,horizontalalignment='center',verticalalignment='center',rotation='vertical')
plt.figtext(.03,.68,ytext,fontsize=18,horizontalalignment='center',verticalalignment='center',rotation='vertical')
plt.figtext(.09,.37, s2ylimtext,fontsize=14)
plt.figtext(.09,.95, s1ylimtext,fontsize=14)
plt.savefig('Ar_3spectrum_s1_s2.png')
plt.show()

#for i in np.arange(6):
	#print(EnergyParams[i][2]/EnergyParams[i][1])
#Energy Combination plot
#colors_allHist=['r','goldenrod', 'lime','cyan', 'deepskyblue',  'magenta']
#colors_allGauss=['darkred', 'darkgoldenrod', 'darkgreen', 'darkcyan','darkblue', 'darkmagenta']
plt.figure(2)
plt.clf()
gs1=gridspec.GridSpec(3,1)
gs1.update(bottom=.135,top=.95,left=.13,right=.95,hspace=.05)
ax1=plt.subplot(gs1[2])
ax2=plt.subplot(gs1[1])
ax3=plt.subplot(gs1[0])
axesE=[ax1,ax2,ax3]
ticks_y_E=[str(0),str(.4),str(.8),str(1.2),str(1.6)]
offset_E=[0,.0016,.0032]
d=.007
kwargs=dict(transform=ax1.transAxes, color='k', clip_on=False)
chosenThree=[0,2,5]
for j in np.arange(3):
	a=chosenThree[j]
	plt.sca(axesE[j])
	aLib.stairs(binss[a]/1000, y_CE[a]+offset_E[j], color=colors_gauss[j],linewidth=3,alpha=.8)
	plt.plot(xE/1000, gaussian(xE/1000, EnergyParams[a][0], EnergyParams[a][1]/1000, EnergyParams[a][2]/1000)+offset_E[j], color='k', marker='', linestyle='-', linewidth=1.8)
	plt.plot((2.8,2.8),(0,.5),color='k',marker='',linestyle='--',linewidth=2)
	plt.xlim([0,4])
	kwargs.update(transform=axesE[j].transAxes)
	plt.grid()
	if j==0:
		plt.xlabel('Combined energy (keV)', fontsize=18)
		plt.yticks(np.arange(0,.0016,.0004),ticks_y_E)
		plt.ylim([offset_E[j],offset_E[j]+.0015])
		axesE[j].spines['top'].set_visible(False)
		axesE[j].tick_params(axis='x', which='both', bottom='on', labelbottom='on', direction='out',labelsize=14)
		axesE[j].tick_params(axis='x', which='both', top='off', labeltop='off')
		axesE[j].tick_params(axis='y',direction='out', right='off',labelsize=14)
		axesE[j].tick_params(axis='y',direction='out', right='off')
		axesE[j].plot((-d,+d),(1-2*d,1+2*d),**kwargs)
		axesE[j].plot((1-d,1+d), (1-2*d,1+2*d), **kwargs)
	elif j==1:
		plt.yticks(np.arange(.0016,.0032,.0004),ticks_y_E)
		plt.ylim([offset_E[j]-.0001,offset_E[j]+.0015])
		axesE[j].spines['top'].set_visible(False)
		axesE[j].spines['bottom'].set_visible(False)
		axesE[j].tick_params(axis='x', which='both',bottom='off', labelbottom='off')
		axesE[j].tick_params(axis='x', which='both',top='off', labeltop='off')
		axesE[j].tick_params(axis='y',direction='out', right='off',labelsize=14)
		axesE[j].tick_params(axis='y',direction='out', right='off')
		axesE[j].plot((-d,+d),(1-2*d,1+2*d),**kwargs)
		axesE[j].plot((1-d,1+d), (1-2*d,1+2*d), **kwargs)
		axesE[j].plot((-d,+d),(-2*d,+2*d),**kwargs)
		axesE[j].plot((1-d,1+d), (-2*d,+2*d), **kwargs)
	else:
		plt.yticks(np.arange(.0032,.0048,.0004),ticks_y_E)
		plt.ylim([offset_E[j]-.0001,offset_E[j]+.0015])
		axesE[j].spines['bottom'].set_visible(False)
		axesE[j].tick_params(axis='x', which='both',bottom='off', labelbottom='off')
		axesE[j].tick_params(axis='x', which='both',top='on', labeltop='off')
		axesE[j].tick_params(axis='y',direction='out', right='off',labelsize=14)
		axesE[j].tick_params(axis='y',direction='out', right='off')
		axesE[j].plot((-d,+d),(-2*d,+2*d),**kwargs)
		axesE[j].plot((1-d,1+d), (-2*d,+2*d), **kwargs)
ytext='Normalized counts/bin'
Eylimtext=r'X10$^{-3}$'
plt.figtext(.03,.55,ytext,fontsize=18,horizontalalignment='center',verticalalignment='center',rotation='vertical')
plt.figtext(.09,.945,Eylimtext,fontsize=14)
plt.savefig('CombinedEnergy.png')
plt.show()


xtickField=np.array([80,90,100,200,300,400,500,600,700,800,900,1000,2000])/1000
xtickFieldLable=['','',str(0.1),'','','','','','','','',str(1)]

#FIGURE Yields
#Akimov ionization yield point
Akimov_2800_S2=47.8
Akimov_2800_S2_Err=5.5
Akimov_field=3750/1000

print('total quanta/energy')
print('2.8 keV')
print(totalQuanta/2.8)
print(totalQuanta_Err/2.8)
print('.27 keV')
print(totalQuanta_270eV/0.27)
print(totalQuanta_270eV_Err/0.27)

#colors: blue-#1f77b4, green-#2ca02c, red-#d62728, orange-#ff7f0e, purple-#9467bd
xtickField_extended=np.array([80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000])/1000
xtickFieldLable_ext=['','',str(0.1),'','','','','','','','',str(1),'']
plt.figure(3)
plt.clf()
#totalQuanta 2.8 keV
h=plt.errorbar(field, totalQuanta/2.8, totalQuanta_Err/2.8, linestyle='', color='#1f77b4',marker='s',mec='#1f77b4',markersize=10,elinewidth=2)
plt.plot(field_fine,ntotal_06calc/2.8,linestyle='--',color='#1f77b4',linewidth=2)
#photons 2.8 keV
j=plt.errorbar(field,s1_2800eV_corr/2.8, (errorbar_s1_total_corr)/2.8, linestyle='', color='#d62728',marker='o', mec='#d62728', markersize=10, elinewidth=2,label='2.8 keV peak S1 yield')
plt.plot(field_fine,nphotons_06calc/2.8,linestyle='--',color='#d62728',linewidth=2)
#electrons 2.8 keV
k=plt.errorbar(field,s2_2800eV_corr/2.8, errorbar_s2_corr/2.8, linestyle='', color='#9467bd',marker='^',mec='#9467bd',markersize=12,elinewidth=2)
plt.plot(field_fine,nelectrons_06calc/2.8,linestyle='--',color='#9467bd', linewidth=2)
n=plt.errorbar(Akimov_field,Akimov_2800_S2,Akimov_2800_S2_Err,linestyle='', color='#9467bd', alpha=.5, marker='^', mec='#9467bd',markersize=12, elinewidth=2)
#electrons .27 keV
l=plt.errorbar(field*1.1, s2_270ev_corr/0.27, errorbar_s2_270eVcorr/0.27, linestyle='', color='#2ca02c',marker='p',mec='#2ca02c',markersize=12, elinewidth=2)
plt.plot(field_fine,nelectrons_270est_06calc/.27,linestyle='--',color='#2ca02c',linewidth=2)
#total quanta .27 keV (estimate)
#m=plt.errorbar(field*1.2, totalQuanta_270eV/0.27, totalQuanta_270eV_Err/0.27, linestyle='', color='#ff7f0e',marker='d',mec='#ff7f0e',markersize=12, elinewidth=2)
#plt.plot(field_fine,ntotal_270est_06calc/.27,linestyle='--',color='#ff7f0e',linewidth=2)
plt.xlabel('Electric field (kV/cm)', fontsize=18)
#plt.ylabel('Quanta per keV', fontsize=18)
#tq_line=mlines.Line2D([],[], color='#17478D', linestyle='',marker='.',markersize=8, label='2.8 keV peak total quanta', fontsize=12)
#s1_28_line=mlines.Line2D([],[], color='#4FB0D6', linestyle='',marker='.',markersize=8, label='2.8 keV peak S1 yield', fontsize=12)
#s2_28_line=mlines.Line2D([],[], color='#F4B23C', linestyle='',marker='.',markersize=8, label='2.8 keV peak S2 yield', fontsize=12)
#s2_027_line=mlines.Line2D([],[], color='#B26208', linestyle='',marker='.',markersize=8, label='0.27 keV peak S2 yield', fontsize=12)
#plt.legend([l,h,k,m,j],['0.27 keV peak S2 yield','2.8 keV peak total quanta', '2.8 keV peak S2 yield', 'Akimov 2.8 keV S2 yield', '2.8 keV peak S1 yield'], fontsize=14, ncol=2)
plt.xscale('log')
plt.xlim([0.08,4])
plt.ylim([15,85])
plt.xticks(xtickField_extended,xtickFieldLable_ext,fontsize=14)
yticks=[str(10),str(20),str(30),str(40),str(50),str(60),str(70),str(80)]
plt.yticks(np.arange(10,90,10),yticks,fontsize=14)
plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.12)
ytext='Quanta per keV'
plt.figtext(.03,.55,ytext,fontsize=18,horizontalalignment='center',verticalalignment='center',rotation='vertical')
plt.grid()
plt.savefig('AllYields.png')
plt.show()

#importing Points from Q. Lin paper
QLinPoints= np.loadtxt('QLinTIBDataTheif.txt',delimiter=',')
QLinField=QLinPoints[:,0]/1000
QLinTIB=QLinPoints[:,1]
QLinETop=QLinPoints[:,2]
QLinEBot=QLinPoints[:,3]
QLinE=[QLinETop,QLinEBot]
QLinA=0.046
QLinDelta=0.140

print(QLinA*(1000**(-QLinDelta)))
print(QLinDelta)
print(poptT06[0])
print(poptT06[1])

#plotting TIB parameter
plt.figure(4)
#plt.plot(field,TIB_fit(field,*poptT2), linestyle='--', linewidth=2.0, color='#17478D')
plt.plot(field,TIB_fit(field,QLinA*(1000**(-QLinDelta)), QLinDelta), linestyle=':', linewidth=3.0, color='#1f77b4', alpha=.5)
plt.plot(field,TIB_fit(field,*poptT06), linestyle='--', linewidth=3.0, color='#d62728')
#lines={'linestyle': 'None'}
#plt.rc('lines',**lines)
#h=plt.errorbar(field,TIB_2,TIB_2_Err, marker='o',markersize=7, linestyle='none', color='#17478D', mec='#17478D')
j=plt.errorbar(QLinField,QLinTIB,QLinE, marker='s',markersize=6, linestyle='none', color='#1f77b4', mec='#1f77b4',elinewidth=1.5, alpha=.5)
i=plt.errorbar(field,TIB_06,TIB_06_Err, marker='^',markersize=7, linestyle='none', color='#d62728', mec='#d62728',elinewidth=2.5)
#lime_line=mlines.Line2D([],[], color='lime', linestyle='--',linewidth=1.5, label='This Paper: '+r'$\alpha=0.2$')
#red_line=mlines.Line2D([],[], color='r', linestyle='--',linewidth=1.5, label='This Paper: '+r'$\alpha=0.06$')
#blue_line=mlines.Line2D([],[], color='deepskyblue', linestyle='--',linewidth=1.5, label='Q.Lin et al. (2015)}')
#plt.legend([i,j], ['This paper','Q.Lin et al. (2015)'], fontsize=14)
plt.subplots_adjust(left=0.135, right=0.95, top=0.95, bottom=0.11)
plt.xlabel('Electric field (kV/cm)',fontsize=18)
plt.ylabel(r'$4\xi/N_{i}$', fontsize=18)
plt.gca().set_xscale('log')
plt.xlim([0.08,2.2])
plt.xticks(xtickField,xtickFieldLable,fontsize=14)
yticks=[str(0.012),str(0.014),str(0.016),str(0.018),str(0.020),str(0.022),str(0.024)]
plt.yticks(np.arange(0.012,0.026,0.002),yticks,fontsize=14)
plt.ylim([0.012, 0.025])
plt.grid()
plt.savefig('TIB_plot.png')
plt.show()


#import NEST and tritium data
NESTdata=np.loadtxt('NEST_ChargeYield.txt')
NESTEnergy=NESTdata[:,0]
NESTChargeYield=NESTdata[:,1]

TritiumData=np.loadtxt('LUX_Dec2013_CH3T_Yields.txt', skiprows=1)
TritiumEnergy=TritiumData[:,0]
TritiumChargeYield=TritiumData[:,5]
TritiumChargeYield_Err=np.sqrt(TritiumData[:,6]**2+TritiumData[:,7]**2)

TritiumChargeYield_UpErr=TritiumChargeYield+TritiumChargeYield_Err
TritiumChargeYield_DownErr=TritiumChargeYield-TritiumChargeYield_Err
TritiumChargeYield_UpErr_savgolsmooth=savgol_filter(TritiumChargeYield_UpErr,25,3)
TritiumChargeYield_DownErr_savgolsmooth=savgol_filter(TritiumChargeYield_DownErr,25,3)

#Xenon127Energy=np.array([0.186,1.1,5.2,33.2])
#Xenon127ChargeYield=np.array([75.8,61.4,30.8,22.72])
#Xenon127ChargeYield_StatErr=np.array([4.4,0.5,0.1,0.03])
#Xenon127ChargeYield_SysErr=np.array([5.3,4.3,2.1,1.58])
#Xenon127ChargeYield_TotalErr=np.sqrt(Xenon127ChargeYield_SysErr**2+Xenon127ChargeYield_StatErr**2)

#Data from L. W. Goetzke et. al
LWG13_E=[1,3]
LWG13_Q=np.ones(2)*55.8
LWG13_QErr=np.ones(2)*np.sqrt(0.4**2+(.054*55.8)**2)
LWG35_E=[3,5]
LWG35_Q=np.ones(2)*41.6
LWG35_QErr=np.ones(2)*np.sqrt(0.2**2+(.054*41.6)**2)
LWG57_E=[5,7]
LWG57_Q=np.ones(2)*36.5
LWG57_QErr=np.ones(2)*np.sqrt(0.1**2+(.054*36.5)**2)
LWG79_E=[7,9]
LWG79_Q=np.ones(2)*34.1
LWG79_QErr=np.ones(2)*np.sqrt(0.1**2+(.054*34.1)**2)
LWG911_E=[9,11]
LWG911_Q=np.ones(2)*32.8
LWG911_QErr=np.ones(2)*np.sqrt(0.1**2+(.054*32.8)**2)
LWG1113_E=[11,13]
LWG1113_Q=np.ones(2)*31.7
LWG1113_QErr=np.ones(2)*np.sqrt(0.1**2+(.054*31.7)**2)
LWG1315_E=[13,15]
LWG1315_Q=np.ones(2)*30.7
LWG1315_QErr=np.ones(2)*np.sqrt(0.1**2+(.054*30.7)**2)
LWG1517_E=[15,17]
LWG1517_Q=np.ones(2)*30.3
LWG1517_QErr=np.ones(2)*np.sqrt(0.1**2+(.054*30.3)**2)
LWG1719_E=[17,19]
LWG1719_Q=np.ones(2)*29.7
LWG1719_QErr=np.ones(2)*np.sqrt(0.1**2+(.054*29.7)**2)
LWG1921_E=[19,21]
LWG1921_Q=np.ones(2)*29.2
LWG1921_QErr=np.ones(2)*np.sqrt(0.1**2+(.054*29.2)**2)

Ar37Energy=[0.27,2.8]
Ar37ChargeYield=[s2_270ev_corr[1]/.27, s2_2800eV_corr[1]/2.8]
Ar37ChargeYield_Err=[errorbar_s2_270eVcorr[1]/.27,errorbar_s2_corr[1]/2.8]

plt.figure(5)
plt.plot(NESTEnergy, NESTChargeYield, marker='', linestyle='-', linewidth=3, color='#ff7f0e')
plt.plot(TritiumEnergy,TritiumChargeYield, marker='s', markersize=4, fillstyle='full',linestyle='-', color='k',mec='k',mfc='w')
plt.plot(TritiumEnergy,TritiumChargeYield_UpErr_savgolsmooth, marker='', linestyle='-', color='k')
plt.plot(TritiumEnergy,TritiumChargeYield_DownErr_savgolsmooth, marker='', linestyle='-', color='k')
plt.plot(LWG13_E, LWG13_Q, marker='', linestyle='-', linewidth=2, color='#2ca02c')
plt.fill_between(LWG13_E,LWG13_Q+LWG13_QErr,LWG13_Q-LWG13_QErr, color='#2ca02c', alpha=.2)
plt.plot(LWG35_E, LWG35_Q, marker='', linestyle='-', linewidth=2, color='#2ca02c')
plt.fill_between(LWG35_E,LWG35_Q+LWG35_QErr,LWG35_Q-LWG35_QErr, color='#2ca02c', alpha=.2)
plt.plot(LWG57_E, LWG57_Q, marker='', linestyle='-', linewidth=2, color='#2ca02c')
plt.fill_between(LWG57_E,LWG57_Q+LWG57_QErr,LWG57_Q-LWG57_QErr, color='#2ca02c', alpha=.2)
plt.plot(LWG79_E, LWG79_Q, marker='', linestyle='-', linewidth=2, color='#2ca02c')
plt.fill_between(LWG79_E,LWG79_Q+LWG79_QErr,LWG79_Q-LWG79_QErr, color='#2ca02c', alpha=.2)
plt.plot(LWG911_E, LWG911_Q, marker='', linestyle='-', linewidth=2, color='#2ca02c')
plt.fill_between(LWG911_E,LWG911_Q+LWG911_QErr,LWG911_Q-LWG911_QErr, color='#2ca02c', alpha=.2)
plt.plot(LWG1113_E, LWG1113_Q, marker='', linestyle='-', linewidth=2, color='#2ca02c')
plt.fill_between(LWG1113_E,LWG1113_Q+LWG1113_QErr,LWG1113_Q-LWG1113_QErr, color='#2ca02c', alpha=.2)
plt.plot(LWG1315_E, LWG1315_Q, marker='', linestyle='-', linewidth=2, color='#2ca02c')
plt.fill_between(LWG1315_E,LWG1315_Q+LWG1315_QErr,LWG1315_Q-LWG1315_QErr, color='#2ca02c', alpha=.2)
plt.plot(LWG1517_E, LWG1517_Q, marker='', linestyle='-', linewidth=2, color='#2ca02c')
plt.fill_between(LWG1517_E,LWG1517_Q+LWG1517_QErr,LWG1517_Q-LWG1517_QErr, color='#2ca02c', alpha=.2)
plt.plot(LWG1719_E, LWG1719_Q, marker='', linestyle='-', linewidth=2, color='#2ca02c')
plt.fill_between(LWG1719_E,LWG1719_Q+LWG1719_QErr,LWG1719_Q-LWG1719_QErr, color='#2ca02c', alpha=.2)
plt.plot(LWG1921_E, LWG1921_Q, marker='', linestyle='-', linewidth=2, color='#2ca02c')
plt.fill_between(LWG1921_E,LWG1921_Q+LWG1921_QErr,LWG1921_Q-LWG1921_QErr, color='#2ca02c', alpha=.2)
plt.errorbar(Ar37Energy, Ar37ChargeYield, Ar37ChargeYield_Err, marker='s', markersize=7, linestyle='', color='#1f77b4',elinewidth=2)
plt.xlabel('Energy (keV)', fontsize=18)
#plt.ylabel('Charge yield  (electrons/keV)', fontsize=18)
plt.xscale('log')
plt.xlim([.1,20])
plt.ylim([15,90])
xticks=[str(.2),str(1),str(5),str(10),str(20)]
plt.xticks([.2,1,5,10,20],xticks,fontsize=14)
yticks=[str(20),str(30),str(40),str(50),str(60),str(70),str(80)]
plt.yticks(np.arange(20,90,10),yticks,fontsize=14)
plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11)
ytext='Charge yield  (electrons/keV)'
plt.figtext(.03,.55,ytext,fontsize=18,horizontalalignment='center',verticalalignment='center',rotation='vertical')
plt.grid()
plt.savefig('ChargeYieldvEnergy.png')
plt.show()

################################
##USEFUL CODE SNIPITS
red_line=mlines.Line2D([],[], color='r', linestyle='--',linewidth=1.5, label='2.8 keV')
blue_line=mlines.Line2D([],[], color='deepskyblue', linestyle='--',linewidth=1.5, label='0.27 keV')
plt.legend(handles=[red_line, blue_line])

#OLD WORK
## OLD WAY OF CALCULATING ENERGY - INSIDE FIRST FOR LOOP
#calculating and fitting combined energy
#	pdata2=pdata2[pdata2 != 0] #S2 area in phe
#	pdata1=pdata1[pdata2 != 0] #s1 area
#	pdata2=pdata2[np.logical_not(np.isnan(pdata1))]
#	pdata1=pdata1[np.logical_not(np.isnan(pdata1))]
#	Ecomb1=(w_fnc*(pdata1/g1+pdata2/g2[i]))
#	yE1,bintemp=np.histogram(Ecomb1, bins=1000, range=(0,4000), density=True)
#	xE=bin_to_x(bintemp)
#	hilim=4000
#	lolim=1000
#	fitx=xE[(xE>lolim)*(xE<hilim)]
#	fity1=yE1[(xE>lolim)*(xE<hilim)]
#	meanE1=sum(fitx*fity1)/np.sum(fity1)
#	sigmaE1=np.sqrt(sum(fity1*(fitx-meanE1)**2)/np.sum(fity1))
#	poptE1,pcovE1=curve_fit(gaussian,fitx,fity1,p0=[1,meanE1,sigmaE1])
#	#calculate same thing with g2+.1
#	Ecomb2=(w_fnc*(pdata1/g1+pdata2/(g2[i]+.1)))
#	yE2,bintemp=np.histogram(Ecomb2, bins=1000, range=(0,4000), density=True)
#	fity2=yE2[(xE>lolim)*(xE<hilim)]
#	meanE2=sum(fitx*fity2)/np.sum(fity2)
#	sigmaE2=np.sqrt(sum(fity2*(fitx-meanE2)**2)/np.sum(fity2))
#	poptE2,pcovE2=curve_fit(gaussian,fitx,fity2,p0=[1,meanE2,sigmaE2])
#	#successively add or subtract .1 to g2 until sigma of Ecomb gaussian is smallest
#	m=0
#	xg2=np.arange(.2,25,.1)
#	while poptE2[2]/poptE2[1]<poptE1[2]/poptE1[1] or poptE2[2]/poptE2[1]==poptE1[2]/poptE1[1]:
#		poptE1=poptE2
#		Ecomb2=(w_fnc*(pdata1/g1+pdata2/(g2[i]+xg2[m])))
#		yE2,bintemp=np.histogram(Ecomb2, bins=1000, range=(0,4000), density=True)
#		fity2=yE2[(xE>lolim)*(xE<hilim)]
#		meanE2=sum(fitx*fity2)/np.sum(fity2)
#		sigmaE2=np.sqrt(sum(fity2*(fitx-meanE2)**2)/np.sum(fity2))
#		poptE2,pcovE2=curve_fit(gaussian,fitx,fity2,p0=[1,meanE2,sigmaE2])
#		m=m+1
#	if m==0:
#		Ecomb2=(w_fnc*(pdata1/g1+pdata2/(g2[i]-.1)))
#		yE2,bin=np.histogram(Ecomb2, bins=1000, range=(0,4000), density=True)
#		fity2=yE2[(xE>lolim)*(xE<hilim)]
#		meanE2=sum(fitx*fity2)/np.sum(fity2)
#		sigmaE2=np.sqrt(sum(fity2*(fitx-meanE2)**2)/np.sum(fity2))
#		poptE2,pcovE2=curve_fit(gaussian,fitx,fity2,p0=[1,meanE2,sigmaE2])
#		while poptE2[2]/poptE2[1]<poptE1[2]/poptE1[1] or poptE2[2]/poptE2[1]==poptE1[2]/poptE1[1]:
#			poptE1=poptE2
#			Ecomb2=(w_fnc*(pdata1/g1+pdata2/(g2[i]-xg2[m])))
#			yE2,bintemp=np.histogram(Ecomb2, bins=1000, range=(0,4000), density=True)
#			fity2=yE2[(xE>lolim)*(xE<hilim)]
#			meanE2=sum(fitx*fity2)/np.sum(fity2)
#			sigmaE2=np.sqrt(sum(fity2*(fitx-meanE2)**2)/np.sum(fity2))
#			poptE2,pcovE2=curve_fit(gaussian,fitx,fity2,p0=[1,meanE2,sigmaE2])
#			m=m+1
#		if m==0:
#			g2_new[i]=g2[i]
#		elif m==1:
#			g2_new[i]=g2[i]-1
#		else:
#			g2_new[i]=g2[i]-xg2[m-2]
#	elif m==1:
#		g2_new[i]=g2[i]+.1
#	else:
#		g2_new[i]=g2[i]+xg2[m-2]
#	#Now keep the ratio between g1 and g2 the same and scale them until Ecomb is centered at 2800 keV
#	ratio=g2_new[i]/g1
#	Ecomb1=(w_fnc*(pdata1/g1+pdata2/(ratio*g1)))
#	yE1,bintemp=np.histogram(Ecomb1, bins=1000, range=(0,4000), density=True)
#	fity1=yE1[(xE>lolim)*(xE<hilim)]
#	meanE1=sum(fitx*fity1)/np.sum(fity1)
#	sigmaE1=np.sqrt(sum(fity1*(fitx-meanE1)**2)/np.sum(fity1))
#	poptE1,pcovE1=curve_fit(gaussian,fitx,fity1,p0=[1,meanE1,sigmaE1])
#	m=0
#	xg1=np.arange(.001,.1,.001)
#	poptE1storage=[]
#	while poptE1[1]<=2800:
#		g1temp=g1-xg1[m]
#		Ecomb1=(w_fnc*(pdata1/g1temp+pdata2/(ratio*g1temp)))
#		yE1,bintemp=np.histogram(Ecomb1, bins=1000, range=(0,4000), density=True)
#		fity1=yE1[(xE>lolim)*(xE<hilim)]
#		meanE1=sum(fitx*fity1)/np.sum(fity1)
#		sigmaE1=np.sqrt(sum(fity1*(fitx-meanE1)**2)/np.sum(fity1))
#		poptE1,pcovE1=curve_fit(gaussian,fitx,fity1,p0=[1,meanE1,sigmaE1])
#		poptE1storage.append(poptE1[1])
#		m=m+1
#	Ediff1=np.abs(poptE1storage[m-1]-2800)
#	Ediff2=np.abs(poptE1storage[m-2]-2800)
#	if Ediff1>Ediff2:
#		g1_new=g1-xg1[m-2]
#	else:
#		g1_new=g1-xg1[m-1]
#	g2_new[i]=ratio*g1_new
#	print('modified g1 = {0}'.format(g1_new))
#	print('modified g2[i] = {0}'.format(g2_new[i]))
#	Ecomb=(w_fnc*(pdata1/g1_new+pdata2/g2_new[i]))
#	yE,bin=np.histogram(Ecomb, bins=50, range=(0,4000), density=True)
#	xE=bin_to_x(bin)
#	hilim=4000
#	lolim=1000
#	fitx=xE[(xE>lolim)*(xE<hilim)]
#	fity=yE[(xE>lolim)*(xE<hilim)]
#	meanE=sum(fitx*fity)/np.sum(fity)
#	sigmaE=np.sqrt(sum(fity*(fitx-meanE)**2)/np.sum(fity))
#	poptE,pcovE=curve_fit(gaussian,fitx,fity,p0=[1,meanE,sigmaE])
#	y_temp,bin=np.histogram(Ecomb, bins=50,range=(0,4000))
#	y_Ecomb.append(y_temp)
#	y_EcombNorm.append(yE)
#	EnergyParams.append(poptE)

#	conf_intSigmaETemp=stats.t.interval(.95, len(Gauss2y)-1, loc=poptE[2], scale=np.sqrt(np.diag((pcovE))[2]))
#	conf_intSigmaE=poptE[2] - conf_intSigmaETemp[0]


##TROUBLESHOOTING POISSON FUNCTION
#bins=np.arange(1,10,.2)
#a=[6,.5]
#sc=1/(bins[1]-bins[0])
#x=np.arange(len(bins))
#p0=poisson(x,a[0])
#xs=x*a[1]
#c=np.cumsum(p0)
#binse=bins+(bins[1]-bins[0])/2
#f=ter.interp1d(x,p0,'quadratic')
#p1=f(binse)
#print(p1)
#print(np.diff(p1))
#p=sc*np.diff(p1)
#print(p)
#plt.figure(0)
#plt.plot(x,p0, color='k')
#plt.plot(x,c, color='b')
#plt.plot(xs,c, color='c')
#plt.plot(binse,p1,color='r')
#plt.plot(bins[1:]+.5,p,color='g')
#plt.xlim([0,20])
#plt.show()

##TROUBLESHOOTING SMEAREDPOISSON FUNCTION
#plt.figure(0)
#lamb=20
#sigma=1
#cutoff_lo=15
#cutoff_hi=25
#poisson_dist=np.random.poisson(lamb,10000)
#binNum=max(poisson_dist)-min(poisson_dist)
#p,bins=np.histogram(poisson_dist, bins=binNum, range=(min(poisson_dist)-0.5, max(poisson_dist)-.5), normed=True)
#k=bin_to_x(bins)
#sum_hist=0
#for i in np.arange(cutoff_lo,cutoff_hi,1):
#	sum_hist=sum_hist+p[i]
#print(sum_hist)
#popt,pcov=curve_fit(SmearedPoisson,k,p,p0=[sum_hist,sigma,lamb])
#p_cutoff=p[(k>cutoff_lo)*(k<cutoff_hi)]
#k_cutoff=k[(k>cutoff_lo)*(k<cutoff_hi)]
#bins_cutoff=x_to_bin(k_cutoff)
#popt_cutoff,pcov_cutoff = curve_fit(SmearedPoisson,k_cutoff,p_cutoff,p0=[sum_hist,sigma,lamb])
#aLib.stairs(bins,p,linewidth=2,color='k')
#plt.plot(k,SmearedPoisson(k, *popt), linestyle='-',marker='',color='b')
#sum_SPwhole=sum(SmearedPoisson(k, *popt))
#print(sum_SPwhole)
#print(popt)
#aLib.stairs(bins_cutoff,p_cutoff,linewidth=4,color='k')
#plt.plot(k,SmearedPoisson_NewRange(k, 1, *popt_cutoff), linestyle='-', marker='', color='g')
#plt.plot(k_cutoff,SmearedPoisson(k_cutoff, *popt_cutoff), linestyle='-', marker='', color='r')
#sum_SPpartialpartial=sum(SmearedPoisson(k_cutoff,*popt_cutoff))
#print(sum_SPpartialpartial)
#sum_SPpartial=sum(SmearedPoisson(k,*popt_cutoff))
#print(sum_SPpartial)
#print(popt_cutoff)
#plt.show()


##TROUBLESHOOTING GAUSS_AREAOFPOISSON FUNCTION
#plt.figure(0)
#lamb=5
#sigma=1
#k_values=np.arange(12)
#plt.plot(k_values, poisson(k_values,lamb), linestyle='', marker='.', markersize=10)
#x_values=np.arange(-5,17,.1)
#sum_gauss=np.zeros(len(x_values))
#for k in k_values:
#	sum_gauss=sum_gauss+gauss_AreaOfPoisson(x_values,k,sigma,lamb)
#	plt.plot(x_values,gauss_AreaOfPoisson(x_values,k,sigma,lamb), linestyle='-', marker='', color='k')
#plt.plot(x_values,sum_gauss, linestyle='-', marker='', color='r')
#x_values2=np.arange(min(k_values)-5,max(k_values)+5,.1)
#plt.plot(x_values,SmearedPoisson(x_values,sigma,lamb), linestyle='-', marker='', color='g')
#plt.xlabel('k')
#plt.ylabel('S2 Yield (e^{-} / keV)')
#plt.xscale('log')
#plt.xlim([80,2200])
#plt.ylim([42,80])
#plt.grid()
#plt.show()

#TROUBLESHOOTING CHISQUARED

#for i in np.arange(3):
#	sumYS1=0
#	sumYS1_GF=0
#	for j in np.arange(len(yy[i])):
#		sumYS1=sumYS1+binWidth*y_s1[i][j]
#	for j in np.arange(len(yy_GF[i])):
#		sumYS1_GF=sumYS1_GF+binWidth*y_s1_GF[i][j]
#	plt.figure(10+i)
#	aLib.stairs(bin_GF,sumYS1*yy_GF[i]*logistic(x_GF,ThreshFitParams[i][0],ThreshFitParams[i][1],ThreshFitParams[i][2]),color='r')
#	aLib.stairs(bin_GF,y_s1_GF[i],color='k')
#	print(y_s1_GF[i]-sumYS1*yy_GF[i]*logistic(x_GF,ThreshFitParams[i][0],ThreshFitParams[i][1],ThreshFitParams[i][2]))
#	for l in np.arange(len(changex0)):
#		NewCurve=yy_GF[i]*logistic(x_GF,ThreshFitParams[i][0],ThreshFitParams[i][1],ThreshFitParams[i][2]-changex0[l])
#		if i==0:
#			chisquared1=np.append(chisquared1,sum((y_s1_GF[i]-sumYS1*NewCurve)**2/(sumYS1*NewCurve)))
#			#chisquared1=np.append(chisquared1,sum((y_s1_GF[i]-NewCurve)**2/(NewCurve)))
#		elif i==1:
#			chisquared2=np.append(chisquared2,sum((y_s1_GF[i]-sumYS1*NewCurve)**2/(sumYS1*NewCurve)))
#			#chisquared2=np.append(chisquared2,sum((y_s1_GF[i]-NewCurve)**2/(NewCurve)))
#		else:
#			chisquared3=np.append(chisquared3,sum((y_s1_GF[i]-sumYS1*NewCurve)**2/(sumYS1*NewCurve)))
#			#chisquared3=np.append(chisquared3,sum((y_s1_GF[i]-NewCurve)**2/(NewCurve)))



#plt.plot(x,yy[i], color='k')
#plt.plot(x,NewCurve1, color='b')
#plt.plot(x,NewCurve2, color='r')
#plt.plot(x,NewCurve3, color='g')
#fitx_NC=x[(x>lolim)*(x<hilim)]
#fity_NC=NewCurve1[(x>lolim)*(x<hilim)]
#sumNewCurveFit=0
#xbin=x[1]-x[0]
#for m in np.arange(len(fitx_NC)):
#	sumNewCurveFit=sumNewCurveFit+NewCurve1[n]*xbin
#popt,pcov=curve_fit(SmearedPoisson,x[(x>lolim)*(x<hilim)],NewCurve1[(x>lolim)*(x<hilim)], p0=[sumNewCurveFit,2.7,8])
#plt.plot(x,SmearedPoisson_NewRange(x,1,*popt), color='g')
#print(popt)
#plt.show()