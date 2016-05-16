import Quandl
import pandas as pd
from numpy import *
import statsmodels.api as sm
import matplotlib.pylab as plt
import plotly.graph_objs as go
import plotly as py
import cufflinks as cf
import datetime as dt
import seaborn as sns
from sklearn import linear_model
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import scipy
import os
import pywt


maturities = asarray([3, 6, 9, 12, 15, 18,21, 24, 30, 36, \
                     48, 60, 72, 84, 96, 108, 120])

beta_names = ['beta1', 'beta2', 'beta3']
lam_t = .0609
_load2 = lambda x: (1.-exp(-lam_t*x)) / (lam_t*x)
_load3 = lambda x: ((1.-exp(-lam_t*x)) / (lam_t*x)) - \
exp(-lam_t*x)

def loadData():
	# load datasets + define parameters 
	lam_t = .0609

	# filter where we only get the last day of every month
	tau = ['Date', '1 MO',  '3 MO',  '6 MO',  '9 MO', '12 MO', '15 MO', '18 MO', '21 MO',  \
	               '24 MO', '30 MO', '36 MO', '48 MO', '60 MO', '72 MO', '84 MO', '96 MO', \
	               '108 MO', '120MO']

	__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
	loc = os.path.join(__location__ + '/Data/FBFitted.csv')
	ratedata = pd.read_csv(loc, index_col=0)

	_load2 = lambda x: (1.-exp(-lam_t*x)) / (lam_t*x)
	_load3 = lambda x: ((1.-exp(-lam_t*x)) / (lam_t*x)) - \
	exp(-lam_t*x)
	ratedata = ratedata.drop('1', axis=1)

	# convert ratedatea indices into datetiems
	ratedata.index = pd.to_datetime(ratedata.index, format='%Y%m%d')

	# Diebold and Li only use Jan 1980 to Dec 2000
	start_idx = ratedata.index.get_loc(dt.datetime.strptime('1985-01-31', '%Y-%m-%d'))
	ratedata = ratedata.iloc[start_idx:]

	# make 2D matrix of all of the beta coeff. for each maturity
	X = zeros((len(maturities), 2))
	# X[:,0] = sm.add_constant(ones(len(maturities)))
	X[:,0] = _load2(maturities)
	X[:,1] = _load3(maturities)
	X = sm.add_constant(X)

	# for each observation, fit the maturity curve 
	beta_fits = zeros((len(ratedata), 3))
	residuals = zeros((len(ratedata), 17))
	for i in range(0, len(ratedata)):
	    model = sm.regression.linear_model.OLS(ratedata.iloc[i], X)
	    results = model.fit()
	    beta_fits[i,:3] = results.params
	    residuals[i,:] = results.resid
	    
	# convert into a dataframe for conv. 
	beta_fits = pd.DataFrame(beta_fits, columns=beta_names)
	residuals = pd.DataFrame(residuals, columns=[str(mat) for mat in maturities])

	beta_fits.index = ratedata.index
	residuals.index = ratedata.index

	return beta_fits, residuals, ratedata


def table2(residuals):
	table2 = zeros((len(maturities), 9)) # initialize a matrix
	table2 = pd.DataFrame(table2, index=[str(mat) for mat in maturities])
	table2.columns = ['Mean', 'Std', 'Min', 'Max', 'MAE', 'RMSE', 'ACF(1)', 'ACF(12)', 'ACF(30)']
	for mat in maturities:
		table2.ix[str(mat), 0] = residuals.ix[:,str(mat)].mean()
		table2.ix[str(mat), 1] = residuals.ix[:,str(mat)].std()
		table2.ix[str(mat), 2] = residuals.ix[:,str(mat)].min()
		table2.ix[str(mat), 3] = residuals.ix[:,str(mat)].max()
		table2.ix[str(mat), 4] = abs(residuals.ix[:,str(mat)]).mean() # MAE
		table2.ix[str(mat), 5] = sqrt(pow(residuals.ix[:,str(mat)],2).mean())
		table2.ix[str(mat), 6] = sm.tsa.stattools.acf(residuals.ix[:,str(mat)], nlags=31)[1]
		table2.ix[str(mat), 7] = sm.tsa.stattools.acf(residuals.ix[:,str(mat)], nlags=31)[13]
		table2.ix[str(mat), 8] = sm.tsa.stattools.acf(residuals.ix[:,str(mat)], nlags=31)[-1]

	return table2


def table3(beta_fits):
	table3 = pd.DataFrame(zeros((3, 8)), index=beta_names)
	table3_columns = ['Mean', 'Std', 'Min', 'Max', 'ACF(1)', 'ACF(12)', 'ACF(30)', 'ADF']
	table3.columns = table3_columns
	for beta in beta_names:
		table3.ix[beta, 0] = beta_fits.ix[:,beta].mean()
		table3.ix[beta, 1] = beta_fits.ix[:,beta].std()
		table3.ix[beta, 2] = beta_fits.ix[:,beta].min()
		table3.ix[beta, 3] = beta_fits.ix[:,beta].max()
		table3.ix[beta, 4] = sm.tsa.stattools.acf(beta_fits.ix[:, beta], nlags=31)[1]
		table3.ix[beta, 5] = sm.tsa.stattools.acf(beta_fits.ix[:, beta], nlags=31)[13]
		table3.ix[beta, 6] = sm.tsa.stattools.acf(beta_fits.ix[:, beta], nlags=31)[-1]
		table3.ix[beta, -1] = sm.tsa.adfuller(beta_fits.ix[:,beta])[0] # note the ADF assumes [maxlag = 12*(nobs/100)^.25]

	return table3


def table4(forecast, actual):
	idx_1994 = actual.index.get_loc(dt.datetime.strptime('1994-01-31', '%Y-%m-%d'))
	idx_2000 = actual.index.get_loc(dt.datetime.strptime('2000-12-29', '%Y-%m-%d'))

	err =  actual.ix[idx_1994:idx_2000,:] - forecast
	table = pd.DataFrame(zeros((5, 5)), index=['3', '12' ,'36', '60', '120'], \
    	columns=['mean', 'std. dev.', 'RMSE', 'ACF(1)', 'ACF(12)'])
    
	for idx in table.index:
		table.ix[idx, 'mean' ] = err.ix[:,idx].mean()
		table.ix[idx, 'std. dev.' ] = err.ix[:,idx].std()
		table.ix[idx, 'RMSE' ] = sqrt(pow(err.ix[:,idx],2).mean())
		table.ix[idx, 'ACF(1)' ] = sm.tsa.acf(err.ix[:, idx])[1]
		table.ix[idx, 'ACF(12)' ] = sm.tsa.acf(err.ix[:, idx])[13]


	return table


def yieldContors(ratedata):

	data = [
	    go.Contour(
	        z = ratedata.as_matrix(),
	        x = ratedata.columns, 
	        y = ratedata.index)
	    ]

	layout = go.Layout(
	    title='Yields vs. Maturities',
	    width=640,
	    height=480,
	                xaxis=dict(
	                title='Maturity (months)',
	                titlefont=dict(
	                    size=16)
	                ),  
	            yaxis=dict(
	                title='Date',
	                titlefont=dict(
	                    size=16)
	                )


	    )

	fig = go.Figure(data=data, layout=layout)

	return fig


def exampleYield(ratedata, loc):
	if type(loc) != list:
		print('You must input a list')

	if len(loc) ==1:
		tit = str('Fig 3: '+ ratedata.index[loc[0]])
	else:
		tit = "Sample Yield Curves"
	layout = go.Layout(
            width=640,
            height=480,
            title=tit,
            titlefont=dict(
                size=24),
    
            xaxis=dict(
                title='Maturity (months)',
                titlefont=dict(
                    size=20)
                ),  
            yaxis=dict(
                title='Yield (percent)',
                titlefont=dict(
                    size=20)
                ),
        
            legend=dict(
                font=dict(
                    size=12))

                )

	trace1 = go.Scatter(
		y = ratedata.ix[loc[0], :], 
		x = ratedata.index)

	trace2 = go.Scatter(
		y = ratedata.ix[loc[1], :], 
		x = ratedata.index)

	trace3 = go.Scatter(
		y = ratedata.ix[loc[2], :], 
		x = ratedata.index)

	data = [trace1, trace2, trace3]

            
	return go.Figure(data=data, layout=layout)


def beta_resid(residuals):
	resid_interest = residuals.ix[:,['3','6', '12', '24', '60', '120']]
	layout = go.Layout(
				title='Residuals for selected maturities (months)',
	            titlefont=dict(
	                size=18),
	            legend= dict(
	                font=dict(
	                    size=16)),
	#             title='Residuals for selected maturity periods',
	            width=640,
	            height=480,
	            )


	trace1 = go.Scatter(
		y = resid_interest.iloc[0, :], 
		x = resid_interest.index, 
		name='3 MO')


	trace2 = go.Scatter(
		y = resid_interest.iloc[1, :], 
		x = resid_interest.index, 
		name = '6 MO')

	trace3 = go.Scatter(
		y = resid_interest.iloc[2, :], 
		x = resid_interest.index,
		name= '12 MO')

	trace4 = go.Scatter(
		y = resid_interest.iloc[3, :], 
		x = resid_interest.index,
		name='24 MO')

	trace5 = go.Scatter(
		y = resid_interest.iloc[4, :], 
		x = resid_interest.index, 
		name='60 MO')

	trace6 = go.Scatter(
		y = resid_interest.iloc[5, :], 
		x = resid_interest.index, 
		name='120 MO')


	fig = py.tools.make_subplots(rows=3, cols=2, print_grid=False)
	fig.append_trace(trace1,1,1)
	fig.append_trace(trace2, 2,1)
	fig.append_trace(trace3, 3,1)
	fig.append_trace(trace4, 1,2)
	fig.append_trace(trace5, 2,2)
	fig.append_trace(trace6, 3,2)

	fig['layout'].update(layout)
	return fig

def beta_dist(beta_fits):
	fig, axes = plt.subplots(1,3, figsize=(10,7))
	fig.suptitle('Fitted Parameters Histogram')
	sns.set(font_scale=1)
	d = sns.distplot(beta_fits.ix[:,'beta1'], ax=axes[0])
	d = sns.distplot(beta_fits.ix[:,'beta2'], ax=axes[1])
	d = sns.distplot(beta_fits.ix[:,'beta3'], ax=axes[2])

	return fig


def fig7(ratedata, beta_fits):
	beta1_hat = ratedata.ix[:, '120']
	beta2_hat = ratedata.ix[:,'120'] - ratedata.ix[:,'3']
	beta3_hat = 2*ratedata.ix[:,'24'] - \
		(ratedata.ix[:,'120'] + ratedata.ix[:,'3'])

	layout = go.Layout(
				title='Residuals for selected maturities (months)',
	            titlefont=dict(
	                size=18),
	            legend= dict(
	                font=dict(
	                    size=16)),
	            width=640,
	            height=480,
	            )


	fig = py.tools.make_subplots(rows=3, 
		shared_xaxes=True, print_grid=False,subplot_titles=(
			'Level', 'Slope', 'Curvature' ))

	beta1a = go.Scatter(
			x = beta1_hat.index, 
			y = beta1_hat, 
			name = 'Emprical')

	beta1b = go.Scatter(
			x = beta_fits.index,
			y = beta_fits.ix[:,'beta1'], 
			name='Fitted')

	beta2a = go.Scatter(
			x = beta2_hat.index, 
			y = beta2_hat, 
			name = 'Emprical')

	beta2b = go.Scatter(
			x = beta_fits.index,
			y = -beta_fits.ix[:,'beta2'], 
			name='Fitted')

	beta3a = go.Scatter(
			x = beta3_hat.index, 
			y = beta3_hat, 
			name = 'Emprical')

	beta3b = go.Scatter(
			x = beta_fits.index,
			y = .3*beta_fits.ix[:,'beta3'], 
			name='Fitted')

	fig.append_trace(beta1a,1,1)
	fig.append_trace(beta1b, 1,1)

	fig.append_trace(beta2a,2,1)
	fig.append_trace(beta2b, 2,1)

	fig.append_trace(beta3a,3,1)
	fig.append_trace(beta3b, 3,1)

	fig['layout'].update(layout)
	return fig


def ACF_beta(beta_fits, fitted_resid):
	titlefont = {'fontsize': 14}
	fig = plt.figure(figsize=(20,12))
	ax1 = fig.add_subplot(321)
	f = sm.graphics.tsa.plot_acf(beta_fits.ix[:,0].values.squeeze(),\
                             lags=60, ax=ax1)
	ax1.set_title('ACF for Level', **titlefont)

	ax2 = fig.add_subplot(322)
	f = sm.graphics.tsa.plot_pacf(beta_fits.ix[:,0].values.squeeze(),\
	                             lags=60, ax=ax2)
	ax2.set_title('PACF for Level', **titlefont)

	ax3 = fig.add_subplot(323)
	f = sm.graphics.tsa.plot_acf(beta_fits.ix[:,1].values.squeeze(),\
	                             lags=60, ax=ax3)
	ax3.set_title('ACF for Slope', **titlefont)

	ax4 = fig.add_subplot(324)
	f = sm.graphics.tsa.plot_pacf(beta_fits.ix[:,1].values.squeeze(),\
	                             lags=60, ax=ax4)
	ax4.set_title('PACF for Slope', **titlefont)

	ax5 = fig.add_subplot(325)
	f = sm.graphics.tsa.plot_acf(beta_fits.ix[:,2].values.squeeze(),\
	                             lags=60, ax=ax5)
	ax5.set_title('ACF for Curvature', **titlefont)

	ax6 = fig.add_subplot(326)
	f = sm.graphics.tsa.plot_pacf(beta_fits.ix[:,2].values.squeeze(),\
	                             lags=60, ax=ax6)
	ax6.set_title('PACF for Curvature', **titlefont)

	fig.tight_layout()


def ARforecast(ratedata, beta_fits):
	# clip the data to only go to 1994
	idx_1994 = ratedata.index.get_loc(dt.datetime.strptime('1994-01-31', '%Y-%m-%d'))
	idx_2000 = ratedata.index.get_loc(dt.datetime.strptime('2000-12-29', '%Y-%m-%d'))

	# from the CWT, predict each level using an AR model 
	N_out = idx_2000 - idx_1994 # N out of sample

	beta_predict = pd.DataFrame(zeros((N_out, 3)), \
	    index=beta_fits.index[idx_1994:idx_2000], columns=beta_fits.columns)

	yield_forecast = pd.DataFrame(zeros((N_out, len(ratedata.columns))), \
	    index=beta_fits.index[idx_1994:idx_2000], columns=ratedata.columns)

	beta_predict_nieve = pd.DataFrame(zeros((N_out, 3)), \
	    index=beta_fits.index[idx_1994:idx_2000], columns=beta_fits.columns)

	yield_forecast_nieve = pd.DataFrame(zeros((N_out, len(ratedata.columns))), \
	    index=beta_fits.index[idx_1994:idx_2000
	                         ], columns=ratedata.columns)

	def perDone(i, length, goal):
	    if i != 0:
	        if (float(i)/length) *100 > goal:
	            print("{}% done".format(goal))
	            return 10.;
	        else:
	            return 0
	    else: return 0
	    
	# we want to save all the fits so we can analyze the noise spectrum
	saveRuns =[]
	# for each date in the withheld series
	d = 10.
	wavelet = 'db2'
	i = 0;
	for date in range(0, N_out):
	    d_updt = perDone(date, N_out, d)
	    d = d_updt + d
	    now = idx_1994+date # step each turn to fit

	    for beta in beta_fits.columns:
	        testb = beta_fits.ix[i:(now),beta]
	        coeff = pywt.swt(testb, wavelet)

	        # coeff will return a list of [level, 2, len(data)]
	        # each of the levels will have the trend and stochiastic componet
	        # so we will predict the trend via AR and add a random var 

	        
	        # make the nieve AR forecast
	        model = sm.tsa.AR(beta_fits.ix[:now,beta]).fit(maxlag=1, \
	                                                       method='cmle')
	        # the len of the data must be even, so shift forward one
	        try:
	            beta_predict_nieve.ix[date, beta] = \
	            model.predict(len(beta_fits.ix[:now,beta])-1\
	                          ,len(beta_fits.ix[:now,beta])).iloc[-1]
	        except KeyError:
	            pdb.set_trace()
	        
	        # for each of the levels in wavelet 
	        for level in coeff:
	            for detail in range(0,len(level)):
	                model = sm.tsa.AR(level[detail])# fit the model
	                params = model.fit(maxlag=1, method='cmle').params
	                # predict the next beta 
	                prediction = model.predict(params, len(level[detail])-1, len(level[detail]))[-1]
	                # roll append the predicted data and roll forward
	                level[detail][:] = hstack((level[detail][1:], prediction))
	       
	        # reconstruct the og signal from the wavlet
	        try:
	            pred = []
	            N_lvl = shape(coeff)[0]
	            # starts with N, so counting back all the way to sqrt(2)
	            for t in range(0, N_lvl):
	                try:
	#                     pdb.set_trace()
	                    # make the stochasitic shock
	                    pred.append((coeff[t][0][-1] + (coeff[t][1][-5])) / sqrt(2)**(N_lvl-t))
	#                     pred.append((coeff[t][0][-1] + coeff[t][1][-1]) / sqrt(2)**(N_lvl-t))




	                except IndexError:
	                    pdb.set_trace()
	                    
	            beta_predict.ix[date,beta] = mean(pred)
	        except ValueError:
	            pdb.set_trace()
	    # forecast the yields at specific maturities 
	    try:
	        yield_forecast.ix[date,:] = beta_predict.ix[date, 'beta1'] + \
	            beta_predict.ix[date, 'beta2']*_load2(asarray(maturities)) +\
	            beta_predict.ix[date, 'beta3']*_load3(asarray(maturities))

	        yield_forecast_nieve.ix[date,:] = beta_predict_nieve.ix[date, 'beta1'] + \
	            beta_predict_nieve.ix[date, 'beta2']*_load2(asarray(maturities)) +\
	            beta_predict_nieve.ix[date, 'beta3']*_load3(asarray(maturities))
	            
	        saveRuns.append(coeff)
	    except TypeError:
	        pdb.set_trace()
	        
	    i = i +1


	return yield_forecast_nieve, yield_forecast, saveRuns

















