def CCnorm(R, yhat):
    # function [CCnorm, CCabs, CCmax] = calc_CCnorm(R,yhat)
    #
    # This function returns the absolute correlation coefficient CCabs and the
    # normalized correlation coefficient CCnorm. The normalization discounts
    # the inherent inter-trial variability of the neural data provided in R.
    #
    # Inputs:
    #
    # R		should be a NxT matrix (N: number of trials, T: number of time
    #       bins) in which each row is the spike train of a given trial. Each
    #       element R(n,t) thus contains the number of spikes that were
    #       recorded during time bin t in trial n.
    #
    # yhat	should be a vector with T elements. It contains the predicted
    #       firing rate over time.


    import numpy as np
    # Check inputs
    (N, T) = R.shape
    assert T > N, 'Please provide R as a NxT matrix.'
    assert len(yhat) == T, 'The prediction yhat should have as many time bins as R.'
    yhat = yhat.ravel()
    y = np.nanmean(R, axis=0)

    # Precalculate basic values for efficiency
    Ey = np.nanmean(y)							    # E[y]
    Eyhat = np.nanmean(yhat)						# E[yhat]
    Vy = np.nansum((y-Ey)**2)/(T-1)			        # Var(y)
    Vyhat = np.nansum((yhat-Eyhat)**2)/(T-1)           # Var(yhat)
    Cyyhat = np.nansum((y-Ey) * (yhat-Eyhat))/(T-1)    # Cov(y,yhat)

    # Calculate signal power (see [1])
    SP = (np.nanvar(np.nansum(R, axis=0), ddof=1)-np.nansum(np.nanvar(R, axis=1, ddof=1)))/(N*(N-1))
    if SP <= 0:
        negSP = True
    else:
        negSP = False

    # Calculate CC values
    CC_abs = Cyyhat/np.sqrt(Vy*Vyhat)
    CC_norm = Cyyhat/np.sqrt(SP*Vyhat)
    CC_max = np.sqrt(SP/Vy)

    if negSP:
        print('Signal power estimate is negative or zero, so CCmax and CCnorm')
        print('cannot be calculated. This happens if the neural data is very')
        print('noisy, i.e. not driven by the stimulus at all.')
        CC_norm = np.NaN
        CC_max = 0

    return (CC_norm, CC_abs, CC_max, SP)


def CCnorm_jackknife(R, yhat, minSP=0.02):
    # Get distributions of CC_norm, CC_abs, CC_max using jackknife
    # (i.e. excluding trials in turn, one at a time) in order to
    # estimate the confidence intervals of each metric.
    #
    # Written by Ben Willmore (benjamin.willmore@dpag.ox.ac.uk), 2016
    # Converted to Python by Michael Oliver (michael.d.oliver@gmail.com), 2016


    import numpy as np
    # Check inputs
    (N, T) = R.shape
    assert T > N, 'Please provide R as a NxT matrix.'
    assert len(yhat) == T, 'The prediction yhat should have as many time bins as R.'
    yhat = yhat.ravel()

    CC_norm = np.zeros(N)
    CC_abs = np.zeros(N)
    CC_max = np.zeros(N)
    SP = np.zeros(N)
    for ii in range(N):
        idx = np.setdiff1d(np.arange(N), ii)
        CC_norm[ii], CC_abs[ii], CC_max[ii], SP[ii] = CCnorm(R[idx, :], yhat, minSP)

    CC_norm_out = {}
    CC_abs_out = {}
    CC_max_out = {}
    SP_out = {}

    CC_norm_out['values'] = CC_norm
    CC_norm_out['median'] = np.median(CC_norm)

    if np.any(np.isnan(CC_norm)):
        CC_norm_out['percentile_5'] = np.NaN
        CC_norm_out['percentile_95'] = np.NaN
    else:
        CC_norm_out['percentile_5'] = np.percentile(CC_norm, 5)
        CC_norm_out['percentile_95'] = np.percentile(CC_norm, 95)

    CC_abs_out['values'] = CC_abs
    CC_abs_out['median'] = np.median(CC_abs)
    CC_abs_out['percentile_5'] = np.percentile(CC_abs, 5)
    CC_abs_out['percentile_95'] = np.percentile(CC_abs, 95)

    CC_max_out['values'] = CC_max
    CC_max_out['median'] = np.median(CC_max)
    CC_max_out['percentile_5'] = np.percentile(CC_max, 5)
    CC_max_out['percentile_95'] = np.percentile(CC_max, 95)

    SP_out['values'] = SP
    SP_out['median'] = np.median(SP)
    SP_out['percentile_5'] = np.percentile(SP, 5)
    SP_out['percentile_95'] = np.percentile(SP, 95)

    return [CC_norm_out, CC_abs_out, CC_max_out, SP_out]


def CCmax(R):
    # function [CCmax, SP] = calc_CCmax(R)
    #
    # This function returns the max correlation coefficient CCmax
    #
    # Inputs:
    #
    # R     should be a NxT matrix (N: number of trials, T: number of time
    #       bins) in which each row is the spike train of a given trial. Each
    #       element R(n,t) thus contains the number of spikes that were
    #       recorded during time bin t in trial n.
    #

    import numpy as np
    # Check inputs
    (N, T) = R.shape
    assert T > N, 'Please provide R as a NxT matrix.'


    y = np.nanmean(R, axis=0)
    # Precalculate basic values for efficiency
    Ey = np.nanmean(y)                              # E[y]
    Vy = np.nansum((y-Ey)**2)/(T-1)                 # Var(y)


    # Calculate signal power (see [1])
    SP = (np.nanvar(np.nansum(R, axis=0), ddof=1)-np.nansum(np.nanvar(R, axis=1, ddof=1)))/(N*(N-1))
    if SP <= 0:
        negSP = True
    else:
        negSP = False


    # Calculate CC values
    CC_max = np.sqrt(SP/Vy)

    if negSP:
#         print('Signal power estimate is negative or zero, so CCmax')
#         print('cannot be calculated. This happens if the neural data is very')
#         print('noisy, i.e. not driven by the stimulus at all.')
        CC_max = 0

    return (CC_max, SP)