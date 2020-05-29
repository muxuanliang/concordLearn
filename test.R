## test
nobs <- 500
p <- 500
x <- array(rnorm(nobs*p), c(nobs, p))
beta_true <- c(1,-1,0.5,-0.5, rep(0, times=p-4))
y <- apply(x, 1, function(t){rbinom(1,1, prob = exp(t%*%beta_true)/ (1+exp(t%*%beta_true)))})

## fit using smoothed_hinge
system.time(fit <- cInfer(x, y, weight = c(0, 0, 1, 1, rep(1, times= p-4)), lossType = 'smoothed_hinge', tol = 1e-3, parallel = FALSE))
system.time(fit <- cInfer(x, y, weight = c(0, 0, 1, 1, rep(1, times= p-4)), lossType = 'smoothed_hinge', tol = 1e-3))
## estimated coef
fit$coef

## estimated coef of covariate of interests
fit$coef[fit$indexToTest]

## pvalues
fit$pvalue

## adjusted estimates and sd estimation, which is asymptotic normal
fit$coefAN
1.96*fit$sigmaAN/sqrt(nobs)

#2* 1.96*fit$sigmaAN/sqrt(nobs) / abs(fit$coefAN)

## fit using logistic
system.time(fit <- cInfer(x, y, weight = c(0, 0, 1, 1, rep(1, times= p-4)), lossType = 'logistic', tol = 1e-3, parallel = FALSE))
system.time(fit <- cInfer(x, y, weight = c(0, 0, 1, 1, rep(1, times= p-4)), lossType = 'logistic', tol = 1e-3))
## estimated coef
fit$coef

## estimated coef of covariate of interests
fit$coef[fit$indexToTest]

## pvalues
fit$pvalue

## adjusted estimates and sd estimation, which is asymptotic normal
fit$coefAN
1.96*fit$sigmaAN/sqrt(nobs)

#2* 1.96*fit$sigmaAN/sqrt(nobs) / abs(fit$coefAN)
