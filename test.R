## tese
nobs <- 500
p <- 500
x <- array(rnorm(nobs*p), c(nobs, p))
beta_true <- c(1,-1,0.5,-0.5, rep(0, times=p-4))
y <- apply(x, 1, function(t){rbinom(1,1, prob = exp(t%*%beta_true)/ (1+exp(t%*%beta_true)))})
fit <- cv.cLearn(x, y, weight = c(0, 0, 0, 0, rep(1, times= p-4)), lossType = 'logistic', tol = 1e-3)
fit$fit$coef[,fit$lambda.seq==fit$lambda.opt]
