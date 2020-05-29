# cLearn takes data=list(X, Y) as input and output beta given a single lambda, weight, gamma
# X is the covariates, Y is the label, in clearn we do not incorperate the intercept but X can incoorperate the intercept.
cLearn <- function(x, y, lambdaSeq = NULL, weight = rep(1, NCOL(x)), lossType='logistic', nlambda = 100, ratio  = 10/11, tol = 1e-4, maxIter = 10^3, ...){
  # check wether only 2 classes
  stopifnot(length(levels(factor(y))) == 2)
  # set y if y is not 1 or -1
  if (as.numeric(levels(factor(y))[1]) != -1){
    y[factor(y)==levels(factor(y))[1]] <- -1
  }
  if (as.numeric(levels(factor(y))[2]) != 1){
    y[factor(y)==levels(factor(y))[2]] <- 1
  }
  # set lambda
  if (is.null(lambdaSeq)){
    lambda_max <- max(abs(apply(x, 2, function(t){mean(t * y)}) * loss(0, type = lossType, order = 1, ...)))
    lambdaSeq <- lambda_max * (ratio)^(1:nlambda)
  } else {
    lambdaSeq <- sort(lambdaSeq, decreasing = TRUE)
  }

  # fit for an increasing seq of lambda and use the previous estimation as a warm start
  coef <- NULL
  betaInit <- c(rep(0.1, NCOL(x)))
  for (lambda in lambdaSeq){
    fit <- solver(x=x, y=y, lambda=lambda, betaInit =  betaInit, weight = weight, lossType=lossType, tol = tol, maxIter = maxIter, ...)
    betaInit <- fit$coef
    coef <- cbind(coef, fit$coef)
  }
  out <- (list(coef=coef, lambda=lambdaSeq))
  class(out) <- "cLearn"
  out
}
