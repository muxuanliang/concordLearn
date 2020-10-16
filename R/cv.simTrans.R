# cv.cLearn utilizes cross-validation to tuning lambda
cv.simTrans <- function(x, y, lambdaSeq = NULL, weight = rep(1, NROW(x)), lossType='logistic', nlambda = 20, nfolds=5, parallel = TRUE,ratio  = 10/11, tol = 1e-4, maxIter = 10^3, intercept=FALSE, offset=NULL, ...){
  # check wether only 2 classes
  stopifnot(length(levels(factor(y))) == 2)
  nobs <- length(y)
  p <- NCOL(x)
  # set y if y is not 1 or -1
  if (as.numeric(levels(factor(y))[1]) != -1){
    y[factor(y)==levels(factor(y))[1]] <- -1
  }
  if (as.numeric(levels(factor(y))[2]) != 1){
    y[factor(y)==levels(factor(y))[2]] <- 1
  }
  if (is.null(offset)){
    offset <- rep(0, times=p)
  }


  lambdaSeq <- grplasso::lambdamax(cbind(1,x), 0.5*(y+1),index=c(NA,1:p), weights = rep(1, nobs),
            offset = offset, coef.init = rep(0, ncol(x)), penscale = sqrt, center = FALSE, standardize = FALSE)* (10/11)^(0:nlambda)

  # fit on all data
  fit <- grplasso::grplasso(cbind(1,x), 0.5*(y+1),index=c(NA,1:p), lambda = lambdaSeq, offset = offset)
  # start cross-validation
  foldid=sample(rep(seq(nfolds),length=nobs))
  if(nfolds<3)stop("nfolds must be bigger than 3; nfolds=5 recommended")
  outlist <- list()

  if (parallel & require(foreach)) {
    library(doParallel)
    n_cores <- detectCores(all.tests = FALSE, logical = TRUE)
    cl <- makeCluster(min(n_cores, nfolds))
    registerDoParallel(cl)
    outlist = foreach (i=seq(nfolds)) %dopar% {
      which=foldid==i
      grplasso::grplasso(x=cbind(1,x[!which,,drop=FALSE]), y=0.5*(y[!which]+1), index=c(NA,1:p), lambda = lambdaSeq, weights = weight[!which], offset = offset[!which])
    }
    stopCluster(cl)
  }else{
    for(i in seq(nfolds)){
      which=foldid==i
      outlist[[i]]=grplasso::grplasso(x=cbind(1,x[!which,,drop=FALSE]), y=0.5*(y[!which]+1), ,index=c(NA,1:p), lambda = lambdaSeq, weights = weight[!which], offset = offset[!which])
    }
  }

  predmat=matrix(NA,length(y),length(lambdaSeq))
  cvraw <- array(0,c(nfolds,length(lambdaSeq)))
  for(i in 1:nfolds){
    cvraw[i,] <- loss.score((foldid==i), outlist[[i]], x, y, offset, lossType = lossType, ...)
  }

  cvm <- apply(cvraw,2,mean,na.rm=TRUE)
  cvsd <- apply(cvraw,2,sd,na.rm=TRUE)

  out <- list(cvm=cvm,cvsd=cvsd,cvup=cvm+cvsd,
              cvlo=cvm-cvsd, cvraw = cvraw, fit=fit, lambda.seq=lambdaSeq, lambda.opt=lambdaSeq[which.min(cvm)])
  class(out) <- "cv.cLearn"
  out
}

# set loss.score
loss.score <- function(testid, fit, x, y, offset, lossType, ...){
  coef <- fit$coefficients
  x.train <- x[testid,]
  y.train <- y[testid]
  offset.train <- offset[testid]
  apply(coef, 2,function(t){mean(loss(y.train *(offset.train+x.train %*% t[-1]+t[1]), lossType=lossType, order = 0, ...))})
}
