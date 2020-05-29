# cv.cLearn utilizes cross-validation to tuning lambda
cv.cLearn <- function(x, y, lambdaSeq = NULL, weight = rep(1, NCOL(x)), lossType='logistic', nlambda = 100, nfolds=5, parallel = TRUE,ratio  = 10/11, tol = 1e-4, maxIter = 10^3, ...){
  # check wether only 2 classes
  stopifnot(length(levels(factor(y))) == 2)
  nobs <- length(y)
  # set y if y is not 1 or -1
  if (as.numeric(levels(factor(y))[1]) != -1){
    y[factor(y)==levels(factor(y))[1]] <- -1
  }
  if (as.numeric(levels(factor(y))[2]) != 1){
    y[factor(y)==levels(factor(y))[2]] <- 1
  }
  # fit on all data
  fit <- cLearn(x=x, y=y, lambdaSeq = lambdaSeq, weight = weight, lossType = lossType, nlambda = nlambda, ratio = ratio, tol = tol, maxIter = maxIter, ...)
  # set lambda
  lambdaSeq <- fit$lambda
  # start cross-validation
  foldid=sample(rep(seq(nfolds),length=nobs))
  if(nfolds<3)stop("nfolds must be bigger than 3; nfolds=5 recommended")
  outlist <- list()

  if (parallel & require(foreach)) {
    library(doParallel)
    n_cores <- detectCores(all.tests = FALSE, logical = TRUE)
    cl <- makeCluster(min(n_cores, nfolds))
    registerDoParallel(cl)
    outlist = foreach (i=seq(nfolds), .packages=c("concordLearn")) %dopar% {
      which=foldid==i
      cLearn(x=x[!which,,drop=FALSE], y=y[!which], lambdaSeq = lambdaSeq, weight = weight, lossType = lossType, nlambda = nlambda, ratio = ratio, tol = tol, maxIter = maxIter, ...)
    }
    stopCluster(cl)
  }else{
    for(i in seq(nfolds)){
      which=foldid==i
      outlist[[i]]=cLearn(x=x[!which,,drop=FALSE], y=y[!which], lambdaSeq = lambdaSeq, weight = weight, lossType = lossType, nlambda = nlambda, ratio = ratio, tol = tol, maxIter = maxIter, ...)
    }
  }

  predmat=matrix(NA,length(y),length(lambdaSeq))
  cvraw <- array(0,c(nfolds,length(lambdaSeq)))
  for(i in 1:nfolds){
    cvraw[i,] <- loss.score((foldid==i), outlist[[i]], x, y, lossType = lossType, ...)
  }

  cvm <- apply(cvraw,2,mean,na.rm=TRUE)
  cvsd <- apply(cvraw,2,sd,na.rm=TRUE)

  out <- list(cvm=cvm,cvsd=cvsd,cvup=cvm+cvsd,
           cvlo=cvm-cvsd, cvraw = cvraw, fit=fit, lambda.seq=lambdaSeq, lambda.opt=lambdaSeq[which.min(cvm)])
  class(out) <- "cv.cLearn"
  out
}

# set loss.score
loss.score <- function(testid, fit, x, y, lossType, ...){
  coef <- fit$coef
  x.train <- x[testid,]
  y.train <- y[testid]
  apply(coef, 2,function(t){mean(loss(y.train * x.train %*% t, lossType=lossType, order = 0, ...))})
}
