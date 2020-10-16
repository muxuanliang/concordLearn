#simTrans implement the Sai Li's naive procedure
simTrans <- function(x, y=list(y1, y2, y3), y_refit = NULL, fit = NULL, weight = rep(1, times=NCOL(x)), lossType='logistic', parallel = TRUE, ...){
  # set how any outcomes
  n.cutoff <- length(y)

  # set sample size
  n <- NROW(x)

  # set y if y is not 1 or -1
  y <- lapply(y, function(t){
    t.ones <- rep(1, length(t))
    if (((levels(factor(t))[1]) != "-1")|(levels(factor(t))[2] != "1")){
      t.ones[factor(t)==levels(factor(t))[1]] <- -1
      t.ones[factor(t)==levels(factor(t))[2]] <- 1
    }
    t.ones
  })

  # form a training matrix
  x.combine <- cbind(apply(-pracma::eye(n.cutoff), 1, function(t){pracma::repmat(t, n, 1)}), pracma::repmat(x, n.cutoff, 1))
  y.combine <- unlist(y)



  # if coef=NULL
  if (is.null(fit)){
    # if unly one outcome
    if (n.cutoff == 1){
      fit <- cv.cLearn(x=x, y=y.combine, lambdaSeq = NULL, weight = weight, lossType = lossType, parallel = parallel, intercept=TRUE, ...)
      coef <- fit$fit$coef[,fit$lambda.seq==fit$lambda.opt]
      if(lossType == "logistic") off.set <- -fit$fit$a0[fit$lambda.seq==fit$lambda.opt]
      else off.set <- -fit$fit$coef[(1:n.cutoff),fit$lambda.seq==fit$lambda.opt]

      return(list(coef=coef, off.set=off.set))
    } else {
      fit <- cv.cLearn(x=x.combine, y=y.combine, lambdaSeq = NULL, weight = c(rep(0, times=length(y)), weight), lossType = lossType, parallel = parallel, ...)
      # set coef and off.set
      coef <- fit$fit$coef[-(1:n.cutoff),fit$lambda.seq==fit$lambda.opt]
      off.set <- fit$fit$coef[(1:n.cutoff),fit$lambda.seq==fit$lambda.opt]
    }
  } else {
    # set coef and off.set
    coef <- fit$fit$coef[-(1:n.cutoff),fit$lambda.seq==fit$lambda.opt]
    off.set <- fit$fit$coef[(1:n.cutoff),fit$lambda.seq==fit$lambda.opt]
  }

  if (!is.null(y_refit)){
    # set y_refit
    y_refit <- lapply(y_refit, function(t){
      t.ones <- rep(1, length(t))
      if (((levels(factor(t))[1]) != "-1")|(levels(factor(t))[2] != "1")){
        t.ones[factor(t)==levels(factor(t))[1]] <- -1
        t.ones[factor(t)==levels(factor(t))[2]] <- 1
      }
      t.ones
    })

    # change to one cutoff
    n.cutoff <- 1
    offset.refit <- x %*% coef
    x.refit <- x
    y.refit <- unlist(y_refit)

    # refit
    fit_refit <- cv.simTrans(x=x.refit, y=y.refit, lambdaSeq = NULL, lossType = lossType, parallel = parallel, intercept=TRUE, offset=offset.refit, ...)
    coef <- fit_refit$fit$coefficients[-1,fit_refit$lambda.seq==fit_refit$lambda.opt]+coef
    off.set <- -fit_refit$fit$coefficients[1,fit_refit$lambda.seq==fit_refit$lambda.opt]


    return(list(coef=coef, off.set=off.set))
  }


  list(coef=coef, off.set=off.set)
}
