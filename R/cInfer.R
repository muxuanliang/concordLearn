#cInfer implement the decorrelated score for testing and interval estimation
cInfer <- function(x, y, fit = NULL, lossType='logistic', parallel = TRUE, indexToTest = NULL, intercept=TRUE, ...){
  #set sample size
  n <- length(y)
  # if coef=NULL
  if (is.null(fit)){
    fit <- cv.cLearn(x=x, y=y, lambdaSeq = NULL, lossType = lossType, parallel = parallel, ...)
  }
  coef <- fit$fit$coef[,fit$lambda.seq==fit$lambda.opt]

  # if indextoTest is null
  if (is.null(indexToTest)){
    indexToTest <- c(1:8)
  }

  # set y if y is not 1 or -1
  if (as.numeric(levels(factor(y))[1]) != -1){
    y[factor(y)==levels(factor(y))[1]] <- -1
  }
  if (as.numeric(levels(factor(y))[2]) != 1){
    y[factor(y)==levels(factor(y))[2]] <- 1
  }
  # fit w
  fit_w <- NULL
  score <- rep(NA, times=length(indexToTest))
  sigma <- rep(NA, times=length(indexToTest))
  coefAN <- rep(NA, times=length(indexToTest))
  I <- rep(NA, times=length(indexToTest))
  if (!parallel){
    for (index in 1:length(indexToTest)){
      # set pseudo data
      pseudo.x <- x[, -indexToTest[index]]
      pseudo.y <- x[,indexToTest[index]]
      # set weight
      weights <- loss(y * x %*% coef, lossType=lossType, order=2, ...)
      #fit w
      fit_w[[index]] <- glmnet::cv.glmnet(x=pseudo.x, y=pseudo.y, weights = weights, intercept = intercept, standardize = FALSE)
      link_w <- predict(fit_w[[index]], newx = pseudo.x, s=fit_w[[index]]$lambda.min)
      # set beta null
      coefNULL <- coef
      coefNULL[indexToTest[index]] <- 0
      # get score under null
      linkNULL <- x %*% coefNULL
      scoreWeightNULL <- y * loss(y * linkNULL, lossType=lossType, order=1, ...)
      tmpNULL <- scoreWeightNULL * (pseudo.y-link_w)
      score[index] <- mean(tmpNULL)
      # set betaAN
      link <- x %*% coef
      scoreWeight <- y * loss(y * link, lossType=lossType, order=1, ...)
      tmp <- scoreWeight * (pseudo.y-link_w)
      Itmp <- loss(y * link, lossType=lossType, order=2, ...) * pseudo.y * (pseudo.y - link_w)
      coefAN[index] <- coef[indexToTest[index]]-mean(tmp) /(mean(Itmp))
      sigma[index] <- sqrt(mean((tmp)^2))
      I[index] <- (mean(Itmp))

      sigma[index] <- sqrt(mean((tmpNULL)^2))
    }
  } else {
    # set multi-thread
    library(doParallel)
    n_cores <- detectCores(all.tests = FALSE, logical = TRUE)
    cl <- makeCluster(min(length(indexToTest), n_cores))
    registerDoParallel(cl)

    #calculate
    res <- foreach(index=indexToTest,.packages = 'glmnet') %dopar%{
      pseudo.x <- x[, -indexToTest[index]]
      pseudo.y <- x[, indexToTest[index]]
      # set weight
      weights <- loss(y * x %*% coef, lossType=lossType, order=2, ...)
      #fit w
      fit_w <- glmnet::cv.glmnet(x=pseudo.x, y=pseudo.y, weights = weights, intercept = intercept, standardize = FALSE)
      link_w <- predict(fit_w, newx = pseudo.x, s=fit_w$lambda.min)
      # set beta null
      coefNULL <- coef
      coefNULL[indexToTest[index]] <- 0
      # get score under null
      linkNULL <- x %*% coefNULL
      scoreWeightNULL <- y * loss(y * linkNULL, lossType=lossType, order=1, ...)
      tmpNULL <- scoreWeightNULL * (pseudo.y-link_w)
      score <- mean(tmpNULL)
      # set betaAN
      link <- x %*% coef
      scoreWeight <- y * loss(y * link, lossType=lossType, order=1, ...)
      tmp <- scoreWeight * (pseudo.y-link_w)
      Itmp <- loss(y * link, lossType=lossType, order=2, ...) * pseudo.y * (pseudo.y - link_w)
      coefAN <- coef[indexToTest[index]]-mean(tmp) /(mean(Itmp))
      sigma <- sqrt(mean((tmp)^2))
      I <- (mean(Itmp))
      sigma <- sqrt(mean((tmpNULL)^2))
      list(fit_w = fit_w, score=score, sigma=sigma, coefAN=coefAN, I=I)
    }
    stopCluster(cl)
    for (index in indexToTest){
      score[index] <- res[[index]]$score
      sigma[index] <- res[[index]]$sigma
      coefAN[index] <- res[[index]]$coefAN
      I[index] <- res[[index]]$I
    }
  }
  list(fit=fit, coef=coef, indexToTest=indexToTest, pvalue=pnorm(-abs(sqrt(n)*score/sigma))*2, coefAN=coefAN, sigmaAN=sigma/I)
}
