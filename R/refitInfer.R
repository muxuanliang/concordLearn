#refitInfer implement the decorrelated score for testing and interval estimation for refitting procedure
refitInfer <- function(x, y, refit = NULL, lossType='logistic', parallel = TRUE, indexToTest = NULL, ...){
  # set sample size
  n <- NROW(x)

  #refit coef and off.set
  coef <- refit$coef
  off.set <- refit$off.set

  # if indextoTest is null
  if (is.null(indexToTest)){
    indexToTest <- c(1:8)
  }

  # calculate link and first order derivative
  link <- x %*% coef-off.set
  first.order <- y * loss(y * link, lossType=lossType, order=1, ...)
  second.order <- loss(y * link, lossType=lossType, order=2, ...)

  # fit w
  fit_w <- NULL
  score <- rep(NA, times=length(indexToTest))
  sigma <- rep(NA, times=length(indexToTest))
  sigmaNULL <- rep(NA, times=length(indexToTest))
  coefAN <- rep(NA, times=length(indexToTest))
  I <- rep(NA, times=length(indexToTest))
  if (!parallel){
    for (index in 1:length(indexToTest)){
      # set pseudo data
      pseudo.x <- cbind(x[, -indexToTest[index]], -1)
      pseudo.y <- x[,indexToTest[index]]
      #fit w
      fit_w[[index]] <- glmnet::cv.glmnet(x=pseudo.x, y=pseudo.y, weights = second.order, penalty.factor=c(rep(1, NCOL(x)-1), 0), intercept = FALSE, standardize = FALSE)
      link_w <- predict(fit_w[[index]], newx = pseudo.x, s=fit_w[[index]]$lambda.min)
      # set beta null
      coefNULL <- coef
      coefNULL[indexToTest[index]] <- 0
      # get score under null
      linkNULL <- x %*% coefNULL-off.set
      first.orderNULL <- y * loss(y * linkNULL, lossType=lossType, order=1, ...)
      tmpNULL <- first.orderNULL*(pseudo.y-link_w)
      score[index] <- mean(tmpNULL)
      sigmaNULL[index] <- sqrt(mean((tmpNULL)^2))

      # set betaAN
      tmp <- first.order*(pseudo.y-link_w)
      Itmp <- second.order * pseudo.y * (pseudo.y-link_w)
      coefAN[index] <- coef[indexToTest[index]]-mean(tmp) /(mean(Itmp))
      sigma[index] <- sqrt(mean((tmp)^2))
      I[index] <- (mean(Itmp))
    }
  } else {
    # set multi-thread
    library(doParallel)
    n_cores <- detectCores(all.tests = FALSE, logical = TRUE)
    cl <- makeCluster(min(length(indexToTest), n_cores))
    registerDoParallel(cl)

    #calculate
    res <- foreach(index=indexToTest,.packages = 'glmnet') %dopar%{
      # set pseudo data
      pseudo.x <- cbind(x[, -indexToTest[index]], -1)
      pseudo.y <- x[,indexToTest[index]]
      #fit w
      fit_w <- glmnet::cv.glmnet(x=pseudo.x, y=pseudo.y, weights = second.order, intercept = FALSE, standardize = FALSE)
      link_w <- predict(fit_w, newx = pseudo.x, s=fit_w$lambda.min)
      # set beta null
      coefNULL <- coef
      coefNULL[indexToTest[index]] <- 0
      # get score under null
      linkNULL <- x %*% coefNULL-off.set
      first.orderNULL <- y * loss(y * linkNULL, lossType=lossType, order=1, ...)
      tmpNULL <- first.orderNULL*(pseudo.y-link_w)
      score <- mean(tmpNULL)
      sigmaNULL <- sqrt(mean((tmpNULL)^2))

      # set betaAN
      tmp <- first.order*(pseudo.y-link_w)
      Itmp <- second.order * pseudo.y * (pseudo.y-link_w)
      coefAN <- coef[indexToTest[index]]-mean(tmp) /(mean(Itmp))
      sigma <- sqrt(mean((tmp)^2))
      I <- (mean(Itmp))
      list(fit_w = fit_w, score=score, sigmaNULL=sigmaNULL, sigma=sigma, coefAN=coefAN, I=I)
    }
    stopCluster(cl)
    for (index in indexToTest){
      score[index] <- res[[index]]$score
      sigmaNULL[index] <- res[[index]]$sigmaNULL
      sigma[index] <- res[[index]]$sigma
      coefAN[index] <- res[[index]]$coefAN
      I[index] <- res[[index]]$I
    }
  }
  list(fit=refit, coef=coef, off.set=off.set, indexToTest=indexToTest, pvalue=pnorm(-abs(sqrt(n)*score/sigmaNULL))*2, coefAN=coefAN, sigmaAN=sigma/I)
}
