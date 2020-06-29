#cInfer implement the decorrelated score for testing and interval estimation
cInfer <- function(x, y=list(y1, y2, y3), fit = NULL, weight = rep(1, times=NCOL(x)), lossType='logistic', parallel = TRUE, indexToTest = NULL, ...){
  # reset weight
  weight <- c(rep(0, times=length(y)), weight)

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
    fit <- cv.cLearn(x=x.combine, y=y.combine, lambdaSeq = NULL, weight = weight, lossType = lossType, parallel = parallel, ...)
  }
  coef <- fit$fit$coef[-(1:n.cutoff),fit$lambda.seq==fit$lambda.opt]
  off.set <- fit$fit$coef[(1:n.cutoff),fit$lambda.seq==fit$lambda.opt]

  # if indextoTest is null
  if (is.null(indexToTest)){
    indexToTest <- c(1:8)
  }

  # calculate link and first order derivative
  link <- matrix(apply(pracma::repmat(x %*% coef, 1, n.cutoff), 1, function(t){t-off.set}), nrow = n, ncol = n.cutoff, byrow = TRUE)
  first.order <- array(0, c(n, n.cutoff))
  for (i in 1:n.cutoff){
    first.order[,i] <- y[[i]] * loss(y[[i]] * link[,i], lossType=lossType, order=1, ...)
  }
  second.order <- array(0, c(n, n.cutoff))
  for (i in 1:n.cutoff){
    second.order[,i] <- loss(y[[i]] * link[,i], lossType=lossType, order=2, ...)
  }

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
      pseudo.x <- cbind(apply(x[, -indexToTest[index]], 2, function(t){t * apply(first.order, 1, sum)}), -first.order)
      pseudo.y <- x[,indexToTest[index]] * apply(first.order, 1, sum)
      #fit w
      fit_w[[index]] <- glmnet::cv.glmnet(x=pseudo.x, y=pseudo.y, intercept = FALSE, standardize = FALSE)
      link_w <- predict(fit_w[[index]], newx = pseudo.x, s=fit_w[[index]]$lambda.min)
      # set beta null
      coefNULL <- coef
      coefNULL[indexToTest[index]] <- 0
      # get score under null
      linkNULL <- matrix(apply(pracma::repmat(x %*% coefNULL, 1, n.cutoff), 1, function(t){t-off.set}), nrow = n, ncol = n.cutoff, byrow = TRUE)
      first.orderNULL <- array(0, c(n, n.cutoff))
      for (i in 1:n.cutoff){
        first.orderNULL[,i] <- y[[i]] * loss(y[[i]] * linkNULL[,i], lossType=lossType, order=1, ...)
      }
      pseudo.xNULL <- cbind(apply(x[, -indexToTest[index]], 2, function(t){t * apply(first.orderNULL, 1, sum)}), -first.orderNULL)
      pseudo.yNULL <- x[,indexToTest[index]] * apply(first.orderNULL, 1, sum)
      link_wNULL <- predict(fit_w[[index]], newx = pseudo.xNULL, s=fit_w[[index]]$lambda.min)
      tmpNULL <- pseudo.yNULL-link_wNULL
      score[index] <- mean(tmpNULL)
      sigmaNULL[index] <- sqrt(mean((tmpNULL)^2))

      # set betaAN
      tmp <- pseudo.y-link_w
      Itmp <- (x[,indexToTest[index]])^2 * apply(second.order, 1, sum) - predict(fit_w[[index]], newx = cbind(apply(x[, -indexToTest[index]], 2, function(t){t * apply(second.order, 1, sum) * x[,indexToTest[index]]}), -apply(second.order, 2, function(t){t * x[,indexToTest[index]]})), s=fit_w[[index]]$lambda.min)
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
      pseudo.x <- cbind(apply(x[, -indexToTest[index]], 2, function(t){t * apply(first.order, 1, sum)}), -first.order)
      pseudo.y <- x[,indexToTest[index]] * apply(first.order, 1, sum)
      #fit w
      fit_w <- glmnet::cv.glmnet(x=pseudo.x, y=pseudo.y, intercept = FALSE, standardize = FALSE)
      link_w <- predict(fit_w, newx = pseudo.x, s=fit_w$lambda.min)
      # set beta null
      coefNULL <- coef
      coefNULL[indexToTest[index]] <- 0
      # get score under null
      linkNULL <- matrix(apply(pracma::repmat(x %*% coefNULL, 1, n.cutoff), 1, function(t){t-off.set}), nrow = n, ncol = n.cutoff, byrow = TRUE)
      first.orderNULL <- array(0, c(n, n.cutoff))
      for (i in 1:n.cutoff){
        first.orderNULL[,i] <- y[[i]] * loss(y[[i]] * linkNULL[,i], lossType=lossType, order=1, ...)
      }
      pseudo.xNULL <- cbind(apply(x[, -indexToTest[index]], 2, function(t){t * apply(first.orderNULL, 1, sum)}), -first.orderNULL)
      pseudo.yNULL <- x[,indexToTest[index]] * apply(first.orderNULL, 1, sum)
      link_wNULL <- predict(fit_w, newx = pseudo.xNULL, s=fit_w$lambda.min)
      tmpNULL <- pseudo.yNULL-link_wNULL
      score <- mean(tmpNULL)
      sigmaNULL <- sqrt(mean((tmpNULL)^2))

      # set betaAN
      tmp <- pseudo.y-link_w
      Itmp <- (x[,indexToTest[index]])^2 * apply(second.order, 1, sum) - predict(fit_w, newx = cbind(apply(x[, -indexToTest[index]], 2, function(t){t * apply(second.order, 1, sum) * x[,indexToTest[index]]}), -apply(second.order, 2, function(t){t * x[,indexToTest[index]]})), s=fit_w$lambda.min)
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
  list(fit=fit, coef=coef, off.set=off.set, indexToTest=indexToTest, pvalue=pnorm(-abs(sqrt(n)*score/sigmaNULL))*2, coefAN=coefAN, sigmaAN=sigma/I)
}
