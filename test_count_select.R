simulation <- function(sample.size = 500, p = 500, rate=0){
  print(system.time(res <- foreach(index = 1:500,.packages = c('concordLearn'),.combine = rbind,.errorhandling='remove')%dopar%{
    ## test
    set.seed(index)
    nobs <- sample.size
    p <- p
    alpha_seq <- c(0,1, 2, 3)

    V <- function(p, rate = 0.5){
      V.matrix <- array(0, c(p,p))
      for (i in 1:p){
        for (j in 1:p){
          V.matrix[i,j] <- rate ^ (abs(i-j))
        }
      }
      V.matrix
    }

    x <- mgcv::rmvn(sample.size, rep(0, times=p), V(p, rate))
    beta_true <- c(1,-1,1,-1, rep(0, times=p-4))
    beta_modify <- c(1,1,1,1, rep(0, times=p-4))
    yaux <- lapply(alpha_seq, function(s){
      U <- sapply(pnorm(x%*%beta_true), function(t){
        v <- rbinom(1, 4, t)
        v
      })
      U>s
    })
    U <- sapply(pnorm(x%*%beta_true), function(t){
      v <- rbinom(1, 4, t)
      v
    })
    y.cutoff <- c(list(U>0), yaux)

    # f1 score
    score <- rankF1(x,  y.cutoff)

    # fit_trans
    iter <- 1
    fit_logistic_trans <- NULL
    select <- score$rank[score$rank!=1]
    y.select <- list(y.cutoff[[1]], y.cutoff[[select[1]]])
    for (index in select[-1]){
      system.time(fit_logistic_trans[[iter]] <- cInfer(x, y=y.select, y_refit = list(y.cutoff[[1]]), weight = c(1, 1, 1, 1, rep(1, times= p-4)), lossType = 'logistic', tol = 1e-3, parallel = FALSE))
      y.select <- c(y.select, list(res=y.cutoff[[index]]))
      iter <- iter+1
    }
    system.time(fit_logistic_trans[[iter]] <- cInfer(x, y=y.select, y_refit = list(y.cutoff[[1]]), weight = c(1, 1, 1, 1, rep(1, times= p-4)), lossType = 'logistic', tol = 1e-3, parallel = FALSE))
    system.time(fit_logistic <- cInfer(x, y=list(y.cutoff[[1]]), weight = c(1, 1, 1, 1, rep(1, times= p-4)), lossType = 'logistic', tol = 1e-3, parallel = FALSE))

    # testing data
    x.test <- mgcv::rmvn(10^4, rep(0, times=p), V(p, rate))
    y.test.nonoise <- (x.test%*%beta_true)
    y.test <-  sapply(pnorm(x.test%*%beta_true), function(t){
      v <- rbinom(1, 4, t)
      v
    })
    y.test.label <- (y.test>0)

    score.class.logistic.trans <- sapply(fit_logistic_trans, function(t){
      mean((x.test %*% t$coef >= t$off.set) == y.test.label)
    })
    score.class.logistic <- mean((x.test %*% fit_logistic$coef >= fit_logistic$off.set) == y.test.label)

    score.rank.logistic.trans <- sapply(fit_logistic_trans, function(t){
      cor(x.test %*% t$coef,y=y.test.nonoise, method='kendall')
    })
    score.rank.logistic <- cor(x.test %*% fit_logistic$coef,y=y.test.nonoise, method='kendall')


    c(score=c(score.class.logistic.trans,score.class.logistic,
              score.rank.logistic.trans,score.rank.logistic), f1=score$rank)
  }))
  apply(res[,1:10], 2, mean, na.rm = TRUE)
  save(res, file=paste0("/mnt/c/Users/lmx19/Documents/Simulations/concordLearn/sim2","_",sample.size,"_",p,"_",rate, "_transfer_select.RData"))
}

library(doParallel)
n_cores <- detectCores(all.tests = FALSE, logical = TRUE)
cl <- makeCluster(n_cores)
registerDoParallel(cl)

rate <- 0
simulation(sample.size = 350, p=1000, rate = rate)
simulation(sample.size = 500, p=1000, rate = rate)

stopCluster(cl)
