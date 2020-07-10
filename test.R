simulation <- function(sample.size = 500, p = 500){
  library(doParallel)
  n_cores <- detectCores(all.tests = FALSE, logical = TRUE)
  cl <- makeCluster(n_cores)
  registerDoParallel(cl)

  print(system.time(res <- foreach(index = 1:10,.packages = c('concordLearn'),.combine = rbind,.errorhandling='remove')%dopar%{
  ## test
  set.seed(index)
  nobs <- sample.size
  p <- p
  x <- array(rnorm(nobs*p), c(nobs, p))
  beta_true <- c(1,-1,0.5,-0.5, rep(0, times=p-4))
  y <- exp(x%*%beta_true) + rnorm(nobs)
  cutoff <- quantile(y)[c(2,4)]
  y.cutoff <- list(y>cutoff[1], y>cutoff[2])

  ## fit using smoothed_hinge
  system.time(fit_hinge_trans <- cInfer(x, y=y.cutoff, y_refit = list(y>cutoff[1]), weight = c(1, 1, 1, 1, rep(1, times= p-4)), lossType = 'smoothed_hinge', tol = 1e-3, parallel = FALSE))
  system.time(fit_hinge <- cInfer(x, y=list(y>cutoff[1]), weight = c(1, 1, 1, 1, rep(1, times= p-4)), lossType = 'smoothed_hinge', tol = 1e-3, parallel = FALSE))

  ## fit using logistic
  system.time(fit_logistic_trans <- cInfer(x, y=y.cutoff, y_refit = list(y>cutoff[1]), weight = c(1, 1, 1, 1, rep(1, times= p-4)), lossType = 'logistic', tol = 1e-3, parallel = FALSE))
  system.time(fit_logistic <- cInfer(x, y=list(y>cutoff[1]), weight = c(1, 1, 1, 1, rep(1, times= p-4)), lossType = 'logistic', tol = 1e-3))

  # testing data
  x.test <- array(rnorm(10^4*p), c(10^4, p))
  y.test.nonoise <- exp(x.test%*%beta_true)
  y.test <- y.test.nonoise + rnorm(10^4)
  y.test.label <- (y.test>cutoff[1])
  score.class.hinge.trans <- mean((x.test %*% fit_hinge_trans$coef > fit_hinge_trans$off.set) == y.test.label)
  score.class.hinge <- mean((x.test %*% fit_hinge$coef > fit_hinge$off.set) == y.test.label)
  score.class.logistic.trans <- mean((x.test %*% fit_logistic_trans$coef > fit_logistic_trans$off.set) == y.test.label)
  score.class.logistic <- mean((x.test %*% fit_logistic$coef > fit_logistic$off.set) == y.test.label)
  score.rank.hinge.trans <- cor(x.test %*% fit_hinge_trans$coef,y=y.test.nonoise, method='kendall')
  score.rank.hinge <- cor(x.test %*% fit_hinge$coef,y=y.test.nonoise, method='kendall')
  score.rank.logistic.trans <- cor(x.test %*% fit_logistic_trans$coef,y=y.test.nonoise, method='kendall')
  score.rank.logistic <- cor(x.test %*% fit_logistic$coef,y=y.test.nonoise, method='kendall')

  c(score=c(score.class.hinge.trans,score.class.hinge,score.class.logistic.trans,score.class.logistic,
            score.rank.hinge.trans,score.rank.hinge,score.rank.logistic.trans,score.rank.logistic), pvalue=c(fit_hinge_trans$pvalue, fit_hinge$pvalue, fit_logistic_trans$pvalue, fit_logistic$pvalue), coefAN = c(fit_hinge_trans$coefAN,fit_hinge$coefAN, fit_logistic_trans$coefAN, fit_logistic$coefAN), sdAN=c(1.96*fit_hinge_trans$sigmaAN/sqrt(nobs), 1.96*fit_hinge$sigmaAN/sqrt(nobs), 1.96*fit_logistic_trans$sigmaAN/sqrt(nobs), 1.96*fit_logistic$sigmaAN/sqrt(nobs)))
  }))
  stopCluster(cl)
  apply(res[,1:8], 2, mean)
  apply(res[,9:40],2,function(t){mean(t<0.05)})
  coef <- apply(res[,41:72],2,mean)
  #sd <- apply(res[,17:32],2,sd)
  apply(apply(res[,40:104],1,function(t){abs(t[1:32]-coef)<t[33:64]}),1,mean)
  save(res, file=paste0("~/Documents/Research/Xiang Zhong/risk score/sim","_",sample.size,"_",p,".RData"))
}
simulation(sample.size = 200, p=200)
