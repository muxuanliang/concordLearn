simulation <- function(sample.size = 500, p = 500){
  library(doParallel)
  n_cores <- detectCores(all.tests = FALSE, logical = TRUE)
  cl <- makeCluster(n_cores)
  registerDoParallel(cl)

  print(system.time(res <- foreach(index = 1:30,.packages = c('concordLearn'),.combine = rbind,.errorhandling='remove')%dopar%{
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
  system.time(fit_hinge <- cInfer(x, y=y.cutoff, weight = c(1, 1, 1, 1, rep(1, times= p-4)), lossType = 'smoothed_hinge', tol = 1e-3, parallel = FALSE))
  #system.time(fit_hinge <- cInfer(x, y=y.cutoff, weight = c(1, 1, 1, 1, rep(1, times= p-4)), lossType = 'smoothed_hinge', tol = 1e-3))

  ## fit using logistic
  system.time(fit_logistic <- cInfer(x, y=y.cutoff, weight = c(1, 1, 1, 1, rep(1, times= p-4)), lossType = 'logistic', tol = 1e-3, parallel = FALSE))
  #system.time(fit_logistic <- cInfer(x, y=y.cutoff, weight = c(1, 1, 1, 1, rep(1, times= p-4)), lossType = 'logistic', tol = 1e-3))

  c(pvalue=c(fit_hinge$pvalue, fit_logistic$pvalue), coefAN = c(fit_hinge$coefAN, fit_logistic$coefAN), sdAN=c(1.96*fit_hinge$sigmaAN/sqrt(nobs), 1.96*fit_logistic$sigmaAN/sqrt(nobs)))
  }))
  stopCluster(cl)
  apply(res[,1:16],2,function(t){mean(t<0.05)})
  coef <- apply(res[,17:32],2,mean)
  apply(apply(res[,17:48],1,function(t){abs(t[1:16]-coef)<t[17:32]}),1,mean)
  save(res, file=paste0("~/Simulations/concordLearn/sim","_",sample.size,"_",p,".RData"))
}
simulation(sample.size = 500, p=500)
