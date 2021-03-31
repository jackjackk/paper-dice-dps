library(pacman)
p_load(coda)
p_load(matrixStats)
p_load(boot)

mcmc.chains = readRDS("brick_mcmc_fgiss_TgissOgour_scauchy_t18802011_z18801900_o4_h50_n10000000_t100_b5-chains.rds")
hd = readRDS("brick_mcmc_fgiss_TgissOgour_scauchy_t18802011_z18801900_o4_h50_n10000000_t100_b5-hd.rds")

nnode.mcmc = length(mcmc.chains)

list.start.iter = c()
for (i in 1:nnode.mcmc) {
  list.start.iter = c(list.start.iter, max(hd[[i]][,2]))
}

n.remaining.chains = sum(!is.na(list.start.iter))
start.iter = max(list.start.iter, na.rm = TRUE)
chains.burned <- vector("list", n.remaining.chains)
j <- 1
for (i in 1:nnode.mcmc) {
  if (is.na(list.start.iter[i])) {
    cat(sprintf("Skipping chain %d\n",i))
    next
  }
  nrows = dim(mcmc.chains[[i]])[1]
  x <- mcmc.chains[[i]][start.iter:nrows,]
  colnames(x) <- parnames
  chains.burned[[j]] = as.mcmc(x)
  cat(sprintf("Removing %d elements, left with %d\n", (start.iter-1), dim(chains.burned[[j]])[1]))
  j <- j + 1
}

mcmc.chains = do.call("as.mcmc.list", list(chains.burned))
cat(sprintf("Left with %d chains", length(mcmc.chains)))

heidel.diag(mcmc.chains)


q.func <- function(x,i){quantile(x[i], seq(0.25,0.75,0.25))}
bootMean$data
str(bootMean)


100*0.020468*1.96/6.357

new.len = dim(mcmc.chains[[1]])[1]
list.max.dev <- c()
for (j in 1:4) {
#q.table <- c()
#for (i in 1:100) { #n.remaining.chains) { #seq(2e3,new.len,1e3)) {
#  q.table <- cbind(q.table, quantile(sample(mcmc.chains[[1]][,j], 1e4, replace = T), seq(0.1,0.9,0.1)))
  #q.table <- rbind(q.table, quantile(tail(mcmc.chains[[1]][,1], as.integer(new.len/i)), seq(0.1,0.9,0.1)))
#}
  x = mcmc.chains[[1]][,j]
  bootMean <- boot(x,q.func,100)
  cat(sprintf("max pct se: %.1f\n", max(100*1.96*rowSds(t(bootMean$t))/rowMeans(t(bootMean$t)))))
}

sqrt(var(x)/length(x))
rowSds(q.table)/rowMeans(q.table)
list.max.dev <- c(list.max.dev, 100*(abs(q.table/rowMeans(q.table)-1)))
}
cat(sprintf("Max deviation from mean statistics: %.1f percent\n", max(list.max.dev)))

1-q.table[1,]/q.table[2,]

gelman.diag()

names(a) <- parnames
colnames(a) <- parnames

summary(data.frame(a[,], row.names=parnames))
library("optparse")
str(parnames)
option_list = list(
  make_option(c("-r", "--rdsfile"), type="character", default="~/CloudStation/psu/projects/brick/doeclim_mcmc_e2011_t1929_o4_n2000000.rds", help="rds file"),
  make_option(c("-t", "--thin"), type="integer", default=1000, help="keep one every N samples"),
  make_option(c("-b", "--burnin"), type="integer", default=5, help="percentage to burn"),
  make_option(c("-n", "--nchain"), type="integer", default=1, help="number of chains")
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

setwd("../../../data/brick")
load("brick_mcmc_fgiss_TgissOgour_scauchy_t18802011_z18801900_o4_h50_n10000000.RData")
amcmc.par1 = readRDS("brick_mcmc_fgiss_TgissOgour_scauchy_t18802011_z18801900_o4_h50_n10000000.rds")

burnin = round(niter.mcmc*opt$burnin/100)				# how much to remove for burn-in
ifirst = burnin

library(coda)
#trace = amcmc.par1[[1]]$samples[(ifirst+1):niter.mcmc,]
#acf(trace[seq(1,nrow(trace),9999)])
thin = opt$thin-1

chains_burned <- vector('list', nnode.mcmc)
for (m in 1:nnode.mcmc) {
  trace <- amcmc.par1[[m]]$samples[(ifirst+1):niter.mcmc,1:length(parnames)]
  chains_burned[[m]] <- trace[seq(1, nrow(trace), thin + 1), ]
}

library(coda)
mcmc1 = as.mcmc(tail(amcmc.par1[[1]]$samples, 1e5))
mcmc2 = as.mcmc(tail(amcmc.par1[[2]]$samples, 1e5))
mcmc3 = as.mcmc(tail(amcmc.par1[[3]]$samples, 1e5))
mcmc4 = as.mcmc(tail(amcmc.par1[[4]]$samples, 1e5))
mcmc.chains = mcmc.list(list(mcmc1, mcmc2, mcmc3, mcmc4))

saveRDS(mcmc.chains, file="brick_mcmc_test.rds")
