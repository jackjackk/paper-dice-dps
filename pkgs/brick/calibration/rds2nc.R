library("optparse")

option_list = list(
  make_option(c("-r", "--rdsfile"), type="character", default="~/CloudStation/psu/projects/brick/doeclim_mcmc_e2011_t1929_o4_n2000000.rds", help="rds file"),
  make_option(c("-t", "--thin"), type="integer", default=1000, help="keep one every N samples"),
  make_option(c("-b", "--burnin"), type="integer", default=5, help="percentage to burn"),
  make_option(c("-n", "--nchain"), type="integer", default=1, help="number of chains")
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

p_load(coda)
#load("brick_mcmc_fgiss_TgissOgour_scauchy_t18802011_z18801900_o4_h50_n10000000.RData")
paste0('Loading ', opt$rdsfile)
amcmc.par1 = readRDS(opt$rdsfile)
paste0('Thinning')
if (is.atomic(amcmc.par1[[1]])) {
  niter.mcmc = nrow(amcmc.par1[[1]])
} else {
  niter.mcmc = amcmc.par1[[1]]$n.sample
}
paste0("niter = ", niter.mcmc)
nnode.mcmc <- length(amcmc.par1)
paste0("nnode = ", nnode.mcmc)

burnin = round(niter.mcmc*opt$burnin/100)				# how much to remove for burn-in
thin = opt$thin-1
paste0("burnin = ", burnin)
paste0("thin = ", thin)
isamples = seq(burnin+1, niter.mcmc, thin + 1)
summary(isamples)

parnames   =c("S", "kappa.doeclim", "alpha.doeclim", "T0", "H0", "beta0", "V0.gsic", "n", "Gs0", "a.te", "b.te", "invtau.te", "TE0", "a.simple", "b.simple", "alpha.simple", "beta.simple", "V0", "sigma.T", "sigma.H", "rho.T", "rho.H", "sigma.gsic", "rho.gsic", "sigma.simple", "rho.simple")
nparams = length(parnames)

basename = paste0(tools::file_path_sans_ext(opt$rdsfile),'_t',opt$thin,'_b',opt$burnin)
                  
library(coda)

chains_burned <- vector("list", nnode.mcmc)

for (m in 1:nnode.mcmc) {
  paste0("Thinning chain ", m)
  if (is.atomic(amcmc.par1[[m]])) {
    chains_burned[[m]] <- as.mcmc(amcmc.par1[[m]][isamples,1:nparams])
  } else {
    chains_burned[[m]] <- as.mcmc(amcmc.par1[[m]]$samples[isamples,])
  }
}
mcmc.chains = mcmc.list(chains_burned)
                                        # Run Heidel test
hd = heidel.diag(mcmc.chains, eps = 0.1, pvalue = 0.05)
print(hd)
saveRDS(hd, file=paste0(basename, "-hd.rds"))

                                        # Save chains RDS
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
cat(sprintf("Left with %d chains\n", n.remaining.chains))
saveRDS(mcmc.chains, file=paste0(basename, "-chains.rds"))

                                        # Save first chain as nc
parameters.posterior <- chains_burned[[opt$nchain]]
                                        #covjump.posterior <- amcmc.par1[[1]]$cov.jump
if (n.remaining.chains>1) {
    for (m in 2:n.remaining.chains) {
        parameters.posterior <- rbind(parameters.posterior, chains_burned[[m]])
    }
}

## Get maximum length of parameter name, for width of array to write to netcdf
## this code will write an n.parameters (rows) x n.ensemble (columns) netcdf file
## to get back into the shape BRICK expects, just transpose it
lmax=0
for (i in 1:length(parnames)){lmax=max(lmax,nchar(parnames[i]))}

library(ncdf4)
dim.parameters <- ncdim_def('n.parameters', '', 1:ncol(parameters.posterior), unlim=FALSE)
dim.name <- ncdim_def('name.len', '', 1:lmax, unlim=FALSE)
dim.ensemble <- ncdim_def('n.ensemble', 'ensemble member', 1:nrow(parameters.posterior), unlim=TRUE)
parameters.var <- ncvar_def('BRICK_parameters', '', list(dim.parameters,dim.ensemble), -999)
parnames.var <- ncvar_def('parnames', '', list(dim.name,dim.parameters), prec='char')
outfile <- paste0(basename, '.nc')
outnc <- nc_create(outfile , list(parameters.var,parnames.var))
ncvar_put(outnc, parameters.var, t(parameters.posterior))
ncvar_put(outnc, parnames.var, parnames)
nc_close(outnc)

