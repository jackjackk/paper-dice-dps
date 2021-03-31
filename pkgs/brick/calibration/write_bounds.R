source('BRICK_calib_driver.R')
df.bounds = data.frame(lo=bound.lower,up=bound.upper,row.names=parnames)
write.csv(df.bounds, file="../results/bounds.csv")
