## calcualte the ICC and discriminability for block size from all 4 scans of all subjects
library(RcppCNPy)


data=npyLoad("blocksize_sub_globlesig_individual_scan.npy")
subn=dim(data)[2]

iccdata=data.frame(matrix(, nrow = subn*4, ncol = 4))
colnames(iccdata)=c('Subject','Session','data','FCMean')

for (i in 1:4){
  print(i)
  
  iccdata$Subject[(1+(i-1)*subn):(subn*i)]=1:subn 
  iccdata$Session[(1+(i-1)*subn):(subn*i)]=i
  iccdata$data[(1+(i-1)*subn):(subn*i)]=data[i,]
}


## ICC

lm=tryCatch({summary(lmer(data ~ (1 | Subject),data=iccdata))}, error=function(e) {return('NA')})
Var_between=as.numeric(attributes(lm$varcor$Subject)$stddev)^2
Var_within=as.numeric(attributes(lm$varcor)$sc)^2
ICC=(Var_between)/(Var_between+Var_within)
print(ICC)


## Discriminability
discr=discr.stat(iccdata$data,c(1:subn,1:subn,1:subn,1:subn))
print(discr$discr)