path="/home/uqxqian/90days/Prediction/ML_linear/smoke_osca/"
source("PlotingFunc2.r")
trait=c("smoke_LBC36")
traits=c("Smoking status")
midd=c("_linear.","_moa.","_moment.","_moment2.");
surffix=c("linear","moa","moment","moment");

mtd=c("Linear","MOA","MOMENT","MOMENT2")
alias=c("a","b","c","d","e","f","g","h","i","j")

for(cat in 1:length(trait))
{	
  print(cat)
  plotfname=paste(path,"plot/",trait[cat],".manh.png",sep="")
  png(plotfname,width = 12, height = 12, units = 'in', res = 300,  type="cairo")
  par(mfrow=c(3,2))
  
  for( j in 1:length(midd))
  {
    print(j)
     rlt=read.table(paste(path, trait[cat],midd[j],surffix[j],sep=""),header=T)
   
     names(rlt)[1:8]=c("CHR","SNP","BP","GENE","ORIEN","BETA","SE","P");
    
    thresh=-log10(0.05/dim(rlt)[1])
   
    manhattan2(rlt, genomewideline = thresh)
    title(main=alias[j],adj=0,cex.main = 1.01)
    title(main=paste(mtd[j],sep=""),cex.main = 1.5)
  }
  dev.off()
}
