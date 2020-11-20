
path="/shares/compbio/Group-Yang/uqfzhan7/PANCAN/LGG2/"
source("PlotingFunc2.r")
trait=c("wLGG.OS","wLGG.DSS","wLGG.PFI","wLGG.DFI")
traits=c("OS","DSS","PFI","DFI")
midd=c(".",".",".",".",".moment2.");
surffix=c("cox","linear","moa","moment","moment");

mtd=c("Cox regression","Linear regression","MOA","MOMENT","MOMENT2")
alias=c("a","b","c","d","e","f","g","h","i","j")

for(cat in 1:length(trait))
{	
  print(cat)
  plotfname=paste(path,"plot/",trait[cat],".manh.png",sep="")
  png(plotfname,width = 12, height = 12, units = 'in', res = 300)
  par(mfrow=c(3,2))
  
  for( j in 1:length(midd))
  {
    print(j)
     rlt=read.table(paste(path,"rlt/", trait[cat],midd[j],surffix[j],sep=""),header=T)
   
     names(rlt)[1:8]=c("CHR","SNP","BP","GENE","ORIEN","BETA","SE","P");
    
    thresh=-log10(0.05/dim(rlt)[1])
   
    manhattan2(rlt, genomewideline = thresh)
    title(main=alias[j],adj=0,cex.main = 1.01)
    title(main=paste(traits[cat]," : ",mtd[j],sep=""),cex.main = 1.01)
  }
  dev.off()
}



