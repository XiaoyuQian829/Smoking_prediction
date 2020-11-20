source("qq_func.R")


path="/home/uqxqian/90days/Prediction/ML_linear/smoke_osca/"
source("PlotingFunc2.r")
trait=c(" ")
traits=c(" ")
midd=c("_linear.","_moa.","_moment.","_moment2.");
surffix=c("linear","moa","moment","moment");

mtd=c("Linear","MOA","MOMENT","MOMENT2")
alias=c("a","b","c","d","e","f","g","h","i","j")
pvalpos=c(8,8,8,8,8);


plotfname=paste(path,"plot/smoke.qq.png",sep="")
png(plotfname,width = 12, height = 12, units = 'in', res = 300,  type="cairo")
par(mfrow=c(2,2))
for(cat in 1:length(trait))
{
	pval=c()
        legend=c()
	for( j in 1:length(midd))
	{
	    rlt=read.table(paste(path, trait[cat],midd[j],surffix[j],sep=""),header=T)
           

	   chi=qchisq(rlt[,pvalpos[j]],1,lower.tail=F)
        lam=median(chi)/0.455
        legend=c(legend,paste(mtd[j]," (",expression(lambda),"=",round(lam,2),")",sep=""))
    	  pval=cbind(pval,rlt[,pvalpos[j]])
	 }
    	colnames(pval)=legend
	cols=c("#375E97","#FB6542","#31A9B8","#e504e1","#4717F6","#B82601")
    	#cols=c("#31A9B8","#4717F6","#e504e1")
        pchs=c(0,2,6,3,1,4,5,7)
    	qqplot_CI(pval,title=traits[cat],cols=cols, pchs=pchs)

}

dev.off()
