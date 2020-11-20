#############################################
####         ploting functions           ####
#############################################
qqplot_CI_single=function(pval, title){
p1 <- pval
p2 <- sort(p1)
n  <- length(p2)
k  <- c(1:n)
alpha   <- 0.05
lower   <- qbeta(alpha/2,k,n+1-k)
upper   <- qbeta((1-alpha/2),k,n+1-k)
expect  <- (k-0.5)/n
biggest <- ceiling(max(-log10(p2),-log10(expect)))
shade <- function(x1, y1, x2, y2, color = col.shade) {
    n <- length(x2)
polygon(c(x1, x2[n:1]), c(y1, y2[n:1]), border = NA, col = color)
}
xlim=max(-log10(expect)+0.1);
ylim=biggest;
plot(-log10(expect),-log10(p2),xlim=c(0,xlim),ylim=c(0,ylim),ylab=expression(paste("Observed  ","-",log[10],"(P)")), xlab=expression(paste("Expected  ","-",log[10],"(P)")), type = "n", cex.main=0.9, mgp=c(1.5,0.2,0), tcl=-0.1, bty="l")
title(title,adj=0)
shade(-log10(expect),-log10(lower),-log10(expect),-log10(upper), color = "gray")
abline(0,1,col="white",lwd=2)
points(-log10(expect),-log10(p2), pch=20, cex=0.8, col=2)
return(c(xlim,ylim))
}

qqplot_CI=function(pval, title, cols, pchs,ylim=0, lab=1.4, axis=1.2){
  par(mgp=c(5,1,0))
  num=dim(pval)[2]
  n  <- dim(pval)[1]
  k  <- c(1:n)
  alpha   <- 0.05
  lower   <- qbeta(alpha/2,k,n+1-k)
  upper   <- qbeta((1-alpha/2),k,n+1-k)
  expect  <- (k-0.5)/n
  biggest <- ceiling(-log10(expect))
  for(i in 1:num)
  {
      p1 <- pval[,i]
      p2 <- sort(p1)
      biggest <- ceiling(max(-log10(p2),biggest))
      
  }
  shade <- function(x1, y1, x2, y2, color = col.shade) {
    n <- length(x2)
    polygon(c(x1, x2[n:1]), c(y1, y2[n:1]), border = NA, col = color)
  }
  xlim=max(-log10(expect)+0.1);
  if(ylim==0) ylim=biggest;
i=1
p1 <- pval[,i]
      p2 <- sort(p1)
      plot(-log10(expect),-log10(p2),xlim=c(0,xlim),ylim=c(0,ylim),
      ylab=expression(paste("Observed ", "-", log[10], "(", italic(P), "-value)", sep="")),
      xlab=expression(paste("Expected ", "-", log[10], "(", italic(P), "-value)", sep="")),
      type = "n", mgp=c(1.5,0.2,0), tcl=-0.1,
      bty="l", cex.lab=lab, cex.axis=axis)
      points(-log10(expect),-log10(p2), pch=pchs[i], cex=0.6, col=cols[i]) 

 shade(-log10(expect),-log10(lower),-log10(expect),-log10(upper), color = "gray88")
  abline(0,1,col="white",lwd=2)
  for(i in 1:num)
  {
      p1 <- pval[,i]
      p2 <- sort(p1)
      par(new=TRUE)
      plot(-log10(expect),-log10(p2),xlim=c(0,xlim),ylim=c(0,ylim),
      ylab=expression(paste("Observed ", "-", log[10], "(", italic(P), "-value)", sep="")),
      xlab=expression(paste("Expected ", "-", log[10], "(", italic(P), "-value)", sep="")),
      type = "n", mgp=c(1.5,0.2,0), tcl=-0.1,
      bty="l", cex.lab=lab, cex.axis=axis)
      points(-log10(expect),-log10(p2), pch=pchs[i], cex=0.6, col=cols[i])
   
  }
  
   
  legend("topleft", colnames(pval), col = cols,
  text.col = cols, pch = pchs,cex=1.5)
  title( main=title, adj=0)  
  return(c(xlim,ylim))
}

