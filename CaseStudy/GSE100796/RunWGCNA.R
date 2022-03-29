# Read data & Preprocess

datadir = "GSE100796/GSE100796_TPM_fixed.txt"
data = read.table(datadir,row.names = 1,header = 1)

genes <- row.names(data)
samples <- colnames(data)
data.human <- data[genes,samples[grepl('Human',samples)]]
data.chimp <- data[genes,samples[grepl('Chimpanzee',samples)]]
rm(data)

# Build Network

library(doParallel)
registerDoParallel(cores=16)
library(WGCNA)
library(dplyr)
run_WGCNA <- function(data){
  Expr <- t(data[order(apply(data,1,mad), decreasing = T),])
  sft = pickSoftThreshold(Expr,verbose = 5)
  power = sft$powerEstimate
  if (is.na(sft$powerEstimate)){
    print('estamate NULL, set power 6')
    power=6
  }
  net = blockwiseModules(
    Expr,
    power = power,
    TOMType = "unsigned",minModuleSize = 100,
    numericLabels = TRUE, pamRespectsDendro = FALSE,
    saveTOMs = F, 
    verbose = 3
  )
  net
}

human_net <- run_WGCNA(data.human)
chimp_net <- run_WGCNA(data.chimp)
# Prevervation 


mData <- list(
  Chimp = list(data=t(data.chimp[genes,])),
  Human = list(data = t(data.human[genes,])))
mColor <- list(
  Chimp=labels2colors(chimp_net$colors[genes]),
  Human = labels2colors(human_net$colors[genes]))

mpres <- modulePreservation(
  multiData = mData,
  multiColor = mColor,
  nPermutations = 100,verbose = 5,calculateQvalue=T
)

labels <- as.data.frame(mColor)
row.names(labels) <- genes
module_ps <- mpres$accuracy$observedFisherPvalues$ref.Chimp$inColumnsAlsoPresentIn.Human
write.csv(labels,'./WGCNA_labels.csv')
write.csv(module_ps,'./preserve_pvalue.csv')


library(RColorBrewer)
module_counts <- mpres$accuracy$observedCounts$ref.Chimp$inColumnsAlsoPresentIn.Human
make.valid.color <- function(x){
  paste0('AA',x)
}
module_ps <- mpres$accuracy$observedFisherPvalues$ref.Chimp$inColumnsAlsoPresentIn.Human
module_ps <- -log10(module_ps)
labeledHeatmap(
  module_counts,
  xLabels = colnames(module_ps) %>% make.valid.color,
  yLabels = rownames(module_ps) %>% make.valid.color,
  xSymbols = colnames(module_ps),
  ySymbols = rownames(module_ps),
  textMatrix = module_counts,
  colors = brewer.pal(n=9,name = 'Blues') ,
  legendLabel='-log(p)'
  )



# Differential connectivity

chimp.k <- intramodularConnectivity.fromExpr(
  datExpr = t(data.chimp[genes,]),
  colors = chimp_net$colors[genes]
)
chimp.kin <- chimp.k$kWithin
names(chimp.kin) <- genes

human.k <- intramodularConnectivity.fromExpr(
  datExpr = t(data.human[genes,]),
  colors = human_net$colors[genes]
)
human.kin <- human.k$kWithin
names(human.kin) <- genes

DC <- log10(human.kin/)
