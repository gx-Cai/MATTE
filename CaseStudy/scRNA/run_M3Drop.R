#BiocManager::install('M3Drop')
library(M3Drop)
library(data.table)

run_M3 <- function(data_dir,subsample_label_dir){
    label <- read.csv(subsample_label_dir,row.names = 1)
    cells <- row.names(label)

    counts <- data.table::fread(data_dir,data.table = F)
    cells_ <- counts$V1
    row.names(counts) <- cells_
    counts <- counts[,-1]

    counts <- counts[cells,]
    genes <- colnames(counts)
    counts <- data.table::transpose(counts)
    row.names(counts) <- genes
    colnames(counts) <- cells
    rm(cells_)

    norm <- M3DropConvertData(counts, is.counts=TRUE)
    norm <- M3DropConvertData(log2(norm+1), is.log=TRUE, pseudocount=1)

    M3Drop_genes <- M3DropFeatureSelection(norm, mt_method="fdr", mt_threshold=0.01)
    ## Other Feature Selection Methods
    HVG <- BrenneckeGetVariableGenes(norm)
    rm(norm)
    ## NBumi
    count_mat <- NBumiConvertData(counts, is.counts=TRUE)
    DANB_fit <- NBumiFitModel(count_mat)
    NBDropFS <- NBumiFeatureSelectionCombinedDrop(DANB_fit, method="fdr", qval.thres=0.01, suppress.plot=FALSE)

    write.csv(x=M3Drop_genes,file='M3drop_genes.csv')
    write.csv(x=NBDropFS,file='NBDropsFS_genes.csv')
    write.csv(x=HVG,file='BrenneckeHV_genes.csv')
}
