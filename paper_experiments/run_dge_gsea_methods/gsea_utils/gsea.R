library("msigdbr")
library("data.table")
library("tidyverse")
library("fgsea")
library("optparse")


option_list = list(
  make_option(
    c("--input_file"),
    type = "character",
    default = "input.csv",
    help = "csv input file",
    metavar = "character"
  ),
  make_option(
    c("--output_file"),
    type = "character",
    default = "gsea.csv",
    help = "csv output file",
    metavar = "character"
  )
)

opt_parser = OptionParser(option_list = option_list)

opt = parse_args(opt_parser)



# Set seed to ensure that results are actually comparable
set.seed(42)

# _______________________________
# Import  data
# _______________________________



# Load (Py)DESeq2 results
# ______________

diffexp_df <-  data.table::fread(opt$input_file)


# MSigDB Signatures
# _______________________________
all_gene_sets <- msigdbr::msigdbr(species = "Homo sapiens")

# Choose collection
collection <- "C2"
sub_collection <- "CP:REACTOME"

msigdbr_df <- all_gene_sets %>%
  dplyr::filter(gs_cat == collection, gs_subcat == sub_collection)

msigdbr_list <-
  base::split(x = msigdbr_df$ensembl_gene, f = msigdbr_df$gs_id)



# _______________________________
# fGSEA analysis
# _______________________________

rank_on <- "stat"
diffexp_df <- diffexp_df %>% dplyr::arrange(.data[[rank_on]])

# Create rank vector
ranks <- diffexp_df[[rank_on]]
names(ranks) <- diffexp_df$gene


# Run fgsea for GSEA analysis. Order results by padj
print("Running fgsea...")
fgsea_res <- fgsea::fgsea(
  pathways = msigdbr_list,
  stats = ranks,
  minSize = 15,
  maxSize = 500,
  eps = 0.0
) %>% dplyr::arrange(padj)

print("Fgsea finished.")

# Prepare for export (correct leading edge list)
fgsea_res_export <- as.matrix(fgsea_res %>%
                                dplyr::mutate(leadingEdge = sapply(leadingEdge, function(x)
                                  paste(unlist(x), collapse = ","))))

write.table(
  data.frame(fgsea_res_export),
  file = opt$output_file,
  row.names = FALSE,
  sep = "\t",
  quote = F
)
