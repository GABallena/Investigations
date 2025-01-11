# Libraries
library(ggraph)
library(igraph)
library(tidyverse)
library(RColorBrewer)

# Helper functions
process_arg_genes <- function(x) {
  ifelse(is.na(x) | !grepl("^\\d+$", x), 0, as.numeric(x))
}

create_color_palette <- function() {
  c(
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", 
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
    "#E69F00A0", "#56B4E9A0", "#009E73A0", "#F0E442A0", 
    "#0072B2A0", "#D55E00A0", "#CC79A7A0", "#000000A0"
  )
}

# Data loading and validation
tryCatch({
  setwd("~/Desktop/Side")
  data <- read_tsv("IMGPR/IMGPR_plasmid_data.tsv") %>%
    mutate(arg_genes = map_dbl(arg_genes, process_arg_genes))
  
  required_cols <- c("ecosystem", "host_taxonomy", "plasmid_id", "topology", "arg_genes")
  missing_cols <- setdiff(required_cols, colnames(data))
  if (length(missing_cols) > 0) {
    stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
  }
}, error = function(e) {
  stop(paste("Error loading data:", e$message))
})

# Data preparation
hierarchy <- data %>%
  select(ecosystem, host_taxonomy, plasmid_id) %>%
  gather(key = "type", value = "from", ecosystem, host_taxonomy) %>%
  select(-type) %>%
  rename(to = plasmid_id) %>%
  distinct()

connect <- data %>%
  select(ecosystem, host_taxonomy, plasmid_id, topology, arg_genes) %>%
  gather(key = "type", value = "from", ecosystem, host_taxonomy) %>%
  select(-type) %>%
  rename(to = plasmid_id)

# Create vertices dataframe
vertices <- data.frame(
  name = unique(c(as.character(hierarchy$from), as.character(hierarchy$to)))
) %>%
  left_join(
    data %>%
      group_by(plasmid_id) %>%
      summarise(value = sum(arg_genes, na.rm = TRUE)),
    by = c("name" = "plasmid_id")
  ) %>%
  mutate(
    value = coalesce(value, 0),
    group = case_when(
      name %in% data$plasmid_id ~ data$topology[match(name, data$plasmid_id)],
      name %in% data$ecosystem ~ "Ecosystem",
      name %in% data$host_taxonomy ~ "Host Taxonomy",
      TRUE ~ "Other"
    )
  )

# Graph creation
mygraph <- graph_from_data_frame(hierarchy, vertices = vertices)
from <- match(connect$from, vertices$name)
to <- match(connect$to, vertices$name)

# Base plot configuration
base_plot <- ggraph(mygraph, layout = 'dendrogram', circular = TRUE) +
  theme_void()

# Final visualization function
create_final_plot <- function(base_plot, vertices, from, to) {
  base_plot +
    geom_conn_bundle(
      data = get_con(from = from, to = to),
      width = 1,
      alpha = 0.2,
      aes(colour = ..index..)
    ) +
    geom_node_point(
      aes(
        filter = leaf,
        x = x * 1.05,
        y = y * 1.05,
        colour = group,
        size = value
      ),
      alpha = 0.8
    ) +
    scale_colour_manual(
      values = rep(create_color_palette(), 
                  length.out = length(unique(vertices$group))),
      name = "Topology"
    ) +
    scale_size_continuous(
      range = c(0.1, 10),
      name = "ARG Genes"
    ) +
    theme(
      legend.position = "right",
      plot.margin = margin(20, 20, 20, 20),
      plot.background = element_rect(fill = "white", color = NA)
    ) +
    guides(
      colour = guide_legend(override.aes = list(size = 4)),
      size = guide_legend(override.aes = list(alpha = 0.8))
    )
}

# Generate and save final plot
final_plot <- create_final_plot(base_plot, vertices, from, to)
ggsave("final_visualization.png", plot = final_plot, width = 10, height = 10)


