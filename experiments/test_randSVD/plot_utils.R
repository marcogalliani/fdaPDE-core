library(ggplot2)
library(scales)
library(magrittr)
library(dplyr)

grouped_boxplot <- function(test_results, 
                            title,
                            var_name.x,var_name.y,
                            group_name, color_map,
                            limits.y=NULL, log.y=F,
                            lines=F){
  plot <- ggplot(test_results, aes(x=get(var_name.x), y=get(var_name.y), fill=get(group_name))) + 
    geom_boxplot(outlier.shape = NA) +
    scale_fill_manual(values = color_map) +
    theme_light() +
    labs(title = title, x=var_name.x, y=var_name.y) 
  
  #options
  if(lines){
    plot <- plot +
      stat_summary(
        fun = median,
        geom = 'line',
        aes(group = get(group_name), color = get(group_name)),
        position = position_dodge(width = 0.85) 
      ) +
      scale_color_manual(values = color_map)
  }
  if(log.y){
    plot <- plot + scale_y_log10(limits=limits.y)
  }
  plot <- plot + guides(color="none",fill="none")
  return(plot)
}


grouped_lineplot <- function(test_results, 
                             title,
                             var_name.x,var_name.y,
                             group_name, color_map,
                             log.x=F,log.y=F,
                             complexity.lines=NULL){
  test_results.grouped <- test_results %>% 
    group_by(group=get(group_name), x=get(var_name.x)) %>%
    summarize(y = median(get(var_name.y))) %>%
    mutate(x = as.numeric(as.character(x)))
  
  plot <- ggplot(test_results.grouped, 
                 aes(x=x, y=y, color=group)) + 
    geom_line() +
    theme_light() +
    guides(color = "none") +
    labs(title = title, x=var_name.x, y=var_name.y) +
    scale_color_manual(values=color_map)
  
  if(log.x){
    plot <- plot + scale_x_log10()
  }
  if(log.y){
    plot <- plot + scale_y_log10() 
  }
  if(log.x & log.y & !is.null(complexity.lines)){
    min.x <- min(test_results.grouped$x)
    min.y <- min(test_results.grouped$y)
    for(line in complexity.lines){
      plot <- plot + geom_abline(intercept = log10(min.y/min.x^line), slope = line,
                                 linetype = "dotted", color = "grey", linewidth = 0.3)
      label_x <- 0.8 * max(test_results.grouped$x)  # Place label near the right edge of the plot
      label_y <- 10^(log10(min.y/min.x^line) + line * log10(label_x))
      
      plot <- plot + annotate("text", 
                              x = label_x, 
                              y = label_y, 
                              label = paste(var_name.x, line, sep = "^"), 
                              hjust = -0.1, color = "grey", size = 3)
    }
  }
  return(plot)
}
