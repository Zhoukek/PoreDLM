#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else "/mnt/zzbnew/rnamodel/liujiaheng/genome/analysis/s18"
plot_dir <- file.path(root, "plot")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

metrics <- read.csv(file.path(root, "metrics.csv"), stringsAsFactors = FALSE, check.names = FALSE)
pred <- read.csv(file.path(root, "chr16_window_predictions.csv"), stringsAsFactors = FALSE, check.names = FALSE)
site <- read.csv(file.path(root, "chr16_site_predictions.csv"), stringsAsFactors = FALSE, check.names = FALSE)
boot <- read.csv(file.path(root, "bootstrap_ci.csv"), stringsAsFactors = FALSE, check.names = FALSE)

models <- c("V600_Apple", "V610_Apple", "V003_Stone")
model_labels <- c(V600_Apple = "V600 Apple", V610_Apple = "V610 Apple", V003_Stone = "V003 Stone")
cols <- c(V600_Apple = "#009E73", V610_Apple = "#0072B2", V003_Stone = "#D55E00")

metric_value <- function(dataset, model, metric) {
  x <- metrics[metrics$dataset == dataset & metrics$model == model, metric]
  if (length(x) == 0) NA_real_ else as.numeric(x[[1]])
}

roc_curve <- function(y, score) {
  keep <- is.finite(y) & is.finite(score)
  y <- as.integer(y[keep])
  score <- score[keep]
  o <- order(score, decreasing = TRUE, method = "radix")
  yy <- y[o]
  tp <- cumsum(yy)
  fp <- cumsum(1 - yy)
  list(x = c(0, fp / max(sum(1 - yy), 1), 1), y = c(0, tp / max(sum(yy), 1), 1))
}

pr_curve <- function(y, score) {
  keep <- is.finite(y) & is.finite(score)
  y <- as.integer(y[keep])
  score <- score[keep]
  o <- order(score, decreasing = TRUE, method = "radix")
  yy <- y[o]
  tp <- cumsum(yy)
  fp <- cumsum(1 - yy)
  list(x = c(0, tp / max(sum(yy), 1)), y = c(1, tp / pmax(tp + fp, 1)))
}

draw_window_roc <- function() {
  y <- pred$label
  plot(c(0, 1), c(0, 1), type = "n", xlab = "False-positive rate", ylab = "True-positive rate",
       main = "A  chr16 window ROC", xaxs = "i", yaxs = "i")
  abline(0, 1, lty = 2, col = "grey60")
  for (model in models) {
    curve <- roc_curve(y, pred[[paste0(model, "_prob")]])
    lines(curve$x, curve$y, lwd = 2.2, col = cols[[model]])
  }
  legend("bottomright", legend = sprintf("%s  %.3f", model_labels[models],
         sapply(models, function(m) metric_value("chr16_window", m, "auroc"))),
         col = unname(cols[models]), lwd = 2, bty = "n", cex = 0.85)
}

draw_window_pr <- function() {
  y <- pred$label
  plot(c(0, 1), c(0, 1), type = "n", xlab = "Recall", ylab = "Precision",
       main = "B  chr16 window precision-recall", xaxs = "i", yaxs = "i")
  abline(h = mean(y), lty = 2, col = "grey60")
  for (model in models) {
    curve <- pr_curve(y, pred[[paste0(model, "_prob")]])
    lines(curve$x, curve$y, lwd = 2.2, col = cols[[model]])
  }
  legend("topright", legend = sprintf("%s  %.3f", model_labels[models],
         sapply(models, function(m) metric_value("chr16_window", m, "auprc"))),
         col = unname(cols[models]), lwd = 2, bty = "n", cex = 0.85)
}

draw_window_metrics <- function() {
  wanted <- c("auroc", "auprc", "balanced_accuracy", "accuracy", "mcc")
  values <- sapply(wanted, function(metric) sapply(models, function(model)
    metric_value("chr16_window", model, metric)))
  barplot(values, beside = TRUE, ylim = c(0, 1), col = unname(cols[models]),
          names.arg = c("AUROC", "AUPRC", "Balanced ACC", "ACC", "MCC"),
          ylab = "Score", main = "C  Window-level metrics", las = 2)
  legend("topright", legend = unname(model_labels[models]), fill = unname(cols[models]), bty = "n", cex = 0.8)
}

draw_site_roc <- function() {
  y <- site$label
  plot(c(0, 1), c(0, 1), type = "n", xlab = "False-positive rate", ylab = "True-positive rate",
       main = "D  chr16 site ROC", xaxs = "i", yaxs = "i")
  abline(0, 1, lty = 2, col = "grey60")
  for (model in models) {
    curve <- roc_curve(y, site[[model]])
    lines(curve$x, curve$y, lwd = 2.2, col = cols[[model]])
  }
  legend("bottomright", legend = sprintf("%s  %.3f", model_labels[models],
         sapply(models, function(m) metric_value("chr16_site_all", m, "auroc"))),
         col = unname(cols[models]), lwd = 2, bty = "n", cex = 0.85)
}

draw_site_metrics <- function() {
  all_values <- sapply(models, function(model) metric_value("chr16_site_all", model, "auroc"))
  ge2_values <- sapply(models, function(model) metric_value("chr16_site_coverage_ge2", model, "auroc"))
  values <- rbind(all_values, ge2_values)
  barplot(t(values), beside = TRUE, ylim = c(0, 1), col = unname(cols[models]),
          names.arg = c("All sites", "Coverage >= 2"), ylab = "AUROC",
          main = "E  Site-level AUROC")
  legend("topright", legend = unname(model_labels[models]), fill = unname(cols[models]), bty = "n", cex = 0.8)
}

draw_baseline_ablation <- function() {
  values <- sapply(models, function(model) c(
    target = metric_value("chr16_window", model, "auroc"),
    chr19 = metric_value("chr16_window_no_target_baseline", model, "auroc")
  ))
  barplot(t(values), beside = TRUE, ylim = c(0, 1), col = unname(cols[models]),
          names.arg = c("Target chr16 baseline", "Chr19 baseline"), ylab = "AUROC",
          main = "F  Baseline sensitivity", las = 2)
  legend("topright", legend = unname(model_labels[models]), fill = unname(cols[models]), bty = "n", cex = 0.8)
}

png(file.path(plot_dir, "figure_s18_three_model_comparison.png"), width = 2600, height = 1900, res = 220)
par(mfrow = c(2, 3), mar = c(5, 5, 3, 1), oma = c(0, 0, 2, 0))
draw_window_roc()
draw_window_pr()
draw_window_metrics()
draw_site_roc()
draw_site_metrics()
draw_baseline_ablation()
mtext("S18: V600 Apple, V610 Apple and V003 Stone on strict 200x 7-mer cohorts", outer = TRUE, cex = 1.25, font = 2)
dev.off()

draw_score_boxplot <- function() {
  boxes <- list()
  labels <- character(0)
  fills <- character(0)
  for (model in models) {
    boxes[[paste0(model, "_0")]] <- pred[[paste0(model, "_prob")]][pred$label == 0]
    boxes[[paste0(model, "_1")]] <- pred[[paste0(model, "_prob")]][pred$label == 1]
    labels <- c(labels, paste0(model_labels[[model]], " 0%"), paste0(model_labels[[model]], " 100%"))
    fills <- c(fills, adjustcolor(cols[[model]], 0.35), adjustcolor(cols[[model]], 0.75))
  }
  boxplot(boxes, names = labels, col = fills, las = 2, cex.axis = 0.75,
          ylab = "Predicted modification probability", main = "A  Window score distributions")
}

draw_calibration <- function() {
  plot(c(0, 1), c(0, 1), type = "n", xlab = "Predicted probability", ylab = "Observed positive fraction",
       main = "B  Window calibration")
  abline(0, 1, lty = 2, col = "grey60")
  for (model in models) {
    p <- pred[[paste0(model, "_prob")]]
    bins <- cut(p, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE)
    mean_p <- tapply(p, bins, mean)
    mean_y <- tapply(pred$label, bins, mean)
    keep <- is.finite(mean_p) & is.finite(mean_y)
    lines(mean_p[keep], mean_y[keep], type = "b", pch = 16, lwd = 2, col = cols[[model]])
  }
  legend("topleft", legend = unname(model_labels[models]), col = unname(cols[models]),
         lwd = 2, pch = 16, bty = "n", cex = 0.8)
}

draw_pairwise_delta <- function() {
  rows <- boot[boot$metric == "auroc" & grepl("_minus_", boot$model), , drop = FALSE]
  if (nrow(rows) == 0) {
    plot.new()
    title("C  Pairwise AUROC delta: unavailable")
    return()
  }
  if (nrow(rows) == 6) {
    values <- cbind(rows$estimate[1:3], rows$estimate[4:6])
    rownames(values) <- c("V600 - V610", "V600 - V003", "V610 - V003")
    colnames(values) <- c("Window", "Site")
    y_range <- range(c(rows$ci_low, rows$ci_high, 0), finite = TRUE)
    pad <- max(diff(y_range) * 0.15, 0.02)
    mids <- barplot(values, beside = TRUE, ylim = y_range + c(-pad, pad),
                    col = c("#999999", "#666666", "#444444"),
                    names.arg = c("Window", "Site"),
                    ylab = "AUROC difference", main = "C  Pairwise AUROC delta")
    arrows(as.vector(mids), rows$ci_low, as.vector(mids), rows$ci_high,
           angle = 90, code = 3, length = 0.05)
    abline(h = 0, lty = 2, col = "grey40")
    legend("topright", legend = rownames(values),
           fill = c("#999999", "#666666", "#444444"), bty = "n", cex = 0.78)
    return()
  }
  values <- rows$estimate
  names(values) <- rows$model
  y_range <- range(c(rows$ci_low, rows$ci_high, 0), finite = TRUE)
  pad <- max(diff(y_range) * 0.15, 0.02)
  delta_labels <- gsub("_minus_", " - ", rows$model, fixed = TRUE)
  mids <- barplot(values, ylim = y_range + c(-pad, pad), col = "#777777",
                  ylab = "AUROC difference", main = "C  Pairwise AUROC delta",
                  names.arg = delta_labels, cex.names = 0.8)
  arrows(mids, rows$ci_low, mids, rows$ci_high, angle = 90, code = 3, length = 0.05)
  abline(h = 0, lty = 2, col = "grey40")
}

png(file.path(plot_dir, "figure_s18_score_calibration.png"), width = 2500, height = 1450, res = 220)
par(mfrow = c(1, 3), mar = c(7, 5, 3, 1))
draw_score_boxplot()
draw_calibration()
draw_pairwise_delta()
dev.off()

cat(sprintf("plots_written=%s\n", plot_dir))
