library(data.table)
library(animint2)
sim.dt <- fread("09-regression-sim.csv")
set.seed(2)
hidden.units <- 100
sim.dt[set_name=="train", split := rep(c("subtrain","validation"),l=.N)]
maxit.vec <- as.integer(10^seq(0.5, 3, l=20))
error.dt.list <- list()
pred.dt.list <- list()
grid.vec <- seq(-1, 1, l=20)
grid.only <- data.table(expand.grid(sin=grid.vec, cos=grid.vec))
grid.dt.list <- list()
for(maxit in maxit.vec){
  for(inputs in c("degrees01","sin_cos")){
    form <- formula(paste0("label~",sub("_","+",inputs)))
    model <- nnet::nnet(
      form,
      sim.dt[split=="subtrain"],
      size=hidden.units,
      skip=FALSE,
      trace=FALSE,
      linout=TRUE,
      maxit=maxit)
    if(inputs=="sin_cos"){
      grid.dt.list[[paste(maxit)]] <- data.table(
        grid.only,
        sin_cos=maxit,
        prediction=as.numeric(predict(model, grid.only)))
    }
    sim.dt$prediction <- predict(model, sim.dt)
    error.dt.list[[paste(maxit,inputs)]] <- sim.dt[, .(
      maxit, inputs,
      total.loss=sum((prediction-label)^2),
      n.data=.N
    ), by=.(set=ifelse(set_name=="test","test",split))]
    pred.dt.list[[paste(maxit,inputs)]] <- data.table(
      maxit, inputs, sim.dt)
  }
}
grid.dt <- rbindlist(grid.dt.list)
error.dt <- rbindlist(error.dt.list)[, log10.max.iterations := log10(maxit)]
pred.dt <- rbindlist(pred.dt.list)

not.test <- error.dt[set!="test"]
not.test[, loss.thresh := ifelse(
  total.loss>max(total.loss[set=="subtrain"]), Inf, total.loss), 
  by=inputs]
min.dt <- not.test[, .SD[which.min(total.loss)], by=.(set, inputs)]
selected.dt <- min.dt[set=="validation"]
ggplot()+
  geom_line(aes(
    maxit, loss.thresh, color=set),
    data=not.test)+
  geom_point(aes(
    maxit, loss.thresh, color=set),
    data=min.dt)+
  geom_text(aes(
    maxit, loss.thresh, label=paste0("maxit=",maxit)),
    hjust=1,
    data=selected.dt)+
  scale_x_log10()+
  facet_grid(inputs ~ ., labeller=label_both, scales="free")

selected.pred <- pred.dt[
  selected.dt, on=.(maxit, inputs)
][
  prediction>max(label), prediction := Inf
][
  prediction<min(label), prediction := -Inf
]
selected.error <- error.dt[set=="test"][selected.dt, on=.(maxit,inputs)]
sim.long <- melt(
  selected.pred, measure=c("prediction","true"), variable.name="Function")
ggplot()+
  geom_point(aes(
    degrees, label, fill=set_name),
    shape=21,
    data=sim.dt)+
  geom_line(aes(
    degrees, value, color=Function),
    data=sim.long)+
  scale_color_manual(values=c(
    true="black",
    prediction="violet"))+
  facet_grid(. ~ inputs, labeller=label_both)

subtrain <- not.test[set=="subtrain"]
rect.width <- median(
  subtrain[, .(diff=diff(log10.max.iterations)), by=inputs]$diff)/2
pred.long <- melt(
  pred.dt, measure=c("prediction","true"), variable.name="Function")[
  value>max(label), value := Inf
][
  value<min(label), value := -Inf
]
grid.dt[, pred.thresh := ifelse(
  prediction>max(sim.dt$label), max(sim.dt$label), ifelse(
    prediction<min(sim.dt$label), min(sim.dt$label), prediction))]
point.size <- 5
viz <- animint(
  loss=ggplot()+
    ggtitle("loss, select #iterations")+
    scale_color_manual(values=c(
      subtrain="blue",
      validation="deepskyblue"))+
    geom_line(aes(
      log10.max.iterations, loss.thresh, color=set, group=set),
      data=not.test)+
    geom_point(aes(
      log10.max.iterations, loss.thresh, color=set),
      fill="white",
      data=min.dt)+
    geom_text(aes(
      log10.max.iterations, loss.thresh, label=paste0("maxit=",maxit)),
      hjust=1,
      data=selected.dt)+
    geom_tallrect(aes(
      xmin=log10.max.iterations-rect.width,
      xmax=log10.max.iterations+rect.width,
      ymin=-Inf,
      ymax=Inf),
      clickSelects=c(inputs="maxit"),
      data=subtrain,
      alpha=0.5)+
    facet_grid(inputs ~ ., labeller=label_both, scales="free"),
  funs=ggplot()+
    ggtitle("Predictions for #iterations, select point")+
    geom_point(aes(
      degrees, label, fill=set_name),
      shape=21,
      size=point.size,
      alpha=0.6,
      clickSelects="degrees",
      data=sim.dt)+
    geom_line(aes(
      degrees, value, color=Function, group=Function),
      showSelected=c(inputs="maxit"),
      data=pred.long)+
    scale_color_manual(values=c(
      true="black",
      prediction="violet"))+
    xlab("input/feature (degrees)")+
    ylab("output/label")+
    facet_grid(inputs ~ ., labeller=label_both),
  grid=ggplot()+
    ggtitle("sin_cos inputs/predictions, select point")+
    scale_fill_gradient(low="white", high="black")+
    geom_tile(aes(
      sin, cos, fill=pred.thresh),
      showSelected="sin_cos",
      data=grid.dt)+
    geom_point(aes(
      sin, cos, color=set_name),
      size=point.size+2,
      alpha=0.6,
      clickSelects="degrees",
      data=sim.dt)+
    geom_point(aes(
      sin, cos, fill=label),
      size=point.size,
      data=sim.dt))
animint2dir(viz, "09-regression-viz", open.browser = FALSE)
if(FALSE){
  servr::httd("09-regression-viz")
  remotes::install_github("animint/animint2@84-move-gallery-from-blocks-to-gh-pages")
  animint2pages(viz, "animint/figure-nnet-regression-degrees")
}
