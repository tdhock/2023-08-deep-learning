import plotnine as p9
import numpy as np
import torch
import pdb
import math
import matplotlib
matplotlib.use("agg")
import pandas as pd
data_info_dict = {
    "forest_fires":("../data/forestfires.csv",",",True),
    "air_foil":("../data/airfoil_self_noise.tsv","\t",False),
}
data_dict = {}
hist_df_list = []
for data_name,(file_name,sep,log_trans) in data_info_dict.items():
    data_df = pd.read_csv(file_name,sep=sep,header=0)
    data_nrow, data_ncol = data_df.shape
    label_col_num = data_ncol-1
    data_label_vec = data_df.iloc[:,label_col_num]
    if log_trans:
        data_label_vec = np.log(data_label_vec+1)
    label_sd = math.sqrt(data_label_vec.var())
    standard_label_vec = (
        data_label_vec-data_label_vec.mean()
    )/label_sd
    is_feature_col = (
        np.arange(data_ncol) != label_col_num
    ) & (
        data_df.dtypes != "object"
    )
    data_features = data_df.loc[:,is_feature_col]
    feature_nrow, feature_ncol= data_features.shape
    feature_mean = data_features.mean().to_numpy().reshape(1,feature_ncol)
    feature_std = data_features.std().to_numpy().reshape(1,feature_ncol)
    feature_scaled = (data_features-feature_mean)/feature_std
    print("%s %s"%(data_name, data_features.shape))
    input_tensor = torch.from_numpy(
        feature_scaled.to_numpy()
    ).float()
    output_tensor = torch.from_numpy(
        standard_label_vec.to_numpy()
    ).float().reshape(data_nrow, 1)
    data_dict[data_name] = (input_tensor, output_tensor)
    hist_df_list.append(pd.DataFrame({
        "data_name":data_name,
        "label":standard_label_vec
    }))
hist_df = pd.concat(hist_df_list)
gg_hist = p9.ggplot()+\
    p9.theme(text=p9.element_text(size=30))+\
    p9.geom_histogram(
        p9.aes(
            x="label"
        ),
        data=hist_df)+\
    p9.facet_wrap(
        ["data_name"], 
        labeller="label_both",
        scales="free")
gg_hist.save("09_homework_hist.png", width=15, height=5)
class TorchModel(torch.nn.Module):
    def __init__(self, units_per_layer):
        super(TorchModel, self).__init__()
        seq_args = []
        second_to_last = len(units_per_layer)-1
        for layer_i in range(second_to_last):
            next_i = layer_i+1
            layer_units = units_per_layer[layer_i]
            next_units = units_per_layer[next_i]
            seq_args.append(torch.nn.Linear(layer_units, next_units))
            if layer_i < second_to_last-1:
                seq_args.append(torch.nn.ReLU())
        self.stack = torch.nn.Sequential(*seq_args)
    def forward(self, features):
        return self.stack(features)
class CSV(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __getitem__(self, item):
        return self.features[item,:], self.labels[item]
    def __len__(self):
        return len(self.labels)
class TorchLearner:
    def __init__(
            self, units_per_layer, step_size=0.1, 
            batch_size=20, max_epochs=100):
        self.max_epochs = max_epochs
        self.batch_size=batch_size
        self.model = TorchModel(units_per_layer)
        self.loss_fun = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=step_size)
    def fit(self, split_data_dict):
        ds = CSV(
            split_data_dict["subtrain"]["X"], 
            split_data_dict["subtrain"]["y"])
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=True)
        train_df_list = []
        for epoch_number in range(self.max_epochs):
            print(epoch_number)
            for batch_features, batch_labels in dl:
                self.optimizer.zero_grad()
                loss_value = self.loss_fun(
                    self.model(batch_features), batch_labels)
                loss_value.backward()
                self.optimizer.step()
            for set_name, set_data in split_data_dict.items():
                pred_vec = self.model(set_data["X"])
                set_loss_value = self.loss_fun(pred_vec, set_data["y"])
                train_df_list.append(pd.DataFrame({
                    "set_name":[set_name],
                    "loss":float(set_loss_value),
                    "epoch":[epoch_number]
                }))
        self.train_df = pd.concat(train_df_list)
    def predict(self, test_features):
        return self.model(test_features)
learner  = TorchLearner([feature_ncol, 100, 10, 1])
class TorchLearnerCV:
    def __init__(self, n_folds = 3, units_per_layer=[data_ncol,1]):
        self.units_per_layer = units_per_layer
        self.n_folds = n_folds
    def fit(self, train_features, train_labels):
        train_nrow, train_ncol = train_features.shape
        times_to_repeat=int(math.ceil(train_nrow/self.n_folds))
        fold_id_vec = np.tile(torch.arange(self.n_folds), times_to_repeat)[:train_nrow]
        np.random.shuffle(fold_id_vec)
        cv_data_list = []
        for validation_fold in range(self.n_folds):
            is_split = {
                "subtrain":fold_id_vec != validation_fold,
                "validation":fold_id_vec == validation_fold
                }
            split_data_dict = {}
            for set_name, is_set in is_split.items():
                set_y = output_tensor[is_set]
                split_data_dict[set_name] = {
                    "X":input_tensor[is_set,:],
                    "y":set_y}
            learner = TorchLearner(self.units_per_layer)
            learner.fit(split_data_dict)
            cv_data_list.append(learner.train_df)
        self.cv_data = pd.concat(cv_data_list)
        self.train_df = self.cv_data.groupby(["set_name","epoch"]).mean().reset_index()
        valid_df = self.train_df.query("set_name=='validation'")
        best_epochs = valid_df["loss"].argmin()
        self.min_df = valid_df.query("epoch==%s"%best_epochs)
        self.final_learner = TorchLearner(self.units_per_layer, max_epochs=best_epochs)
        self.final_learner.fit({"subtrain":{"X":train_features,"y":train_labels}})
    def predict(self, test_features):
        return self.final_learner.predict(test_features)
model = TorchModel([feature_ncol, 100, 10, 1])
pred_vec = model(input_tensor)
loss_fun = torch.nn.MSELoss()
loss_value = loss_fun(pred_vec, output_tensor)
loss_value.backward() #computes gradients!@!!
learner=TorchLearnerCV()
learner.fit(input_tensor, output_tensor)
import plotnine as p9
gg = p9.ggplot()+\
    p9.theme(text=p9.element_text(size=30))+\
    p9.geom_line(
        p9.aes(
            x="epoch",
            y="loss",
            color="set_name"
        ),
        data=learner.train_df)+\
    p9.geom_point(
        p9.aes(
            x="epoch",
            y="loss",
            color="set_name"
        ),
        data=learner.min_df)
gg.save("06_homework.png", width=10, height=4)
