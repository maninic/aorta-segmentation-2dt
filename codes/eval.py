from Options.train_options import TrainOptions
from Datahandler.data_utils import data_from_dict
import monai
from torch.utils.tensorboard.writer import SummaryWriter
from pathlib import Path
import torch
import pandas as pd
import os

options = TrainOptions()
info = options.gather_options()
d_folder = Path(info.model_path)
dev = "cuda:"+str(info.gpu_ids)

d_test, test_dataset, test_dataloader = data_from_dict(d_folder,phase='test')
folder = d_folder / 'evaluation'
device = torch.device(dev if torch.cuda.is_available() else "cpu")
post_trans = monai.transforms.Compose([
    monai.transforms.EnsureType(), 
    monai.transforms.Activations(sigmoid=True), 
    monai.transforms.AsDiscrete(threshold=info.threshold), 
    monai.transforms.KeepLargestConnectedComponent(applied_labels=[1])
    ])
#check if I need to restore the original shapes (add 3rd dim)
p_folder = folder / 'predictions'
saver_pred = monai.transforms.SaveImage(output_dir=p_folder, output_ext=".nii.gz", output_postfix="pred")
saver_im = monai.transforms.SaveImage(output_dir=p_folder, output_ext=".nii.gz", output_postfix="im")
saver_label = monai.transforms.SaveImage(output_dir=p_folder, output_ext=".nii.gz", output_postfix="label")

metrics = info.eval_metrics
metrics_df = pd.DataFrame(columns=metrics)
metrics_list = []
if 'DICE' in metrics:
    dice_metric = monai.metrics.DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    metrics_list.append(dice_metric)
if 'hausdorff' in metrics:
    hausdorff_distance = monai.metrics.HausdorffDistanceMetric(include_background=True, distance_metric="euclidean", get_not_nans=False)
    metrics_list.append(hausdorff_distance)
if 'ASSD' in metrics:
    AvSurfDist = monai.metrics.SurfaceDistanceMetric(include_background=False, symmetric=True, distance_metric='euclidean') #Before symmetric=False
    metrics_list.append(AvSurfDist)
if 'sens' in metrics:
    sensitivity = monai.metrics.ConfusionMatrixMetric(include_background=False, metric_name='sensitivity', compute_sample=False, get_not_nans=False)
    metrics_list.append(sensitivity)
if 'prec' in metrics:
    precision = monai.metrics.ConfusionMatrixMetric(include_background=False, metric_name='precision', compute_sample=False, get_not_nans=False)
    metrics_list.append(precision)
if 'acc' in metrics:
    accuracy = monai.metrics.ConfusionMatrixMetric(include_background=False, metric_name='accuracy', compute_sample=False, get_not_nans=False)
    metrics_list.append(accuracy)
if 'mr' in metrics:
    missrate = monai.metrics.ConfusionMatrixMetric(include_background=False, metric_name='miss rate', compute_sample=False, get_not_nans=False)
    metrics_list.append(missrate)

log_dir = folder / "log_inference"
writer = SummaryWriter(log_dir=log_dir)

model_file = d_folder / 'model_best'
model = torch.load(model_file)
model.eval()
roi_size = (info.patch_size[0],info.patch_size[1],info.Tresample)
sw_batch_size = 1
with torch.no_grad():
    i = 0
    for test_data in test_dataloader:
        i += 1
        case_info = []
        test_images, test_labels = test_data["inputs"].to(device), test_data["mask"].to(device)
        test_outputs = monai.inferers.sliding_window_inference(test_images,roi_size, sw_batch_size, model, overlap=0.5, mode="gaussian")
        test_outputs = [post_trans(i) for i in monai.data.utils.decollate_batch(test_outputs)]
        for name,m in zip(metrics,metrics_list):
            print(name)
            m(y_pred=test_outputs, y=test_labels)
            if name == "DICE" or name == "hausdorff" or name == "ASSD":
                val = m.aggregate().item()
                print(val)
            else:
                val = m.aggregate()[0]
                print(val)
            case_info.append(val)
            writer.add_scalar(name,val,i)
            m.reset()
        test_images = monai.data.utils.decollate_batch(test_images)
        test_labels = monai.data.utils.decollate_batch(test_labels)
        if info.save_predictions:
            for test_output,test_image,test_label in zip(test_outputs,test_images,test_labels):
                saver_pred(test_output)
                saver_im(test_image)
                saver_label(test_label)
        metrics_df = pd.concat([metrics_df,pd.DataFrame([case_info], columns=metrics)], ignore_index=True)      
csv_file =  os.path.join(folder,"test_metrics.csv")
metrics_df.to_csv(csv_file)