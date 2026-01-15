from Options.train_options import TrainOptions
from Datahandler.data_utils import data_from_dict
import monai
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from pathlib import Path

options = TrainOptions()
info = options.gather_options()
dev = "cuda:"+str(info.gpu_ids)

def model_setup():
    device = torch.device(dev if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
            spatial_dims = info.sd,
            in_channels = info.input_nc,
            out_channels = info.output_nc,
            channels = info.channels,
            strides = info.strides,
            num_res_units = info.res_unit,
            dropout = info.dropout,
        ).to(device)
    return device, model

folder = Path(info.model_path)
cases_train, tr_dataset, tr_dataloader = data_from_dict(folder, "training")
cases_val, val_dataset, val_dataloader = data_from_dict(folder, "validation")
device, model = model_setup()
loss_function = monai.losses.DiceCELoss(sigmoid=True, ce_weight=torch.tensor([info.weight_celoss]).to(device))
optimizer = torch.optim.Adam(model.parameters(), info.lr) #info.lr
#setup metrics
dice_metric = monai.metrics.DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
hausdorff_distance = monai.metrics.HausdorffDistanceMetric(include_background=True, distance_metric="euclidean", get_not_nans=False)
post_trans = monai.transforms.Compose([
    monai.transforms.EnsureType(), 
    monai.transforms.Activations(sigmoid=True), 
    monai.transforms.AsDiscrete(threshold=info.threshold), 
    monai.transforms.KeepLargestConnectedComponent(applied_labels=[1])
    ])
if info.writer:
    logdir = folder / 'logfiles'
    writer = SummaryWriter(log_dir=logdir)
model_folder = folder / 'model_best'

options.print_options(info.model_path)

val_interval = info.val_nepoch
val_dice_best = 0
for epoch in range(info.n_epoch):
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in tr_dataloader:
        step += 1
        inputs, labels = batch_data['inputs'].to(device), batch_data["mask"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(tr_dataset) // tr_dataloader.batch_size
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
    epoch_loss/=step
    if epoch % info.print_freq == 0:
        print("-" * 10)
        print(f"epoch {epoch + 1}/{info.n_epoch} average loss: {epoch_loss:.4f}")
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            val_images = None
            val_labels = None
            val_outputs = None
            for val_data in val_dataloader:
                val_images, val_labels = val_data["inputs"].to(device), val_data["mask"].to(device)
                roi_size = (info.patch_size[0],info.patch_size[1],info.Tresample)
                sw_batch_size = 1
                val_outputs = monai.inferers.sliding_window_inference(val_images, roi_size, sw_batch_size, model, overlap=0.5, mode="gaussian")
                #val_outputs = monai.inferers.SlidingWindowInferer(roi_size, sw_batch_size, model,  mode="gaussian")(val_images)
                val_outputs = [post_trans(i) for i in monai.data.utils.decollate_batch(val_outputs)]
                dice_metric(y_pred=val_outputs, y=val_labels)
                hausdorff_distance(y_pred=val_outputs,y=val_labels)
            metric = dice_metric.aggregate().item()
            writer.add_scalar("dice_metric", metric, epoch + 1)
            Hmetric = hausdorff_distance.aggregate().item()
            writer.add_scalar("hausdorff_distance", Hmetric, epoch + 1)
            # reset the status for next validation round
            dice_metric.reset()
            hausdorff_distance.reset()
            #add to try to get the best model
            if metric > val_dice_best:
                val_dice_best = metric
                epoch_save = epoch+1
                best_loss = loss
                global_step_best = step
                val_hd_best = Hmetric
                torch.save(model, model_folder)
                print(f"epoch {epoch + 1}/{info.n_epoch}: Model SAVED! Current best avg. Dice {val_dice_best:.4f} Current Avg. HD {Hmetric:.4f}")
            else:
                if epoch % info.print_freq == 0:
                    print(f"epoch {epoch + 1}/{info.n_epoch}: Model NOT Saved! Current best avg. Dice {val_dice_best:.4f} Current avg. Dice {metric:.4f}")  
if info.save_final:
    model_final_f = folder / "model_final"
    torch.save(model, model_final_f)
print('training finished')
print(f"Best model")
print('-'*10)
print(f"epoch: {epoch_save}/{info.n_epoch}")
print(f"loss: {best_loss:.4f}")
print(f"validation Dice Score: {val_dice_best:.4f}")
print(f"validation hausdorff distance: {val_hd_best:.4f}")