import os, json, time
from typing import Callable
from pathlib import Path
from skimage.transform import resize
from progressbar import ProgressBar
import numpy as np
import imageio
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

import utils, args

def generate(trained_model: torch.nn.Module, 
             img_size: list, 
             y: np.array, 
             temp=0.8, 
             cuda=False) -> np.array:
    trained_model.eval()
    generated_image = np.zeros([y.shape[0], 1] + img_size, dtype='float32')
    # tip: when from_numpy() is used tensor and np_array share same memory
    # changes are reflected on each other. so used when converting back to
    # numpy array 
    generated_image, y = torch.from_numpy(generated_image), torch.from_numpy(y)
    if cuda:
        y, generated_image = y.cuda(), generated_image.cuda()   
             
    p_bar = ProgressBar()
    print('Generating images...')
    for r in p_bar(range(img_size[0])):
        for c in range(img_size[1]):
            out = trained_model(generated_image, y)
            p = torch.exp(out)[:, :, r, c]
            p = torch.pow(p, 1/temp)
            p = p/torch.sum(p, -1, keepdim=True)
            sample = p.multinomial(1)
            generated_image[:, :, r, c] = sample.float()/(out.shape[1] - 1)
            
    utils.clearline()
    utils.clearline()
    return (255 * generated_image.data.cpu().numpy()).astype('uint8')

def generate_images(trained_model: torch.nn.Module, 
                    img_size: list, 
                    n_classes: int, 
                    label2onehot: Callable, 
                    cuda: bool=False) -> np.array:
    "generate images from the trained model. Hard coded to 5 samples per class"
    # if n_classes = 3; y = [0,1,2] * 5
    y = list(range(min(n_classes, 10))) * 5 # generate 5 images per class    
    y = np.concatenate([label2onehot(np.array(x))[np.newaxis, :] for x in y])
    return generate(trained_model, img_size, y, cuda=cuda)

def generate_between_classes(model: torch.nn.Module, 
                             img_size: list, 
                             classes: int, 
                             saveto: str,
                             n_classes: int, 
                             cuda:bool =False) -> None:
    y = np.zeros((1, n_classes), dtype='float32')
    y[:, classes] = 1/len(classes)
    y = np.repeat(y, 10, axis=0)
    gen = utils.tile_images(generate(model, img_size, y, cuda=cuda), n_rows=1)
    imageio.imsave(saveto, gen.astype('uint8'))
    
def plot_loss(train_loss: list, val_loss: list):
    fig = plt.figure(
        num=1, figsize=(4, 4), dpi=70, facecolor='w', edgecolor='k')
    plt.plot(range(1, len(train_loss) + 1), train_loss, 'r', label='training')
    plt.plot(range(1, len(val_loss) + 1), val_loss, 'b', label='validation')
    plt.title(f'After {len(train_loss)} epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-entropy loss')
    plt.rcParams.update({'font.size': 10})
    fig.tight_layout(pad=1)
    fig.canvas.draw()

    # now convert the plot to a numpy array
    plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return plot

def fit(train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        exp_path: str,
        label_preprocess: Callable,
        loss_fn: torch.autograd.Function,
        label2onehot: Callable,
        n_classes:int =10,
        optimizer:str ='adam',
        learning_rate:float =1e-4,
        cuda:bool =True,
        patience:int =10,
        max_epochs:int =200,
        resume:bool =False):
    def _save_img(x: np.array, filename: str):
        "save numpy array representaion of image to image file"
        Image.fromarray((255*x).astype('uint8')).save(filename)
    
    def _run_epoch(dataloader: torch.utils.data.DataLoader, 
                   training: bool):
        p_bar = ProgressBar()
        losses = []
        mean_outs = []
        # import pdb; pdb.set_trace()
        # batch_images = [bs, ch, w, h], batch_targets = [bs, n_classes] OHE
        for batch_images, batch_targets in p_bar(dataloader):
            batch_labels = label_preprocess(batch_images)
            if cuda:
                batch_images, batch_targets = batch_images.cuda(), batch_targets.cuda()
                batch_labels = batch_labels.cuda()

            if training:
                optimizer.zero_grad()
                model.train()
            else:
                model.eval()
            predicted_labels = model(batch_images, batch_targets)
            # cross entropy between image's pixel value vs predicted
            loss = loss_fn(predicted_labels, batch_labels)
            # track mean output
            predicted_labels = predicted_labels.data.cpu().numpy()
            # TODO: why do we need to keep track of mean?
            mean_outs.append(
                np.mean(np.argmax(predicted_labels, axis=1))/predicted_labels.shape[1])
            if training:
                loss.backward()
                optimizer.step()
            losses.append(loss.data.cpu().numpy())
        utils.clearline()
        return float(np.mean(losses)), np.mean(mean_outs)
    
    if cuda:
        model = model.cuda()
   
    exp_path = Path(exp_path)    
     
    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)
    statsfile = exp_path / 'stats.json'
    
    optimizer = {'adam': torch.optim.Adam(model.parameters(), lr=learning_rate),
                 'sgd': torch.optim.SGD(
                     model.parameters(), lr=learning_rate, momentum=0.9),
                 'adamax': torch.optim.Adamax(model.parameters(), lr=learning_rate)
                 }[optimizer.lower()]
    
    # load a single example from the iterator to get the image size
    x = train_loader.sampler.data_source.__getitem__(0)[0]
    img_size = list(x.numpy().shape[1:])
    
    if not resume:
        stats = {'loss': {'train': [], 'val': []},
                 'mean_output': {'train': [], 'val': []}}
        best_val = np.inf
        stall = 0
        start_epoch = 0
        generated = []
        plots = []
        
    else:
        with open(statsfile, 'r') as js:
            stats = json.load(js)
        best_val = np.min(stats['loss']['val'])
        stall = len(stats['loss']['val'])-np.argmin(stats['loss']['val'])-1
        start_epoch = len(stats['loss']['val'])-1
        generated = list(np.load(exp_path/'generated.npy'))
        plots = list(np.load(exp_path/'generated_plots.npy'))
        print(f'Resuming from epoch {start_epoch}')
    
    for epoch in range(start_epoch, max_epochs):
        # Training
        t0 = time.time()
        loss, mean_out = _run_epoch(train_loader, training=True)
        time_per_example = (time.time() - t0) / len(train_loader.dataset)
        stats['loss']['train'].append(loss)
        stats['mean_output']['train'].append(mean_out)
        print((f'Epoch {epoch}:    Training loss = {loss:.4f}    mean output = {mean_out:.2f}    '
               f'{time_per_example*1000:.2f} msec/example'))
        
        # Validation
        t0 = time.time()
        loss, mean_out = _run_epoch(val_loader, training=False)
        time_per_example = (time.time()-t0)/len(val_loader.dataset)
        stats['loss']['val'].append(loss)
        stats['mean_output']['val'].append(mean_out)
        print((f'            Validation loss = {loss:.4f}    mean output = {mean_out:.2f}    '
               f'{time_per_example*1000:.2f} msec/example'))
        
        # Generate images and update gif
        new_frame = utils.tile_images(generate_images(
            model, img_size, n_classes, label2onehot, cuda))
        generated.append(new_frame)
        
        # Update gif with loss plot
        plot_frame = plot_loss(stats['loss']['train'], stats['loss']['val'])
        if new_frame.ndim == 2:
            new_frame = np.repeat(new_frame[:, :, np.newaxis], 3, axis=2)
        nw = int(new_frame.shape[1]*plot_frame.shape[0]/new_frame.shape[0])
        new_frame = resize(new_frame, [plot_frame.shape[0], nw],
                           order=0, preserve_range=True, mode='constant')
        plots.append(np.concatenate((plot_frame.astype('uint8'),
                                     new_frame.astype('uint8')),
                                    axis=1))
        
        # Save gif arrays so it can resume training if interrupted
        np.save(exp_path/'generated.npy', generated)
        np.save(exp_path/'generated_plots.npy', plots)
        
        # Save stats and update training curves
        with open(statsfile, 'w') as sf:
            json.dump(stats, sf)
        utils.plot_stats(stats, exp_path)
        
        # Early stopping
        torch.save(model, exp_path/'last_checkpoint.pth')
        
        if loss < best_val:
            best_val = loss
            stall = 0
            torch.save(model, exp_path/'best_checkpoint.pth')
            imageio.imsave(exp_path/'best_generated.jpeg',
                           generated[-1].astype('uint8'))
            imageio.imsave(exp_path/'best_generated_plots.jpeg',
                           plots[-1].astype('uint8'))
            imageio.mimsave(exp_path/'generated.gif',
                            np.array(generated), format='gif', loop=0, fps=2)
            imageio.mimsave(exp_path/'generated_plot.gif',
                            np.array(plots), format='gif', loop=0, fps=2)
        else:
            stall += 1
        if stall >= patience:
            break