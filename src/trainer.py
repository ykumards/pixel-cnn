import os
import json
import time
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

def generate(trained_model, img_size, y, temp=0.8, cuda=False):
    trained_model.eval()
    gen = torch.from_numpy(np.zeros([y.shape[0], 1] + img_size, dtype='float32'))
    y = torch.from_numpy(y)
    
    if cuda:
        y, gen = y.cuda(), gen.cuda()   
             
    p_bar = ProgressBar()
    print('Generating images...')
    for r in p_bar(range(img_size[0])):
        for c in range(img_size[1]):
            out = trained_model(gen, y)
            p = torch.exp(out)[:, :, r, c]
            p = torch.pow(p, 1/temp)
            p = p/torch.sum(p, -1, keepdim=True)
            sample = p.multinomial(1)
            gen[:, :, r, c] = sample.float()/(out.shape[1] - 1)
    utils.clearline()
    utils.clearline()
    return (255*gen.data.cpu().numpy()).astype('uint8')

def generate_images(trained_model, 
                    img_size, 
                    n_classes, 
                    onehot_fcn, 
                    cuda=False):
    y = np.array(list(range(min(n_classes, 10))) * 5) # generate 5 images per class
    y = np.concatenate([onehot_fcn(x)[np.newaxis, :] for x in y])
    return generate(trained_model, img_size, y, cuda=cuda)

def generate_between_classes(model, 
                             img_size, 
                             classes, 
                             saveto,
                             n_classes, 
                             cuda=False):
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

def fit(train_loader,
        val_loader,
        model,
        exp_path,
        label_preprocess,
        loss_fcn,
        onehot_fcn,
        n_classes=10,
        optimizer='adam',
        learning_rate=1e-4,
        cuda=True,
        patience=10,
        max_epochs=200,
        resume=False):
    def _save_img(x, filename):
        Image.fromarray((255*x).astype('uint8')).save(filename)
    
    def _run_epoch(dataloader, training):
        p_bar = ProgressBar()
        losses = []
        mean_outs = []
        for x, y in p_bar(dataloader):
            label = label_preprocess(x)
            if cuda:
                x, y = x.cuda(), y.cuda()
                label = label.cuda()

            if training:
                optimizer.zero_grad()
                model.train()
            else:
                model.eval()
            output = model(x, y)
            loss = loss_fcn(output, label)
            # track mean output
            output = output.data.cpu().numpy()
            mean_outs.append(
                np.mean(np.argmax(output, axis=1))/output.shape[1])
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
    
    for e in range(start_epoch, max_epochs):
        # Training
        t0 = time.time()
        loss, mean_out = _run_epoch(train_loader, training=True)
        time_per_example = (time.time() - t0) / len(train_loader.dataset)
        stats['loss']['train'].append(loss)
        stats['mean_output']['train'].append(mean_out)
        print((f'Epoch {e}:    Training loss = {loss:.4f}    mean output = {mean_out:.2f}    '
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
            model, img_size, n_classes, onehot_fcn, cuda))
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