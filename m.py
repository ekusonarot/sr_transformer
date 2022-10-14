from importlib.resources import path
from matplotlib.scale import scale_factory
from models.fsrcnn import FSRCNN
import sys
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
import cv2
from torch.nn import MSELoss
from utils import Converter
from torch.nn import functional as f

epoch = 1000
lr = 1e-2
early_stopping = 5

def ycbcr_to_bgr_tensor(img):
    r = (298.082 * img[:,0,:,:] / 255. + 408.583 * img[:,2,:,:] / 255. - 222.921 / 255. /255.).unsqueeze(1)
    g = (298.082 * img[:,0,:,:] / 255. - 100.291 * img[:,1,:,:] / 255. - 208.120 * img[:,2,:,:] / 255. + 135.576 / 255. / 255.).unsqueeze(1)
    b = (298.082 * img[:,0,:,:] / 255. + 516.412 * img[:,1,:,:] / 255. - 276.836 / 255. /255.).unsqueeze(1)
    return torch.cat((b, g, r),dim=1)

def bgr_to_ycbcr_tensor(img):
    y = (16. / 255. + (64.738 * img[:,2,:,:] + 129.057 * img[:,1,:,:] + 25.064 * img[:,0,:,:]) / 255.).unsqueeze(1)
    cb = (128. / 255. + (-37.945 * img[:,2,:,:] - 74.494 * img[:,1,:,:] + 112.439 * img[:,0,:,:]) / 255.).unsqueeze(1)
    cr = (128. / 255. + (112.439 * img[:,2,:,:] - 94.154 * img[:,1,:,:] - 18.285 * img[:,0,:,:]) / 255.).unsqueeze(1)
    return torch.cat([y, cb, cr], dim=1)

def bgr_to_y_tensor(img):
    return (16. / 255. + (64.738 * img[:,2,:,:] + 129.057 * img[:,1,:,:] + 25.064 * img[:,0,:,:]) / 255.).unsqueeze(1)

def unfold(img, kernel_size, stride):
    t = f.unfold(img, kernel_size=kernel_size, stride=stride)
    t = t.transpose(1,2)
    t = t.reshape([-1]+[img.shape[1]]+[kernel_size, kernel_size])
    return t

def fold(img, output_shape, kernel_size, stride):
    img = img.reshape([output_shape[0]]+[-1]+[kernel_size**2]).transpose(1,2)
    img = f.fold(img, output_shape[-2:], kernel_size=kernel_size, stride=stride)
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video_path", type=str, help="video path", required=True)
    parser.add_argument("-s", "--scale", type=float, help="save video scale", default=1/4)
    parser.add_argument("-f", "--format", type=str, help="save video format", default="mp4v")
    parser.add_argument("-o", "--output_path", type=str, help="save video path", default="output.mp4")
    parser.add_argument("-b", "--batch_size", type=int, help="batch size", default=48)
    parser.add_argument("-p", "--patch_size", type=int, help="output patch size", default=16)
    
    args = parser.parse_args()

    video_path = args.video_path
    scale = args.scale
    fmt = args.format
    output_path = args.output_path
    batch_size = args.batch_size
    patch_size = args.patch_size
    t_patch_size = int(patch_size/scale)

    cap = cv2.VideoCapture(video_path)
    if cap.isOpened() == False:
        print("file not found", file=sys.stderr)
        exit(1)
    
    fmt = cv2.VideoWriter_fourcc(fmt[0], fmt[1], fmt[2], fmt[3])
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*scale)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*scale)
    writer = cv2.VideoWriter(output_path, fmt, fps, (width, height))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_index = width//patch_size
    h_index = height//patch_size
    in_buf = [None] * batch_size
    gt_buf = [None] * batch_size
    frame_buf = np.zeros((height, width))
    count = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FSRCNN(4).to(device)
    state_dict = model.state_dict()
    for n, p in torch.load("./weights/fsrcnn_x4.pth", map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    upsample = torch.nn.Upsample(scale_factor=1/scale, mode='bicubic')
    sigmoid = torch.nn.Sigmoid()

    while True:
        start = time.perf_counter()
        for i in range(batch_size):
            ret, frame = cap.read()
            if ret == False:
                del in_buf[i:]
                del gt_buf[i:]
                break
            resized_img = cv2.resize(frame, dsize=None, fx=scale, fy=scale)
            in_buf[i] = resized_img/255.
            gt_buf[i] = frame/255.
        
        if len(in_buf) == 0:
            break
        input = np.array(in_buf).transpose([0,3,1,2])
        input = torch.tensor(input, requires_grad=True, device=device, dtype=torch.float)
        target = np.array(gt_buf).transpose([0,3,1,2])
        target = torch.tensor(target, device=device, dtype=torch.float)
        optimizer = optim.Adam([input], lr=lr)
        mseloss = MSELoss()
        minloss = 1e+10
        early_stop = 0
        print(time.perf_counter() -start)
        for e in range(epoch):
            optimizer.zero_grad()
            y = bgr_to_y_tensor(input)
            input_y = unfold(y, kernel_size=patch_size, stride=patch_size)
            sr_y = model(input_y)
            upscaled_input = upsample(input)
            upscaled_ycbcr = bgr_to_ycbcr_tensor(upscaled_input)
            sr_y = fold(sr_y, output_shape=target.shape, kernel_size=int(patch_size/scale), stride=int(patch_size/scale))
            output = torch.cat((sr_y, upscaled_ycbcr[:,1:,:,:]), dim=1)
            output = ycbcr_to_bgr_tensor(output)
            loss = mseloss(output, target)
            loss.backward()
            optimizer.step()
            bar = int(e/epoch*40)
            print("\r[train] epoch{}[{}] loss:{}".format(e, '='*bar+'-'*(40-bar), loss.item()), end="")
            if minloss <= float(loss.item()):
                early_stop += 1
                if early_stopping < early_stop:
                    print("[train] early stopping")
                    break
            else:
                minloss = float(loss.item())
                early_stop = 0
            del loss
        print("\n")
        input = input.mul(255.).cpu().detach().numpy().transpose([0,2,3,1])
        for i in input:
            writer.write(np.clip(i, 0., 255.).astype(np.uint8))
        progress = int(count/frame_count*40)
        print("[{}]{}/{}".format("="*progress+"-"*(40-progress),count,frame_count))
        count+=batch_size

    writer.release()
    cap.release()