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

epoch = 500
lr = 1e-2
early_stopping = 10

def ycbcr_to_rgb_tensor(img):
    r = (298.082 * img[:,0,:,:] / 255. + 408.583 * img[:,2,:,:] / 255. - 222.921 / 255.).unsqueeze(1)
    g = (298.082 * img[:,0,:,:] / 255. - 100.291 * img[:,1,:,:] / 255. - 208.120 * img[:,2,:,:] / 255. + 135.576 / 255.).unsqueeze(1)
    b = (298.082 * img[:,0,:,:] / 255. + 516.412 * img[:,1,:,:] / 255. - 276.836 / 255.).unsqueeze(1)
    return torch.cat((b, g, r),dim=1)


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
    out_buf = [None] * batch_size
    mid_buf = [None] * batch_size
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

    while True:
        start = time.perf_counter()
        for i in range(batch_size):
            ret, frame = cap.read()
            if ret == False:
                del in_buf[i:]
                del mid_buf[i:]
                del gt_buf[i:]
                del out_buf[i:]
                break
            resized_img = cv2.resize(frame, dsize=None, fx=scale, fy=scale)
            upscaled_img = cv2.resize(resized_img, dsize=None, fx=1/scale, fy=1/scale, interpolation = cv2.INTER_CUBIC)
            in_buf[i] = Converter.convert_bgr_to_y(resized_img)
            mid_buf[i] = Converter.convert_bgr_to_ycbcr(upscaled_img)/255.
            gt_buf[i] = frame/255.
            out_buf[i] = Converter.convert_bgr_to_ycbcr(resized_img)
        
        if len(in_buf) == 0:
            break
        input = np.array([in_buf[index//(h_index*w_index)]
                    [index%(h_index*w_index)//w_index*patch_size : index%(h_index*w_index)//w_index*patch_size+patch_size,
                    index%(h_index*w_index)%w_index*patch_size : index%(h_index*w_index)%w_index*patch_size+patch_size]
                    for index in range(batch_size*w_index*h_index)])
        input = np.expand_dims(input, 1)
        input = torch.tensor(input, requires_grad=True, device=device, dtype=torch.float)

        mid = np.array([mid_buf[index//(h_index*w_index)]
                    [index%(h_index*w_index)//w_index*t_patch_size : index%(h_index*w_index)//w_index*t_patch_size+t_patch_size,
                    index%(h_index*w_index)%w_index*t_patch_size : index%(h_index*w_index)%w_index*t_patch_size+t_patch_size,:]
                    for index in range(batch_size*w_index*h_index)]).transpose([0,3,1,2])
        mid = torch.tensor(mid, device=device, dtype=torch.float)
        
        target = np.array([gt_buf[index//(h_index*w_index)]
                    [index%(h_index*w_index)//w_index*t_patch_size : index%(h_index*w_index)//w_index*t_patch_size+t_patch_size,
                    index%(h_index*w_index)%w_index*t_patch_size : index%(h_index*w_index)%w_index*t_patch_size+t_patch_size,:]
                    for index in range(batch_size*w_index*h_index)]).transpose([0,3,1,2])
        target = torch.tensor(target, device=device, dtype=torch.float)
        optimizer = optim.Adam([input], lr=lr)
        mseloss = MSELoss()
        minloss = 1e+10
        early_stop = 0
        print(time.perf_counter() -start)
        for e in range(epoch):
            optimizer.zero_grad()
            output = model(input)
            output = torch.cat((output, mid[:,1:,:,:]), dim=1)
            output = ycbcr_to_rgb_tensor(output)
            loss = mseloss(output, target)
            loss.backward()
            optimizer.step()
            bar = int(e/epoch*40)
            print("\r[train] epoch{}[{}] loss:{}".format(e, '='*bar+'-'*(40-bar), loss.item()), end="")
            if minloss < float(loss.item()):
                early_stop += 1
                if early_stopping < early_stop:
                    print("[train] early stopping")
                    break
            else:
                minloss = float(loss.item())
                early_stop = 0
            del loss
        print("\n")
        input = input.mul(255.).cpu().detach().numpy()
        for i in range(0, len(input), w_index*h_index):
            frame = input[i:i+w_index*h_index,:,:,:]
            for j in range(w_index*h_index):
                x = j//w_index*patch_size
                y = j%w_index*patch_size
                frame_buf[x:x+patch_size,y:y+patch_size] = frame[x//patch_size*w_index+y//patch_size,0,:,:]
            k = i//(w_index*h_index)
            f = np.array([frame_buf, out_buf[k][..., 1], out_buf[k][..., 2]]).transpose([1,2,0])
            f = Converter.convert_ycbcr_to_bgr(f)
            writer.write(np.clip(f, 0., 255.).astype(np.uint8))
        progress = int(count/frame_count*40)
        print("[{}]{}/{}".format("="*progress+"-"*(40-progress),count,frame_count))
        count+=batch_size

    writer.release()
    cap.release()