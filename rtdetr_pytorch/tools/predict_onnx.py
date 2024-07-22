import os 
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor
import argparse
import time

def main(args, ):
    im = Image.open(args.img).convert('RGB')
    im = im.resize((640, 640))
    im_data = ToTensor()(im)[None]
    # (width, height) = im.size
    print(im_data.shape)
    # print(width, height)
    # size = torch.tensor([[width, height]])
    size = torch.tensor([[640, 640]])
    sess = ort.InferenceSession(args.model)
    
    start_time = time.time()
    output = sess.run(
        # output_names=['labels', 'boxes', 'scores'],
        output_names=None,
        input_feed={'images': im_data.data.numpy(), "orig_target_sizes": size.data.numpy()}        
    )
    end_time = time.time()
    # inf_time = time.time() - start_time
    inf_time = end_time - start_time
    fps = float(1/inf_time)
    print("Inferece time = {} s".format(inf_time, '.4f'))
    print("FPS = {} ".format(fps, '.1f') )
    #print(type(output))
    #print([out.shape for out in output])

    labels, boxes, scores = output
    
    draw = ImageDraw.Draw(im)  # Draw on the original image
    thrh = 0.6

    for i in range(im_data.shape[0]):

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]

        #print(i, sum(scr > thrh))

        for b in box:
            draw.rectangle(list(b), outline='red',)
            # font = ImageFont.truetype("Arial.ttf", 15)
            draw.text((b[0], b[1]), text=str(lab[i]), fill='yellow', )

    # Save the original image with bounding boxes
    file_dir = os.path.dirname(args.img)
    new_file_name = os.path.basename(args.img).split('.')[0] + '_onnx'+ os.path.splitext(args.img)[1]
    new_file_path = file_dir + '/' + new_file_name
    print('new_file_path: ', new_file_path)
    im.save(new_file_path)
 

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', '-i', type=str, )
    parser.add_argument('--model', '-m', type=str, default='model.onnx')

    args = parser.parse_args()

    main(args)