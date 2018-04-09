import argparse
import sys
import time
import os
import cv2
"""
DEL_X=2     #change in X and Y
DEL_Y=1
SIDE_X=30    #initial side lengths
SIDE_Y=40

def crop_img(path,pos):    #pos is the index of the sliding window
    img=cv2.imread(path)
    [x_max,y_max]=img.size
    if((pos*DELTA_X/x_max)*DELTA_Y<y_max):
        os.remove('result.csv')
        cropped_img=img[int((pos*DELTA_X/x_max)*DELTA_Y):int((pos*DELTA_X/x_max)*DELTA_Y)+SIDE_Y , int(pos*DELTA_X%x_max):int(pos*DELTA_X%x_max)+SIDE_X]
        cv2.imwrite(path2)
        os.system("python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --image="+path2)
        file=open('Database.csv')
        reader=csv.reader(file)
        data=list(reader)
        for x in data:
            if x[1]>0.95 and x[0]!='negative':
                print(x[0])
                print("at window no. " + str(pos))
                break
    else:
        exit(0)
    
    


i=0
while True:
    crop_img("",i)
    i=i+1
    """

import numpy as np
import tensorflow as tf

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def make_pred(image):
  file_name = image
  model_file = "tf_files/retrained_graph.pb"
  label_file = "tf_files/retrained_labels.txt"
  input_height = 224
  input_width = 224
  input_mean = 128
  input_std = 128
  input_layer = "input"
  output_layer = "final_result"
  graph = load_graph(model_file)
  t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);

  with tf.Session(graph=graph) as sess:
    start = time.time()
    results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})   #my code
    end=time.time()
  i=np.where(results==max(results[0]))[1][0]
  
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1]
  
  labels = load_labels(label_file)
  print(labels[i])
  print(max(results))
  return [labels,results]    #mine

  
def sliding_window(file):
    points=[]
    xmin=0
    ymin=0
    xmax=160
    ymax=160
    image=cv2.imread(file)
    step=30
    path="tf_files/folder/buff"
    while xmax<1000:
        xmin=xmin+step                            #correct this
        xmax=xmax+step
        ymin=0
        ymax=160
        while ymax<600:
            ymin=ymin+step
            ymax=ymax+step
            cropped_image = image[ymin:ymax,xmin:xmax]
            path_max=path+str(xmin)+","+str(ymin)+".jpg"
            cv2.imwrite(path_max,cropped_image)
            #cv2.imshow("",cropped_image)
            #time.sleep(2)
            #cv2.destroyAllWindows()
            result = make_pred(path_max)
            print("on window",xmin,ymin)
            '''for x in range(len(result[0])):
                if(result[0][x]>=0.99):
                    points.append({((xmax-xmin)/2),((ymax-ymin)/2)})
                    break
			'''
    return points               #returns a 1d array of centroids of all clusters
            
            
sliding_window("C:/Users/Atharv/Desktop/ITSP/Photos/Digene.jpg")

            
