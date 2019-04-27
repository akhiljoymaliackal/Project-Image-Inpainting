import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import requests
from tornado.options import define, options
import urllib
import json
import tornado.httpclient
import tensorflow as tf
import numpy as np
import cv2
from secondtry import *
import os
import threading
import base64
img_path="input_image.png"
mask_path="Mask_Image.png"
completion=0
stage=0
image_received=0
#sess = tf.Session()
class initial(tornado.web.RequestHandler):
    def get(self):
        global completion,stage,image_received
        completion=0
        stage=0
        image_received=0
class checkstatus(tornado.web.RequestHandler):
    def get(self):
        global stage,image_received
        if stage==0:
            self.write("1")
            print "1"
            stage=1
        if stage==1 and image_received==0:
            self.write("2")
            print "2"
        if stage==1 and image_received==1:
            self.write("3")
            print "processing the data"
            stage=2
        if stage==2 and completion==0:
            self.write("4")
            print "inpainting..."
        if stage==2 and completion==1:
            self.write("5")
            print "data is ready to deliver"

class sendimage(tornado.web.RequestHandler):
    def get(self):

        img=cv2.imread("result.png")
        img=cv2.resize(img,(64,64))
        with open("result.png", "rb") as imageFile:
            str = base64.b64encode(img)
            #print str
            print len(str)
            print "hai"
            a=urllib.quote(str.encode('utf-8'))
            self.write(str)
class download(tornado.web.RequestHandler):
    def get(self):

        img=cv2.imread("result.png")
        img=cv2.resize(img,(64,64))
        with open("result.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            self.write(encoded_string)


class dataprocess(tornado.web.RequestHandler):


    def get(self):

        global stage,image_received
        data=self.request.uri
        image_string=data.split("/")
        print image_string[2]
        h=image_string[2].split("AKHILJOY")
        mask_encoded=urllib.unquote(h[0]).decode('utf8')
        fh = open("input_image.png", "wb")
        fh.write(mask_encoded.decode('base64'))
        fh.close()
        Image_encoded=urllib.unquote(h[1]).decode('utf8')
        fh = open("Mask_Image.png", "wb")
        fh.write(Image_encoded.decode('base64'))
        fh.close()
        image_received=1
        t1 = threading.Thread(target=imple,args=(self,))
        t1.start()

def imple(self):
    global completion
    completion=0
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config)as sess:
        model = network()
        print "Started"
        test(self,sess, model)
        sess.close()
        completion=1

def process_img(img):
    mask=cv2.imread(mask_path)
    test_img = cv2.resize(img, (input_height, input_width))/127.5 - 1
    test_mask = cv2.resize(mask, (input_height,input_width))/255.0
    test_img = (test_img * (1-test_mask)) + test_mask
    return np.tile(test_img[np.newaxis,...], [batch_size,1,1,1]), np.tile(test_mask[np.newaxis,...], [batch_size,1,1,1])
def test(self,sess,model):
    global completion
    completion=0
    saver = tf.train.Saver()
    last_ckpt = tf.train.latest_checkpoint(checkpoints_path)
    saver.restore(sess, last_ckpt)
    print "The model has been loaded "
    img=cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    orig_test = cv2.resize(img, (input_height,input_width))/127.5 - 1
    orig_test = np.tile(orig_test[np.newaxis,...],[batch_size,1,1,1])
    orig_test = orig_test.astype(np.float32)
    orig_w, orig_h = img.shape[0], img.shape[1]
    test_img, mask=process_img(img)
    test_img = test_img.astype(np.float32)
    print "Testing ..."
    res_img = sess.run(model.test_res_imgs, feed_dict={model.single_orig:orig_test,
                                                       model.single_test:test_img,
                                                       model.single_mask:mask})
    orig = cv2.resize((orig_test[0]+1)/2, (orig_h/2, orig_w/2))
    test = cv2.resize((test_img[0]+1)/2, (orig_h/2, orig_w/2))
    recon = cv2.resize((res_img[0]+1)/2, (orig_h*2, orig_w*2))
    wi,hi,_=recon.shape
    res1=recon
    print wi,hi
    for i in range(0,wi):
        for j in range(0,hi):
            for k in range(0,3):
                res1[i][j][k]=recon[i][j][k]*255
    pdp= cv2.cvtColor(res1, cv2.COLOR_BGR2RGB)
    cv2.imwrite("result.png",pdp)
    completion=1


def main():
    tornado.options.parse_command_line()
    application = tornado.web.Application([
        (r"/PING",checkstatus),(r"/DATA.*",dataprocess),(r"/PINGFIRST",initial),(r"/DOWNLOAD",download)])
    http_server = tornado.httpserver.HTTPServer(application,max_body_size=1024*1024*150*12)
    http_server.listen(8000, address='192.168.1.103')

    #http_server.listen(options.port)
    tornado.ioloop.IOLoop.current().start()
if __name__ == "__main__":
	main()
