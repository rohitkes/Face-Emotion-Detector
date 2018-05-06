
from __future__ import print_function
import argparse,sys

try:
    from FeatureGen import*
except ImportError:
    exit()

try:
    import dlib
    from skimage import io
    import numpy
    import cv2
    from sklearn.externals import joblib
except ImportError:
        exit()

emotions={ 1:"Anger", 2:"Contempt", 3:"Disgust", 4:"Fear", 5:"Happy", 6:"Sadness", 7:"Surprise"}

def Predict_Emotion(filename):

    print("Opening image....")
    try:
        img=io.imread(filename)
        cvimg=cv2.imread(filename)
    except:
        print("Exception: File Not found.")
        return

    win.clear_overlay()
    win.set_image(img)

    dets=detector(img)
	

    if len(dets)==0:
        print("Unable to find any face")
        return

    for k,d in enumerate(dets):

        shape=predictor(img,d)
        landmarks=[]
        for i in range(68):
            landmarks.append(shape.part(i).x)
            landmarks.append(shape.part(i).y)
        
    
        landmarks=numpy.array(landmarks)
    
        print("Generating features")
        features=generateFeatures(landmarks)
        features= numpy.asarray(features)

        print("Performing PCA Transform")
        pca_features=pca.transform(features)

		
		
		start_training(pca_features);
		
        
        emo_predicts=classify.predict(pca_features,image)
        print("Predicted emotion using trained data is { " + emotions[int(emo_predicts[0])] + " }")
      

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cvimg,emotions[int(emo_predicts[0])],(20,20), font, 1,(0,255,255),2)

        win.add_overlay(shape)

    cv2.namedWindow("Output")
    cv2.imshow("Output",cvimg)
    cv2.waitKey(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, nargs='+', help="Enter the filenames with extention of an Image")
    arg=parser.parse_args()
    
    if not len(sys.argv) > 1:
        parser.print_help()
        exit()

    landmark_path="shape_predictor_68_face_landmarks.dat"

    
    detector= dlib.get_frontal_face_detector()

    print("Loading landmark identification data")
    try:
        predictor= dlib.shape_predictor(landmark_path)
    except:
        print("Unable to find trained facial shape predictor")
        exit()

    win=dlib.image_window()

    print("Loading trained data")

    try:
        classify=joblib.load("traindata.pkl")
        pca=joblib.load("pcadata.pkl")
    except:
        print("Unable to load trained data")
        exit()

    for filename in arg.i:
        Predict_Emotion(filename)

		
		
		
def build_network(self):
    
    print('[+] Building CNN')
    self.network = input_data(shape = [None, SIZE_FACE, SIZE_FACE, 1])
    self.network = conv_2d(self.network, 64, 5, activation = 'relu')
    #self.network = local_response_normalization(self.network)
    self.network = max_pool_2d(self.network, 3, strides = 2)
    self.network = conv_2d(self.network, 64, 5, activation = 'relu')
    self.network = max_pool_2d(self.network, 3, strides = 2)
    self.network = conv_2d(self.network, 128, 4, activation = 'relu')
    self.network = dropout(self.network, 0.3)
    self.network = fully_connected(self.network, 3072, activation = 'relu')
    self.network = fully_connected(self.network, len(EMOTIONS), activation = 'softmax')
    self.network = regression(self.network,
      optimizer = 'momentum',
      loss = 'categorical_crossentropy')
    self.model = tflearn.DNN(
      self.network,
      checkpoint_path = SAVE_DIRECTORY + '/emotion_recognition',
      max_checkpoints = 1,
      tensorboard_verbose = 2
    )
    self.load_model()
		

		

def start_training(self):
    self.load_saved_dataset()
    self.build_network()
    if self.dataset is None:
      self.load_saved_dataset()
    # Training
    print('[+] Training network')
    self.model.fit(
      self.dataset.images, self.dataset.labels,
      validation_set = (self.dataset.images_test, self.dataset._labels_test),
      n_epoch = 100,
      batch_size = 50,
      shuffle = True,
      show_metric = True,
      snapshot_step = 200,
      snapshot_epoch = True,
      run_id = 'emotion_recognition'
    )
		
		
		
def predict(self, image):
    if image is None:
      return None
    image = image.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
    return self.model.predict(image)

