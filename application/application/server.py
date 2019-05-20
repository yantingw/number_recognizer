
import os
from sklearn.externals import joblib
import tornado.web
from machines.machine_loader import MachineLoader
import machines.number_recognizer
from machines.number_recognizer.validator import Validator
import numpy as np
from pandas import DataFrame
from keras import backend as K
DATA_DIR = os.path.join(os.path.dirname(machines.number_recognizer.__file__),"onlinedata")
clf_dir = os.path.dirname(machines.number_recognizer.__file__)
class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html", title="title")


class BaseHandler(tornado.web.RequestHandler):
    MACHINE_SESSION_KEY = "number_recognizer"


class PredictionHandler(BaseHandler):
    def get(self):
        num = self.get_argument('num', True)
        PATH = os.path.join(DATA_DIR,"y_test.npy")
        try:
            y_test = np.load(PATH)
            y_test =np.append(y_test,num)
            print("find the record")
        except:
            y_test = np.array([num])
 
        np.save(PATH,y_test)         
        
    def post(self):
        resp = {"result": str(-1)}
        data = self.get_arguments("data[]")
        print('get post/n')
        validated = Validator.validate_data(data)

        machine = MachineLoader.load(machines.number_recognizer)
        if len(validated) > 0:
            validated = np.array(validated).reshape((1,28,28,1))
            print(validated.shape)

      #%%
      #find the name of file
            try:
                file = open(os.path.join(DATA_DIR,'count.txt'),'r')
                count =int(file.readline())
                count+=1
                file = open(os.path.join(DATA_DIR,'count.txt'),'w')
                file.write(str(count))
                file.close()
            except:
                file = open(os.path.join(DATA_DIR,'count.txt'),'w')
                file.write("0")
                count = 0
                
#%%            # 轉換色彩 0~255 資料為 0~1
            validated = validated.astype('float64')
            #validated[validated > 0] = 1 # 二質化
          # validated /= np.max(validated)
          #  path=os.path.join(os.path.dirname(machines.number_recognizer.__file__),"onlinedata",'img2')
            filename = 'img'+str(count)
            path=os.path.join(DATA_DIR,filename)
            np.save(path, validated)
            #print(validated)
            print ("download complete AT "+os.path.join(path,))
         
     #%%       
            #predicted = machine.predict(validated)
            #predict_number = np.argmax(predicted)
     
            
            pre =machine.predict(validated)
            clf = joblib.load(os.path.join(DATA_DIR,'clf_linear.pkl'))
            predict_number =clf.predict(pre)[0]

      
            resp["result"] = str(predict_number)
        self.write(resp)


class FeedbackHandler(BaseHandler):

    def post(self):
        data = self.get_arguments("data[]")
        result = ""

        feedback = Validator.validate_feedback(data)
        if len(feedback) > 0:
            # save feedback to file
            MachineLoader.feedback(machines.number_recognizer, feedback)

            # online training
            machine = MachineLoader.load(machines.number_recognizer)
            machine.partial_fit(feedback[1:], [feedback[0]])
            MachineLoader.save(machines.number_recognizer, machine)
        else:
            result = "feedback format is wrong."

        resp = {"result": result}
        self.write(resp)


class Application(tornado.web.Application):

    def __init__(self):
        handlers = [
            (r"/", IndexHandler),
            (r"/predict", PredictionHandler),
            (r"/feedback", FeedbackHandler),
        ]

        settings = dict(
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            cookie_secret=os.environ.get("SECRET_TOKEN", "__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__"),
            xsrf_cookies=True,
            debug=True,
        )

        super(Application, self).__init__(handlers, **settings)
