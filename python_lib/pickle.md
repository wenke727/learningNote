# [pickle](https://docs.python.org/3/library/pickle.html)

使用pickle模块存储对象

``` python
import time
import hashlib
import pickle
import os

# ref: python：类的实例保存为pickle文件 https://zhuanlan.zhihu.com/p/165639894

class PickleSaver():
    def __init__(self, folder='../cache'):
        self.create_time=time.time()
        self.folder = folder

    def md5(self):
        m=hashlib.md5()
        m.update(str(self.create_time).encode('utf-8'))
        return m.hexdigest()

    def save(self, obj, fn):
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

        if '.pkl' not in fn:
            fn = f"{fn}.pkl"
            
        with open(os.path.join(self.folder, fn),'wb') as f:
            pickle.dump(obj, f)
        
        return True

    # @staticmethod
    def read(self, fn):
        if '/' not in fn:
            fn = os.path.join(self.folder, fn)
            
        with open(fn,'rb') as f:
            try:
                obj = pickle.load(f)
                return obj
            except Exception as e:
                pass
        
        return None
```
