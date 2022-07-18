import io

import cv2
import numpy as np


class Memcached:
    import sys
    sys.path.append('/mnt/lustre/share/pymc/py3')
    try:
        _mc = __import__('mc')
    except:
        _mc = None
    _instance = None

    @classmethod
    def _init_memcached(cls):
        if cls._instance is None:
            server_list_config = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config = "/mnt/lustre/share/memcached_client/client.conf"
            cls._instance = cls._mc.MemcachedClient.GetInstance(server_list_config, client_config)

    @classmethod
    def cv2_imread(cls, file_path):
        if cls._mc is None:
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        else:
            cls._init_memcached()
            value = cls._mc.pyvector()
            cls._instance.Get(file_path, value)
            value_buf = cls._mc.ConvertBuffer(value)
            img_array = np.frombuffer(value_buf, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        return img

    @classmethod
    def np_load(cls, file_path):
        if cls._mc is None:
            res = np.load(file_path)
        else:
            cls._init_memcached()
            value = cls._mc.pyvector()
            cls._instance.Get(file_path, value)
            value_buf = cls._mc.ConvertBuffer(value)
            value_buf = io.BytesIO(value_buf)
            res = np.load(value_buf)
        return res
