#from .handler import handle

import importlib.util
import sys
spec = importlib.util.spec_from_file_location("handler", "/home/app/function/handler.py")
handlerModule = importlib.util.module_from_spec(spec)
sys.modules["handler"] = handlerModule
spec.loader.exec_module(handlerModule)

def test_handle():
    result = handlerModule.handle("test-data")
    print(result)

if __name__ == "__main__":
    test_handle()