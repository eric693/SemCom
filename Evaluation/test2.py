import ctypes

decoder = ctypes.CDLL('./inference_encoder.so')
i = do_test()
print(i)

