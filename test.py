# while True:
#     _, image = cap.read()

#     h, w = image.shape[:2]
#     blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
#     net.setInput(blob)

#     start = time.perf_counter()
#     layer_outputs = net.forward(ln)
#     time_took = time.perf_counter() - start
#     print("Time took:", time_took)
#     boxes, confidences, class_ids = [], [], []

# while True:
        
#         ret, img = cap.read()
#         sized = cv2.resize(img, (m.width, m.height))
#         sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

#         start = time.time()
#         boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
#         finish = time.time()
#         print('Predicted in %f seconds.' % (finish - start))

#         result_img = plot_boxes_cv2(img, boxes[0], savename='resul.avi', class_names=class_names)

#         #cv2.imshow('Yolo demo', result_img)
#         cv2.waitKey(1)
