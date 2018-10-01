import cv2
path_to_new_labels = ''    
if path_to_new_labels==False:
    print('Path to new labels missing')
    
alls = os.listdir(path_to_labels)
for al in alls:
    GT = cv2.imread(path_to_labels+al,0)
    GT[GT!=255]=0
    cl = cv2.imwrite(path_to_new_labels+al,GT)
    print('Label written')
