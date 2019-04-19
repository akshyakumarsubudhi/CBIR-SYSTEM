import cv2 as cv # To read and write an image
import numpy as np # To perform mathematical operatinon on data
import os # To get system path.
import xlsxwriter # To create and write on excel file
import pandas as pd # To perform operation on excel file
import random as rm # To generate a random number 
import matplotlib.pyplot as plt # For plotting purposes
import pickle # Keep Traack of object it has already serialized

##Function 1 :- To Load images from Folder
def load_images_from_folder(folder):
    print("reading from database...")
    subfolder = os.listdir(folder) # listdir function will return all subfolder available under a folder
    images = [] # To hold all available images in a folder.
    for val in subfolder:
        path = os.path.join(folder, val) # path to the subfolder images
        for image in os.listdir(path):
            if image.endswith(".jpg"):
                img = cv.imread(os.path.join(path, image),1)
                if img is not None:
                    images.append(img)
    print(len(images)," images have been read...")
    return images
##Function 2
def lbp_feature(list_of_images,filename):
    print("LBP feature Extraction")
    workbook = xlsxwriter.Workbook(filename) # Creating Workbook object.
    sheet = workbook.add_worksheet('LBP') # Creating a sheet.
    
    if sheet == '\0':
        print('error in file creation')
    else:
        print('file created')
        w = [[128,64,32],[1,0,16],[2,4,8]] # Kernal for Binary to Decimal Conversion
        for ite in range(len(list_of_images)):
            if ite%100 == 0:
                print(ite)
            gray = cv.cvtColor(list_of_images[ite],cv.COLOR_BGR2GRAY) # Converting Color from BGR to GRAY.
            array = np.asarray(gray) # Converting image into an numpy array for easy calculation.
            row,col,_ = list_of_images[ite].shape
            binary_decimal_list = []
            for j in range(1,row-1):
                for k in range(1,col-1):
                    x = array[j-1:j+2,k-1:k+2]
                    wl = 0
                    for m in range(3):
                        for n in range(3):
                            if x[m,n] > array[j,k]:
                                wl+=w[m][n] # Binary To Decimal Conversion using Kernal.
                    binary_decimal_list.append(str(wl))
            #Histogram calculation
            j=0
            for intensity in range(0,256,2):
                count = 0
                for val in range(len(binary_decimal_list)):
                    if int(binary_decimal_list[val])==intensity:
                        count+=1
                sheet.write(ite,j,count) # writing into a sheet. 
                j+=1                 
    workbook.close() # Closing workbook
    print("finished")
##Function 2.1
# To Use during Image Search
def lbp_of_test_image(test_image):
    w = [[128,64,32],[1,0,16],[2,4,8]]
    array = np.asarray(test_image)
    row,col = array.shape
    Feature = []
    binary_decimal_list = []
    for j in range(1,row-1):
        for k in range(1,col-1):
            x = array[j-1:j+2,k-1:k+2]
            wl = 0
            for m in range(3):
                for n in range(3):
                    if x[m,n] > array[j,k]:
                       wl+=w[m][n]
            binary_decimal_list.append(str(wl))
            
    for intensity in range(0,256,2):
        count = 0
        for val in range(len(binary_decimal_list)):
            if int(binary_decimal_list[val])==intensity:
                count+=1
        Feature.append(count)
    return Feature
##Function 3
def zig_zag(array):
    r,c = array.shape
    i = j = 0
    array2 = []
    array2.append(array[i,j])
    while(i != r-1 or j != c-1):
        j+=1
        array2.append(array[i,j])
        while(j!=0):
            i+=1
            j-=1
            array2.append(array[i,j])
        if(i != r-1 or j != 0):
            i+=1
            array2.append(array[i,j])
            while(i!=0):
                j+=1
                i-=1
                array2.append(array[i,j])
        else:
            j+=1
            array2.append(array[i,j])
            while(i != r-1 or j != c-1):
                while(j != c-1):
                    i-=1
                    j+=1
                    array2.append(array[i,j])
                i+=1
                array2.append(array[i,j])
                while(i!=r-1):
                    j-=1
                    i+=1
                    array2.append(array[i,j])
                j+=1
                array2.append(array[i,j])
    return array2
##Function 4
def dct(list_of_images,filename):
    print("DCT feature Extraction")
    workbook = xlsxwriter.Workbook(filename) # Creating workbook object
    sheet = workbook.add_worksheet('DCT')

    if sheet == '\0':
        print('error in file creation')
    else:
        print('file created')
        for ite in range(len(list_of_images)):
            gray = cv.cvtColor(list_of_images[ite],cv.COLOR_BGR2GRAY)
            array =cv.resize(gray,(64,64))
            imf = np.float32(array)  # float conversion/scale
            discrete_coeff = cv.dct(imf) # Extracting DCT Coefficient
            fp = zig_zag(discrete_coeff) # To extract data in zig-zag manner
            for i in range(20):
                sheet.write(ite,i,fp[i])
    workbook.close()
##Function 4.1
# To Use During Searching
def dct_of_test_image(image):
    array =cv.resize(np.asarray(image),(64,64))
    imf = np.float32(array)  # float conversion/scale
    discrete_coeff = cv.dct(imf)           # the dct
    fp = zig_zag(discrete_coeff)
    return fp[:20]
##Function 5
def read_from_excel(filename):
    read_file = pd.read_excel(filename,header=None, index=None) # return a dataframe
    array = np.asarray(read_file) # Converting datframe into numpy array
    return array
##Function 6
def hybrid_feature(filename1,filename2):
    print("Concatenating two.")
    array1 = read_from_excel(filename1)
    array2 = read_from_excel(filename2)
    hybrid_array = np.concatenate((array1,array2),axis=1) # Concatenating Two Features from different file
    return hybrid_array
##Function 6
def Dataframe_to_Excel(array,filename3): # To store a dataframe into a Excel File.
    df = pd.DataFrame(array)
    writer = pd.ExcelWriter(filename3, engine='xlsxwriter') # Creating Writer object from pandas ExcelWriter
    df.to_excel(writer,sheet_name='Sheet1', header=False, index=False)
    writer.close()
##Function 7
# To calculate distance of serach image with the database image for similarity.
def distance(feature_vector,new_feature):
    a = np.asarray(feature_vector)
    b = np.asarray(new_feature)
    summ = 0
    for i in range(len(a)):
        summ = summ + ((a[i]-b[i])**2) # Euclidean Distance Measure
    answer = (summ)**0.5
    return answer
##Function 8
def search_image(new_image,filename):
    excel_file = read_from_excel(filename) # Reading from ExcelFile To search
    row,col = excel_file.shape
    final_array = []
    for i in range(row):
        Distance = distance(excel_file[i],new_image) # Calculating Distance by Using Euclidean Distance Measure.
        final_array.append(Distance)
    #print(min(final_array))    
    A = np.argsort(final_array) # Sorting Data By their indexes in increasing order
    print(A[:10])
    return A # Returning sorted indexes
##Function 9
def search_from_outside(row,col):
    image = input("Enter Full path of image:") # path of image
    image1 = cv.imread(image)
    image2 = cv.resize(image1,(col,row))
    plt.imshow(image2) #plotting image in console.
    plt.title("Test Image")
    plt.xticks([])
    plt.yticks([])
    plt.show()
    return image2
##Function 10    
def inside_case(random_image,list_of_images):
    # Files to be used during Search
    filename_1 = "LbpFeatureMyDB1_32.xlsx"
    filename_2 = "DctFeatureMyDB1_10.xlsx"
    #filename_3 = "HybridFeatureMyDB1_1.xlsx"
    filename_3 = "HybridAfterScaling.xlsx"
    
    while True:
        print("Menu")
        print("1. Search By LBP Feature.")
        print("2. Search By DCT Feature.")
        print("3. Search By Hybrid Feature.")
        print("4. Exit.")
        ch  = int(input("Enter Your Choice. "))
        if ch == 1:
            print("Searching...")
            new_feature = lbp_of_test_image(random_image)
            search_list = search_image(new_feature,filename_1)
            # Printing First 10 images
            for i in range(10):
                val = search_list[i]
                print('processing %s...' % str(i),)
                plt.subplot(2,5,i+1)
                plt.imshow(list_of_images[val])
                plt.title("reslut"+str(i+1))
                plt.xticks([])
                plt.yticks([])
            plt.show()
            print("Finished...")    
        elif ch == 2:
            print("Searching...")
            new_feature = dct_of_test_image(random_image)
            search_list = search_image(new_feature,filename_2)
            # Printing First 10 images
            for i in range(10):
                val = search_list[i]
                print('processing %s...' % str(i),)
                plt.subplot(2,5,i+1)
                plt.imshow(list_of_images[val])
                plt.title("reslut"+str(i+1))
                plt.xticks([])
                plt.yticks([])
            plt.show()
            print("Finished...")    
        elif ch == 3:
            print("searching...")
            new_feature1 = lbp_of_test_image(random_image)
            new_feature2 = dct_of_test_image(random_image)
            hybrid = np.concatenate((new_feature1,new_feature2))
            search_list = search_image(hybrid,filename_3)
            # Printing First 10 images
            for i in range(10):
                val = search_list[i]
                print('processing %s...' % str(i),)
                plt.subplot(2,5,i+1)
                plt.imshow(list_of_images[val])
                plt.title("reslut"+str(i+1))
                plt.xticks([])
                plt.yticks([])
            plt.show()
            print("Finished...") 
        else:
            break

def Result():
    final_result1 = []
    filename_3 = "HybridFeatureMyDB1_1.xlsx"
    image_list = read_from_excel(filename_3)
    Distance_list = []
    for k in range(1000):
        locs = search_image(image_list[k],filename_3)
        Distance_list.append(locs)
    for j in range(1,1001):
        #print(j,'operating point')
        Result = []
        for k in range(0,1000):
            #print(k)
            locs = Distance_list[k]
            counter = int(k/100) + 1
            start = (counter * 100) - 100
            end = (counter * 100)
            relevant_IDs = list(range(start,end))
            final_location = [(locs.tolist()).index(i) for i in relevant_IDs]
            final_sorted = sorted(final_location)
            #print(final_sorted)
            TP = len([final_sorted.index(x) for x in final_sorted if x<j])
            #print(TP)
            FP = j - TP
            FN = 100 - TP
            #TN = 900 - FP
            precision = TP/(TP+FP)
            recall = TP/(TP+FN)
            #print(precision,recall)
            Result.append([precision,recall])
    
        result = np.array(Result)
        final_result = []
        for i in range(0,1000,100):
            final_result.append([(np.sum(result[i:i+100,0])/100),(np.sum(result[i:i+100,1])/100)])
        result1 = np.array(final_result)
        final_result1.append([(np.sum(result1[:,0])/10),(np.sum(result1[:,1])/10)])
        #print(final_result1)
    return final_result1


if __name__ == "__main__": 
    # Files to store Feature of images.
    filename1 = "LbpFeatureMyDB1.xlsx"
    filename2 = "DctFeatureMyDB1.xlsx"
    filename3 = "HybridFeatureMyDB1.xlsx"
    
    # Loading images from a database into a list called "image_list" (Images are Color image).
    image_list = load_images_from_folder('MyDB')
    
    # Getting shape of the first image. Here shape represent row, column & channel(for Color image).
    # In case of grayscale image shape will return only rows and columns.
    row, col, _ = image_list[0].shape
    
    # Preparing a menu for  easy understanding.
    while(True):
        print("Menu")
        print("1. To Extract LBP Feature.")
        print("2. To Extract DCT Feature.")
        print("3. To Extract HYBRID Feature.")
        print("4. To Search for new image.")
        print("5. To Search from Outside.")
        print("6. Result")
        print("7. To Exit.")
        ch = int(input("Enter Your Choice. "))
        if ch==1:
            # Calling Function to Extract lbp features. filename1 =  "LbpFeatureMyDB1_32.xlsx"
            lbp_feature(image_list,filename1)
        elif ch==2:
            #images_list1 = Preprocess_DCT(image_list)
            dct(image_list,filename2)
        elif ch==3:
            hybridfeature = hybrid_feature(filename1,filename2)
            Dataframe_to_Excel(hybridfeature,filename3)
        elif ch==4:
            r_i = rm.choice(image_list)
            plt.imshow(r_i)
            plt.title("Test Image")
            plt.xticks([])
            plt.yticks([])
            plt.show()
            random_image = cv.cvtColor(r_i,cv.COLOR_BGR2GRAY)
            inside_case(random_image,image_list)
        elif ch==5:
            r_i = search_from_outside(row,col)
            test_image = cv.cvtColor(r_i,cv.COLOR_BGR2GRAY)
            inside_case(test_image,image_list)
        elif ch==6:
            result = np.array(Result())
            #with open('Result.pickle','wb') as f:
            #    pickle.dump(result,f)
            #print(result)
            #pickle_in = open('Result.pickle','rb')
            #result = pickle.loads(pickle_in)
            plt.plot(result[:,0],result[:,1])
            break
        else:
            break
