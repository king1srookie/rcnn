# RCNN 
Rich feature hierarchies for accurate object detection and semantic segmentation   

# 工程内容
这个程序是基于tensorflow实现RCNN功能,其能识别并定位常见的宠物，初衷是为了防止当朋友们在谈论宠物时,自己却叫不上名字的尴尬  

# 开发环境  
windows10 + python3.5 + tensorflow1.6 + tflearn + cv2 + scikit-learn   
Ryzen 5 2600 + GTX 1070Ti   

# 数据集
采用pets据集, 官网下载：http://www.robots.ox.ac.uk/~vgg/data/pets/ 

# 程序说明   
1、config.py---网络定义、训练与数据处理所需要用到的参数      
2、Networks.py---用于定义Alexnet_Net模型、fineturn模型、SVM模型、边框回归模型   
4、process_data.py---用于对训练数据集与微调数据集进行处理（选择性搜索、数据存取等）    
5、train_and_test.py---用于各类模型的训练与测试、主函数     
6、selectivesearch.py---选择性搜索源码       


# 文件说明   
1、train_list.txt---预训练数据，数据在17Pets文件夹中

2、fine_tune_list.txt---微调数据2Pets文件夹中

3、直接用选择性搜索的区域划分　　  

4、通过RCNN后的区域划分

5、通过SVM与边框回归之后的最终结果
![selectivesearch_1](https://github.com/king1srookie/rcnn/raw/master/result/1.PNG)
![RCNN_1](https://github.com/king1srookie/rcnn/raw/master/result/2.PNG)　　　
![RCNN_2](https://github.com/king1srookie/rcnn/raw/master/result/3.PNG)
                        



# 参考   
1、论文参考：        
   https://www.computer.org/csdl/proceedings/cvpr/2014/5118/00/5118a580-abs.html          
2、代码参考：     
   https://github.com/yangxue0827/RCNN     
   https://github.com/edwardbi/DeepLearningModels/tree/master/RCNN          
3、博客参考：       
   http://blog.csdn.net/u011534057/article/details/51218218        
   http://blog.csdn.net/u011534057/article/details/51218250        
  
