![fashion-2013-03-0304-dvf-nyfw-fall2013](https://user-images.githubusercontent.com/47801267/70408584-b2666080-1a6e-11ea-9279-a7f49b9d8bcf.gif)
## Fashion Class Classification - Using CNN(Convolutional Neural Network
###### ==> Means to NN after Converting features(image pixels) to desired form
![nervecell](https://user-images.githubusercontent.com/47801267/70408760-53edb200-1a6f-11ea-9350-44cebc027473.gif)

The Global fashion industry is valued at 3 Trillion Dollar & Accounts for 2% of World's GDP. Here we are going to use ML technique to classify the Fashion Class to build a virtual stylish assistant that looks at customer's social media images & classify what styles the customers are wearing, The virtual assistant can help retailer to find fashion trends & launch targeted fashion campaigns. Here we are going to use fashion MNIST data which contains images of bags, shoes, dresses, etc. & we are asking the deep network to classify the images into 10 classes.Here We want to build say app or model which can learn & say what kind of class this image belongs to whether it is bag, shorts, dress, hat. That is we want to build a deep learning model that can classify the images into different categories.Here we have gray-scale images & have to classify these gray-scale images into 10 class or 10 cathegories. The Prime example that we are going to build can be Amazon's Echo look style Assistant.

![Boot](https://user-images.githubusercontent.com/47801267/70408772-61a33780-1a6f-11ea-8d9a-961135b668ee.gif)

The fashion MNIST data-set contains 70k images,they are divided into 60k training set & 10k test set. All these 70k images are gray-scaled 28x28 images. For example consider following images of 28x28.

![image](https://user-images.githubusercontent.com/47801267/70408789-74b60780-1a6f-11ea-918c-64017849ab88.png)
![image](https://user-images.githubusercontent.com/47801267/70408796-797abb80-1a6f-11ea-84c6-768b732839ba.png)
![image](https://user-images.githubusercontent.com/47801267/70408807-7e3f6f80-1a6f-11ea-93c4-ebffce8589c3.png)
![image](https://user-images.githubusercontent.com/47801267/70408813-813a6000-1a6f-11ea-8272-74e03ff8cec8.png)

Like the above image of sneakers we have 70k gray-scaled images, We will use binary numbers to show image pixels, for that we consider 0 to 255 numners to show it. For example binary numbers '00000000' represented by 0(white color) & '11111111' represented by 255(most dark color i.e. black) & other numbers shoes the in-between gray-scaled(shades) colours. These images are bunch of pixels. This is how we come to conversion from binary to 256 numbers. The data set rows represents the images & columns represents the pixels, Here we have 28x28 pixels this is way we have 28x28 = 784 independent pixel columns & 1 label column of 10 categories labeled as from '0 to 9'

### Label values ==> 0: T-shirt/Top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat, 5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle Boot
![image](https://user-images.githubusercontent.com/47801267/70409707-c3b16c00-1a72-11ea-9220-c80877b69794.png)
![image](https://user-images.githubusercontent.com/47801267/70409746-e0e63a80-1a72-11ea-8d8f-24ae1f9d4a00.png)
![image](https://user-images.githubusercontent.com/47801267/70409808-11c66f80-1a73-11ea-92fe-a3ddbec0c7ac.png)
![image](https://user-images.githubusercontent.com/47801267/70409814-15f28d00-1a73-11ea-8ea0-eedd5c04cf18.png)
![image](https://user-images.githubusercontent.com/47801267/70409836-27d43000-1a73-11ea-9cc5-b8b3da2fa331.png)
![image](https://user-images.githubusercontent.com/47801267/70409839-2a368a00-1a73-11ea-8c34-94060b798c7e.png)
![image](https://user-images.githubusercontent.com/47801267/70409867-43d7d180-1a73-11ea-9fea-1dc0cfa6a019.png)
![image](https://user-images.githubusercontent.com/47801267/70409886-57833800-1a73-11ea-9e8b-340a61c18885.png)
![image](https://user-images.githubusercontent.com/47801267/70409900-5d791900-1a73-11ea-8347-2010892d2e87.png)
![1_bhFifratH9DjKqMBTeQG5A](https://user-images.githubusercontent.com/47801267/70409917-62d66380-1a73-11ea-8f25-211b7c814559.gif)
![image](https://user-images.githubusercontent.com/47801267/70409951-7aade780-1a73-11ea-9935-2fb0f23cad33.png)
![1_ZCjPUFrB6eHPRi4eyP6aaA](https://user-images.githubusercontent.com/47801267/70409961-81d4f580-1a73-11ea-8b5b-5272ae9a19bf.gif)
![padding](https://user-images.githubusercontent.com/47801267/70409972-8b5e5d80-1a73-11ea-81b8-b059a1ae7b1e.gif)
![convSobel](https://user-images.githubusercontent.com/47801267/70409986-91543e80-1a73-11ea-837e-1722503cb5af.gif)
![image](https://user-images.githubusercontent.com/47801267/70410008-a03af100-1a73-11ea-914a-33bed4ca4981.png)
![activation-functions](https://user-images.githubusercontent.com/47801267/70410017-a5983b80-1a73-11ea-80ce-30ead0545da3.gif)
![image](https://user-images.githubusercontent.com/47801267/70410034-b2b52a80-1a73-11ea-9876-aa7f53a18bbc.png)
![maxpool_animation](https://user-images.githubusercontent.com/47801267/70410039-b9dc3880-1a73-11ea-9890-a4b94a1fd718.gif)
![73_blog_image_1](https://user-images.githubusercontent.com/47801267/70410048-c52f6400-1a73-11ea-8124-5fe8aa30b95c.png)
![image](https://user-images.githubusercontent.com/47801267/70410067-cfe9f900-1a73-11ea-9580-7f821a98e957.png)
![catordog-flow](https://user-images.githubusercontent.com/47801267/70410079-d7110700-1a73-11ea-85b1-cfa49dfe198a.gif)
![image](https://user-images.githubusercontent.com/47801267/70410099-e6905000-1a73-11ea-8f37-84d6b57781ed.png)
![image](https://user-images.githubusercontent.com/47801267/70410110-eee88b00-1a73-11ea-9436-00561621cef5.png)

![image](https://user-images.githubusercontent.com/47801267/70410291-8f3eaf80-1a74-11ea-9089-7f8481d2cdc3.png)
![image](https://user-images.githubusercontent.com/47801267/70410323-ac737e00-1a74-11ea-8cda-e160b8f60a57.png)
![image](https://user-images.githubusercontent.com/47801267/70410372-e2b0fd80-1a74-11ea-95bc-a6151f668014.png)
![image](https://user-images.githubusercontent.com/47801267/70410403-fceadb80-1a74-11ea-9296-5da0683fde02.png)
![image](https://user-images.githubusercontent.com/47801267/70410419-0aa06100-1a75-11ea-9e22-62b68f4000a0.png)
![image](https://user-images.githubusercontent.com/47801267/70410427-0d9b5180-1a75-11ea-81a3-3eb3c3193f09.png)
![image](https://user-images.githubusercontent.com/47801267/70410461-24da3f00-1a75-11ea-9c84-d781f5781026.png)
![image](https://user-images.githubusercontent.com/47801267/70410469-2b68b680-1a75-11ea-9195-ab177f474d99.png)
![image](https://user-images.githubusercontent.com/47801267/70410492-42a7a400-1a75-11ea-95a8-da113e5ec774.png)
![image](https://user-images.githubusercontent.com/47801267/70410496-4509fe00-1a75-11ea-8c75-5cb05f30ce97.png)
![image](https://user-images.githubusercontent.com/47801267/70410514-5c48eb80-1a75-11ea-8ba4-7e5ff9a9fde5.png)
![image](https://user-images.githubusercontent.com/47801267/70410544-6ff45200-1a75-11ea-81de-7a80282c56ba.png)
![image](https://user-images.githubusercontent.com/47801267/70410567-84384f00-1a75-11ea-8001-5d452168af1d.png)
![image](https://user-images.githubusercontent.com/47801267/70410572-869aa900-1a75-11ea-8a13-bac468ac5621.png)
![image](https://user-images.githubusercontent.com/47801267/70410600-9dd99680-1a75-11ea-812c-5bf81e39844a.png)
![1IrdJ5PghD9YoOyVAQ73MJw](https://user-images.githubusercontent.com/47801267/70410616-a7fb9500-1a75-11ea-8ed2-aecfb2e84911.gif)
![image](https://user-images.githubusercontent.com/47801267/70410638-b9dd3800-1a75-11ea-96f0-2fbc705925d4.png)
![image](https://user-images.githubusercontent.com/47801267/70410662-d5484300-1a75-11ea-98d4-b875e3b5f948.png)
![image](https://user-images.githubusercontent.com/47801267/70410671-da0cf700-1a75-11ea-8df9-1cacd0efd8ad.png)
![image](https://user-images.githubusercontent.com/47801267/70410694-edb85d80-1a75-11ea-9ccc-24d557ba2ceb.png)
![image](https://user-images.githubusercontent.com/47801267/70410699-f0b34e00-1a75-11ea-9042-e6e78d71f785.png)

Advanced Techniques can be used using following dataset    

**Check out:** https://vue.ai/   
**Deep Fashion Dataset:** http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
# NOTE : Checkout ipython file to contribute to the project
**link:** https://github.com/GreatestAnalyst/Fashion-Class-Classification/blob/master/Fashion%20Class%20Classification.ipynb

![1_oB3S5yHHhvougJkPXuc8og](https://user-images.githubusercontent.com/47801267/70410914-a7173300-1a76-11ea-9601-d7e4db3c606c.gif)
