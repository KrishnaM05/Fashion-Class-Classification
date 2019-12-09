![fashion-2013-03-0304-dvf-nyfw-fall2013](https://user-images.githubusercontent.com/47801267/70408584-b2666080-1a6e-11ea-9279-a7f49b9d8bcf.gif)
## Fashion Class Classification - Using CNN(Convolutional Neural Network
### ==> Means to NN after Converting features(image pixels) to desired form
![nervecell](https://user-images.githubusercontent.com/47801267/70408760-53edb200-1a6f-11ea-9350-44cebc027473.gif)

The Global fashion industry is valued at 3 Trillion Dollar & Accounts for 2% of World's GDP. Here we are going to use ML technique to classify the Fashion Class to build a virtual stylish assistant that looks at customer's social media images & classify what styles the customers are wearing, The virtual assistant can help retailer to find fashion trends & launch targeted fashion campaigns. Here we are going to use fashion MNIST data which contains images of bags, shoes, dresses, etc. & we are asking the deep network to classify the images into 10 classes.Here We want to build say app or model which can learn & say what kind of class this image belongs to whether it is bag, shorts, dress, hat. That is we want to build a deep learning model that can classify the images into different categories.Here we have gray-scale images & have to classify these gray-scale images into 10 class or 10 cathegories. The Prime example that we are going to build can be Amazon's Echo look style Assistant.

![Boot](https://user-images.githubusercontent.com/47801267/70408772-61a33780-1a6f-11ea-8d9a-961135b668ee.gif)

The fashion MNIST data-set contains 70k images,they are divided into 60k training set & 10k test set. All these 70k images are gray-scaled 28x28 images. For example consider following images of 28x28.

![image](https://user-images.githubusercontent.com/47801267/70408789-74b60780-1a6f-11ea-918c-64017849ab88.png)
![image](https://user-images.githubusercontent.com/47801267/70408796-797abb80-1a6f-11ea-84c6-768b732839ba.png)
![image](https://user-images.githubusercontent.com/47801267/70408807-7e3f6f80-1a6f-11ea-93c4-ebffce8589c3.png)
![image](https://user-images.githubusercontent.com/47801267/70408813-813a6000-1a6f-11ea-8272-74e03ff8cec8.png)

Like the above image of sneakers we have 70k gray-scaled images, We will use binary numbers to show image pixels, for that we consider 0 to 255 numners to show it. For example binary numbers '00000000' represented by 0(white color) & '11111111' represented by 255(most dark color i.e. black) & other numbers shoes the in-between gray-scaled(shades) colours. These images are bunch of pixels. This is how we come to conversion from binary to 256 numbers. The data set rows represents the images & columns represents the pixels, Here we have 28x28 pixels this is way we have 28x28 = 784 independent pixel columns & 1 label column of 10 categories labeled as from '0 to 9'
