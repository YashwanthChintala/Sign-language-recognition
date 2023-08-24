SIGN LANGUAGE RECOGNITION USING CONVOLUTIONAL NEURAL NETWORKS


ABSTRACT
Sign Language Recognition (SLR) targets on interpreting the sign language into text or speech, so as to facilitate the communication between deaf-mute people and ordinary people. This task has broad social impact, but is still very challenging due to the complexity and large variations in hand actions. Existing methods for SLR use hand-crafted features to describe sign language motion and build classification models based on those features. However, it is difficult to design reliable features to adapt to the large variations of hand gestures. To approach this problem, we propose a novel  convolutional neural network (CNN) which extracts discriminative spatial-temporal features from raw video stream automatically without any prior knowledge, avoiding designing features. To boost the performance, multi-channels of video streams, including color information, depth clue, and body joint positions, are used as input to the  CNN in order to integrate color, depth and trajectory information. We validate the proposed model on a real dataset collected with Microsoft Kinect and demonstrate its effectiveness over the traditional approaches based on hand-crafted features













Introduction:
Sign language, as one of the most widely used communication means for hearing-impaired people, is expressed by variations of hand-shapes, body movement, and even facial expression. Since it is difficult to collaboratively exploit the information from hand-shapes and body movement trajectory, sign language recognition is still a very challenging task. This paper proposes an effective recognition model to translate sign language into text or speech in order to help the hearing impaired communicate with normal people through sign language. 
Technically speaking, the main challenge of sign language recognition lies in developing descriptors to express hand-shapes and motion trajectory. In particular, hand-shape description involves tracking hand regions in video stream, segmenting hand-shape images from complex background in each frame and gestures recognition problems. Motion trajectory is also related to tracking of the key points and curve matching. Although lots of research works have been conducted on these two issues for now, it is still hard to obtain satisfying result for SLR due to the variation and occlusion of hands and body joints. Besides, it is a nontrivial issue to integrate the hand-shape features and trajectory features together. To address these difficulties, we develop a  CNNs to naturally integrate hand-shapes, trajectory of action and facial expression. Instead of using commonly used color images as input to networks like [1, 2], we take color images, depth images and body skeleton images simultaneously as input which are all provided by Microsoft Kinect. 
Kinect is a motion sensor which can provide color stream and depth stream. With the public Windows SDK, the body joint locations can be obtained in real-time as shown in Fig.1. Therefore, we choose Kinect as capture device to record sign words dataset. The change of color and depth in pixel level are useful information to discriminate different sign actions. And the variation of body joints in time dimension can depict the trajectory of sign actions. Using multiple types of visual sources as input leads CNNs paying attention to the change not only in color, but also in depth and trajectory. It is worth mentioning that we can avoid the difficulty of tracking hands, segmenting hands from background and designing descriptors for hands because CNNs have the capability to learn features automatically from raw data without any prior knowledge [3]. 
CNNs have been applied in video stream classification recently years. A potential concern of CNNs is time consuming. It costs several weeks or months to train a CNNs with million-scale in million videos. Fortunately, it is still possible to achieve real-time efficiency, with the help of CUDA for parallel processing. We propose to apply CNNs to extract spatial and temporal features from video stream for Sign Language Recognition (SLR). Existing methods for SLR use hand-crafted features to describe sign language motion and build classification model based on these features. In contrast,  CNNs can capture motion information from raw video data automatically, avoiding designing features. We develop a CNNs taking multiple types of data as input. This architecture integrates color, depth and trajectory information by performing convolution and subsampling on adjacent video frames. Experimental results demonstrate that 3D CNNs can significantly outperform Gaussian mixture model with Hidden Markov model (GMM-HMM) baselines on some sign words recorded by ourselves.
























CHAPTER 2
TECHNOLOGIES LEARNT
What is Python :-
 Below are some facts about  Python.
 Python is currently the most widely used multi-purpose, high-level programming language.
 Python allows programming in Object-Oriented and Procedural paradigms. Python programs generally    are smaller than other programming languages like Java.
 Programmers have to type relatively less and indentation requirement of the language, makes them readable all the time.
Python language is being used by almost all tech-giant companies like – Google, Amazon, Facebook, Instagram, Dropbox, Uber… etc.
The biggest strength of Python is huge collection of standard library which can be used for the following –
•	Machine Learning
•	GUI Applications (like Kivy, Tkinter, PyQt etc. )
•	Web frameworks like Django (used by YouTube, Instagram, Dropbox)
•	Image processing (like OpenCV, Pillow)
•	Web scraping (like Scrapy, BeautifulSoup, Selenium)
•	Test frameworks
•	Multimedia

Advantages of Python :-
Let’s see how Python dominates over other languages.
1. Extensive Libraries	
Python downloads with an extensive library and it contain code for various purposes like regular expressions, documentation-generation, unit-testing, web browsers, threading, databases, CGI, email, image manipulation, and more. So, we don’t have to write the complete code for that manually.
2. Extensible
As we have seen earlier, Python can be extended to other languages. You can write some of your code in languages like C++ or C. This comes in handy, especially in projects.
3. Embeddable
Complimentary to extensibility, Python is embeddable as well. You can put your Python code in your source code of a different language, like C++. This lets us add scripting capabilities to our code in the other language.
4. Improved Productivity
The language’s simplicity and extensive libraries render programmers more productive than languages like Java and C++ do. Also, the fact that you need to write less and get more things done.
5. IOT Opportunities
Since Python forms the basis of new platforms like Raspberry Pi, it finds the future bright for the Internet Of Things. This is a way to connect the language with the real world.
6. Simple and Easy
When working with Java, you may have to create a class to print ‘Hello World’. But in Python, just a print statement will do. It is also quite easy to learn, understand, and code. This is why when people pick up Python, they have a hard time adjusting to other more verbose languages like Java.
7. Readable
Because it is not such a verbose language, reading Python is much like reading English. This is the reason why it is so easy to learn, understand, and code. It also does not need curly braces to define blocks, and indentation is mandatory. This further aids the readability of the code.
8. Object-Oriented
This language supports both the procedural and object-oriented programming paradigms. While functions help us with code reusability, classes and objects let us model the real world. A class allows the encapsulation of data and functions into one.
9. Free and Open-Source
Like we said earlier, Python is freely available. But not only can you download Python for free, but you can also download its source code, make changes to it, and even distribute it. It downloads with an extensive collection of libraries to help you with your tasks.
10. Portable
When you code your project in a language like C++, you may need to make some changes to it if you want to run it on another platform. But it isn’t the same with Python. Here, you need to code only once, and you can run it anywhere. This is called Write Once Run Anywhere (WORA). However, you need to be careful enough not to include any system-dependent features.
11. Interpreted
Lastly, we will say that it is an interpreted language. Since statements are executed one by one, debugging is easier than in compiled languages.
Any doubts till now in the advantages of Python? Mention in the comment section.
Advantages of Python Over Other Languages
1. Less Coding
Almost all of the tasks done in Python requires less coding when the same task is done in other languages. Python also has an awesome standard library support, so you don’t have to search for any third-party libraries to get your job done. This is the reason that many people suggest learning Python to beginners.
2. Affordable
Python is free therefore individuals, small companies or big organizations can leverage the free available resources to build applications. Python is popular and widely used so it gives you better community support.
The 2019 Github annual survey showed us that Python has overtaken Java in the most popular programming language category.

3. Python is for Everyone
Python code can run on any machine whether it is Linux, Mac or Windows. Programmers need to learn different languages for different jobs but with Python, you can professionally build web apps, perform data analysis and machine learning, automate things, do web scraping and also build games and powerful visualizations. It is an all-rounder programming language.

Disadvantages of Python
So far, we’ve seen why Python is a great choice for your project. But if you choose it, you should be aware of its consequences as well. Let’s now see the downsides of choosing Python over another language.
1. Speed Limitations
We have seen that Python code is executed line by line. But since Python is interpreted, it often results in slow execution. This, however, isn’t a problem unless speed is a focal point for the project. In other words, unless high speed is a requirement, the benefits offered by Python are enough to distract us from its speed limitations.
2. Weak in Mobile Computing and Browsers
While it serves as an excellent server-side language, Python is much rarely seen on the client-side. Besides that, it is rarely ever used to implement smartphone-based applications. One such application is called Carbonnelle.
The reason it is not so famous despite the existence of Brython is that it isn’t that secure.
3. Design Restrictions
As you know, Python is dynamically-typed. This means that you don’t need to declare the type of variable while writing the code. It uses duck-typing. But wait, what’s that? Well, it just means that if it looks like a duck, it must be a duck. While this is easy on the programmers during coding, it can raise run-time errors.
4. Underdeveloped Database Access Layers
Compared to more widely used technologies like JDBC (Java DataBase Connectivity) and ODBC (Open DataBase Connectivity), Python’s database access layers are a bit underdeveloped. Consequently, it is less often applied in huge enterprises.
5. Simple
No, we’re not kidding. Python’s simplicity can indeed be a problem. Take my example. I don’t do Java, I’m more of a Python person. To me, its syntax is so simple that the verbosity of Java code seems unnecessary.
This was all about the Advantages and Disadvantages of Python Programming Language.
History of Python  : -

What do the alphabet and the programming language Python have in common? Right, both start with ABC. If we are talking about ABC in the Python context, it's clear that the programming language ABC is meant. ABC is a general-purpose programming language and programming environment, which had been developed in the Netherlands, Amsterdam, at the CWI (Centrum Wiskunde & Informatica). The greatest achievement of ABC was to influence the design of Python.Python was conceptualized in the late 1980s. Guido van Rossum worked that time in a project at the CWI, called Amoeba, a distributed operating system. In an interview with Bill Venners1, Guido van Rossum said: "In the early 1980s, I worked as an implementer on a team building a language called ABC at Centrum voor Wiskunde en Informatica (CWI). I don't know how well people know ABC's influence on Python. I try to mention ABC's influence because I'm indebted to everything I learned during that project and to the people who worked on it."Later on in the same Interview, Guido van Rossum continued: "I remembered all my experience and some of my frustration with ABC. I decided to try to design a simple scripting language that possessed some of ABC's better properties, but without its problems. So I started typing. I created a simple virtual machine, a simple parser, and a simple runtime. I made my own version of the various ABC parts that I liked. I created a basic syntax, used indentation for statement grouping instead of curly braces or begin-end blocks, and developed a small number of powerful data types: a hash table (or dictionary, as we call it), a list, strings, and numbers."
What is Machine Learning : -
Before we take a look at the details of various machine learning methods, let's start by looking at what machine learning is, and what it isn't. Machine learning is often categorized as a subfield of artificial intelligence, but I find that categorization can often be misleading at first brush. The study of machine learning certainly arose from research in this context, but in the data science application of machine learning methods, it's more helpful to think of machine learning as a means of building models of data.
Fundamentally, machine learning involves building mathematical models to help understand data. "Learning" enters the fray when we give these models tunable parameters that can be adapted to observed data; in this way the program can be considered to be "learning" from the data. Once these models have been fit to previously seen data, they can be used to predict and understand aspects of newly observed data. I'll leave to the reader the more philosophical digression regarding the extent to which this type of mathematical, model-based "learning" is similar to the "learning" exhibited by the human brain.Understanding the problem setting in machine learning is essential to using these tools effectively, and so we will start with some broad categorizations of the types of approaches we'll discuss here.
Categories Of Machine Leaning :-
At the most fundamental level, machine learning can be categorized into two main types: supervised learning and unsupervised learning.
Supervised learning involves somehow modeling the relationship between measured features of data and some label associated with the data; once this model is determined, it can be used to apply labels to new, unknown data. This is further subdivided into classification tasks and regression tasks: in classification, the labels are discrete categories, while in regression, the labels are continuous quantities. We will see examples of both types of supervised learning in the following section.
Unsupervised learning involves modeling the features of a dataset without reference to any label, and is often described as "letting the dataset speak for itself." These models include tasks such as clustering and dimensionality reduction. Clustering algorithms identify distinct groups of data, while dimensionality reduction algorithms search for more succinct representations of the data. We will see examples of both types of unsupervised learning in the following section.
Need for Machine Learning
Human beings, at this moment, are the most intelligent and advanced species on earth because they can think, evaluate and solve complex problems. On the other side, AI is still in its initial stage and haven’t surpassed human intelligence in many aspects. Then the question is that what is the need to make machine learn? The most suitable reason for doing this is, “to make decisions, based on data, with efficiency and scale”.
Lately, organizations are investing heavily in newer technologies like Artificial Intelligence, Machine Learning and Deep Learning to get the key information from data to perform several real-world tasks and solve problems. We can call it data-driven decisions taken by machines, particularly to automate the process. These data-driven decisions can be used, instead of using programing logic, in the problems that cannot be programmed inherently. The fact is that we can’t do without human intelligence, but other aspect is that we all need to solve real-world problems with efficiency at a huge scale. That is why the need for machine learning arises.




Challenges in Machines Learning :-

While Machine Learning is rapidly evolving, making significant strides with cybersecurity and autonomous cars, this segment of AI as whole still has a long way to go. The reason behind is that ML has not been able to overcome number of challenges. The challenges that ML is facing currently are −
Quality of data − Having good-quality data for ML algorithms is one of the biggest challenges. Use of low-quality data leads to the problems related to data preprocessing and feature extraction.
Time-Consuming task − Another challenge faced by ML models is the consumption of time especially for data acquisition, feature extraction and retrieval.
Lack of specialist persons − As ML technology is still in its infancy stage, availability of expert resources is a tough job.
No clear objective for formulating business problems − Having no clear objective and well-defined goal for business problems is another key challenge for ML because this technology is not that mature yet.
Issue of overfitting & underfitting − If the model is overfitting or underfitting, it cannot be represented well for the problem.
Curse of dimensionality − Another challenge ML model faces is too many features of data points. This can be a real hindrance.
Difficulty in deployment − Complexity of the ML model makes it quite difficult to be deployed in real life.
Applications of Machines Learning :-

Machine Learning is the most rapidly growing technology and according to researchers we are in the golden year of AI and ML. It is used to solve many real-world complex problems which cannot be solved with traditional approach. Following are some real-world applications of ML −
•	Emotion analysis
•	Sentiment analysis
•	Error detection and prevention
•	Weather forecasting and prediction
•	Stock market analysis and forecasting
•	Speech synthesis
•	Speech recognition
•	Customer segmentation
•	Object recognition
•	Fraud detection
•	Fraud prevention
•	Recommendation of products to customer in online shopping
How to Start Learning Machine Learning?
Arthur Samuel coined the term “Machine Learning” in 1959 and defined it as a “Field of study that gives computers the capability to learn without being explicitly programmed”.
And that was the beginning of Machine Learning! In modern times, Machine Learning is one of the most popular (if not the most!) career choices. According to Indeed, Machine Learning Engineer Is The Best Job of 2019 with a 344% growth and an average base salary of $146,085 per year.
But there is still a lot of doubt about what exactly is Machine Learning and how to start learning it? So this article deals with the Basics of Machine Learning and also the path you can follow to eventually become a full-fledged Machine Learning Engineer. Now let’s get started!!!
How to start learning ML?
This is a rough roadmap you can follow on your way to becoming an insanely talented Machine Learning Engineer. Of course, you can always modify the steps according to your needs to reach your desired end-goal!
Step 1 – Understand the Prerequisites
In case you are a genius, you could start ML directly but normally, there are some prerequisites that you need to know which include Linear Algebra, Multivariate Calculus, Statistics, and Python. And if you don’t know these, never fear! You don’t need a Ph.D. degree in these topics to get started but you do need a basic understanding.
(a) Learn Linear Algebra and Multivariate Calculus
Both Linear Algebra and Multivariate Calculus are important in Machine Learning. However, the extent to which you need them depends on your role as a data scientist. If you are more focused on application heavy machine learning, then you will not be that heavily focused on maths as there are many common libraries available. But if you want to focus on R&D in Machine Learning, then mastery of Linear Algebra and Multivariate Calculus is very important as you will have to implement many ML algorithms from scratch.
(b) Learn Statistics
Data plays a huge role in Machine Learning. In fact, around 80% of your time as an ML expert will be spent collecting and cleaning data. And statistics is a field that handles the collection, analysis, and presentation of data. So it is no surprise that you need to learn it!!!
Some of the key concepts in statistics that are important are Statistical Significance, Probability Distributions, Hypothesis Testing, Regression, etc. Also, Bayesian Thinking is also a very important part of ML which deals with various concepts like Conditional Probability, Priors, and Posteriors, Maximum Likelihood, etc.
(c) Learn Python
Some people prefer to skip Linear Algebra, Multivariate Calculus and Statistics and learn them as they go along with trial and error. But the one thing that you absolutely cannot skip is Python! While there are other languages you can use for Machine Learning like R, Scala, etc. Python is currently the most popular language for ML. In fact, there are many Python libraries that are specifically useful for Artificial Intelligence and Machine Learning such as Keras, TensorFlow, Scikit-learn, etc.
So if you want to learn ML, it’s best if you learn Python! You can do that using various online resources and courses such as Fork Python available Free on GeeksforGeeks.
Step 2 – Learn Various ML Concepts
Now that you are done with the prerequisites, you can move on to actually learning ML (Which is the fun part!!!) It’s best to start with the basics and then move on to the more complicated stuff. Some of the basic concepts in ML are:
(a) Terminologies of Machine Learning
•	Model – A model is a specific representation learned from data by applying some machine learning algorithm. A model is also called a hypothesis.
•	Feature – A feature is an individual measurable property of the data. A set of numeric features can be conveniently described by a feature vector. Feature vectors are fed as input to the model. For example, in order to predict a fruit, there may be features like color, smell, taste, etc.
•	Target (Label) – A target variable or label is the value to be predicted by our model. For the fruit example discussed in the feature section, the label with each set of input would be the name of the fruit like apple, orange, banana, etc.
•	Training – The idea is to give a set of inputs(features) and it’s expected outputs(labels), so after training, we will have a model (hypothesis) that will then map new data to one of the categories trained on.
•	Prediction – Once our model is ready, it can be fed a set of inputs to which it will provide a predicted output(label).
(b) Types of Machine Learning
•	Supervised Learning – This involves learning from a training dataset with labeled data using classification and regression models. This learning process continues until the required level of performance is achieved.
•	Unsupervised Learning – This involves using unlabelled data and then finding the underlying structure in the data in order to learn more and more about the data itself using factor and cluster analysis models.
•	Semi-supervised Learning – This involves using unlabelled data like Unsupervised Learning with a small amount of labeled data. Using labeled data vastly increases the learning accuracy and is also more cost-effective than Supervised Learning.
•	Reinforcement Learning – This involves learning optimal actions through trial and error. So the next action is decided by learning behaviors that are based on the current state and that will maximize the reward in the future.
Advantages of Machine learning :-
1. Easily identifies trends and patterns -
Machine Learning can review large volumes of data and discover specific trends and patterns that would not be apparent to humans. For instance, for an e-commerce website like Amazon, it serves to understand the browsing behaviors and purchase histories of its users to help cater to the right products, deals, and reminders relevant to them. It uses the results to reveal relevant advertisements to them.
2. No human intervention needed (automation)
With ML, you don’t need to babysit your project every step of the way. Since it means giving machines the ability to learn, it lets them make predictions and also improve the algorithms on their own. A common example of this is anti-virus softwares; they learn to filter new threats as they are recognized. ML is also good at recognizing spam.
3. Continuous Improvement 
As ML algorithms gain experience, they keep improving in accuracy and efficiency. This lets them make better decisions. Say you need to make a weather forecast model. As the amount of data you have keeps growing, your algorithms learn to make more accurate predictions faster.
4. Handling multi-dimensional and multi-variety data
Machine Learning algorithms are good at handling data that are multi-dimensional and multi-variety, and they can do this in dynamic or uncertain environments.
5. Wide Applications
You could be an e-tailer or a healthcare provider and make ML work for you. Where it does apply, it holds the capability to help deliver a much more personal experience to customers while also targeting the right customers.
Disadvantages of Machine Learning :-
1. Data Acquisition
Machine Learning requires massive data sets to train on, and these should be inclusive/unbiased, and of good quality. There can also be times where they must wait for new data to be generated.
2. Time and Resources
ML needs enough time to let the algorithms learn and develop enough to fulfill their purpose with a considerable amount of accuracy and relevancy. It also needs massive resources to function. This can mean additional requirements of computer power for you.
3. Interpretation of Results
Another major challenge is the ability to accurately interpret results generated by the algorithms. You must also carefully choose the algorithms for your purpose.
4. High error-susceptibility
Machine Learning is autonomous but highly susceptible to errors. Suppose you train an algorithm with data sets small enough to not be inclusive. You end up with biased predictions coming from a biased training set. This leads to irrelevant advertisements being displayed to customers. In the case of ML, such blunders can set off a chain of errors that can go undetected for long periods of time. And when they do get noticed, it takes quite some time to recognize the source of the issue, and even longer to correct it.

Python Development Steps  : -
Guido Van Rossum published the first version of Python code (version 0.9.0) at alt.sources in February 1991. This release included already exception handling, functions, and the core data types of list, dict, str and others. It was also object oriented and had a module system.
Python version 1.0 was released in January 1994. The major new features included in this release were the functional programming tools lambda, map, filter and reduce, which Guido Van Rossum never liked.Six and a half years later in October 2000, Python 2.0 was introduced. This release included list comprehensions, a full garbage collector and it was supporting unicode.Python flourished for another 8 years in the versions 2.x before the next major release as Python 3.0 (also known as "Python 3000" and "Py3K") was released. Python 3 is not backwards compatible with Python 2.x. The emphasis in Python 3 had been on the removal of duplicate programming constructs and modules, thus fulfilling or coming close to fulfilling the 13th law of the Zen of Python: "There should be one -- and preferably only one -- obvious way to do it."Some changes in Python 7.3:
•	Print is now a function
•	Views and iterators instead of lists
•	The rules for ordering comparisons have been simplified. E.g. a heterogeneous list cannot be     sorted, because all the elements of a list must be comparable to each other.
•	There is only one integer type left, i.e. int. long is int as well.
•	The division of two integers returns a float instead of an integer. "//" can be used to have the "old" behaviour.
•	Text Vs. Data Instead Of Unicode Vs. 8-bit
Purpose :-  
We demonstrated that our approach enables successful segmentation of intra-retinal layers—even with low-quality images containing speckle noise, low contrast, and different intensity ranges throughout—with the assistance of the ANIS feature.
Python
Python is an interpreted high-level programming language for general-purpose programming. Created by Guido van Rossum and first released in 1991, Python has a design philosophy that emphasizes code readability, notably using significant whitespace. 
Python features a dynamic type system and automatic memory management. It supports multiple programming paradigms, including object-oriented, imperative, functional and procedural, and has a large and comprehensive standard library. 
•	Python is Interpreted − Python is processed at runtime by the interpreter. You do not need to compile your program before executing it. This is similar to PERL and PHP. 
•	Python is Interactive − you can actually sit at a Python prompt and interact with the interpreter directly to write your programs. 
Python also acknowledges that speed of development is important. Readable and terse code is part of this, and so is access to powerful constructs that avoid tedious repetition of code. Maintainability also ties into this may be an all but useless metric, but it does say something about how much code you have to scan, read and/or understand to troubleshoot problems or tweak behaviors. This speed of development, the ease with which a programmer of other languages can pick up basic Python skills and the huge standard library is key to another area where Python excels. All its tools have been quick to implement, saved a lot of time, and several of them have later been patched and updated by people with no Python background - without breaking.
Modules Used in Project  :-
Tensorflow
TensorFlow is a free and open-source software library for dataflow and differentiable programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks. It is used for both research and production at Google. 
TensorFlow was developed by the Google Brain team for internal Google use. It was released under the Apache 2.0 open-source license on November 9, 2015.
Numpy
Numpy is a general-purpose array-processing package. It provides a high-performance multidimensional array object, and tools for working with these arrays.
It is the fundamental package for scientific computing with Python. It contains various features including these important ones:
	A powerful N-dimensional array object
	Sophisticated (broadcasting) functions
	Tools for integrating C/C++ and Fortran code
	Useful linear algebra, Fourier transform, and random number capabilities
Besides its obvious scientific uses, Numpy can also be used as an efficient multi-dimensional container of generic data. Arbitrary data-types can be defined using Numpy which allows Numpy to seamlessly and speedily integrate with a wide variety of databases.
Pandas
Pandas is an open-source Python Library providing high-performance data manipulation and analysis tool using its powerful data structures. Python was majorly used for data munging and preparation. It had very little contribution towards data analysis. Pandas solved this problem. Using Pandas, we can accomplish five typical steps in the processing and analysis of data, regardless of the origin of data load, prepare, manipulate, model, and analyze. Python with Pandas is used in a wide range of fields including academic and commercial domains including finance, economics, Statistics, analytics, etc.
Matplotlib
Matplotlib is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms. Matplotlib can be used in Python scripts, the Python and IPython shells, the Jupyter Notebook, web application servers, and four graphical user interface toolkits. Matplotlib tries to make easy things easy and hard things possible. You can generate plots, histograms, power spectra, bar charts, error charts, scatter plots, etc., with just a few lines of code. For examples, see the sample plots and thumbnail gallery.
For simple plotting the pyplot module provides a MATLAB-like interface, particularly when combined with IPython. For the power user, you have full control of line styles, font properties, axes properties, etc, via an object oriented interface or via a set of functions familiar to MATLAB users.
Scikit – learn
Scikit-learn provides a range of supervised and unsupervised learning algorithms via a consistent interface in Python. It is licensed under a permissive simplified BSD license and is distributed under many Linux distributions, encouraging academic and commercial use. Python
Python is an interpreted high-level programming language for general-purpose programming. Created by Guido van Rossum and first released in 1991, Python has a design philosophy that emphasizes code readability, notably using significant whitespace. 
Python features a dynamic type system and automatic memory management. It supports multiple programming paradigms, including object-oriented, imperative, functional and procedural, and has a large and comprehensive standard library. 
•	Python is Interpreted − Python is processed at runtime by the interpreter. You do not need to compile your program before executing it. This is similar to PERL and PHP. 
•	Python is Interactive − you can actually sit at a Python prompt and interact with the interpreter directly to write your programs. 
Python also acknowledges that speed of development is important. Readable and terse code is part of this, and so is access to powerful constructs that avoid tedious repetition of code. Maintainability also ties into this may be an all but useless metric, but it does say something about how much code you have to scan, read and/or understand to troubleshoot problems or tweak behaviors. This speed of development, the ease with which a programmer of other languages can pick up basic Python skills and the huge standard library is key to another area where Python excels. All its tools have been quick to implement, saved a lot of time, and several of them have later been patched and updated by people with no Python background - without breaking.
Install Python Step-by-Step in Windows and Mac :
Python a versatile programming language doesn’t come pre-installed on your computer devices. Python was first released in the year 1991 and until today it is a very popular high-level programming language. Its style philosophy emphasizes code readability with its notable use of great whitespace.
The object-oriented approach and language construct provided by Python enables programmers to write both clear and logical code for projects. This software does not come pre-packaged with Windows.

How to Install Python on Windows and Mac :

There have been several updates in the Python version over the years. The question is how to install Python? It might be confusing for the beginner who is willing to start learning Python but this tutorial will solve your query. The latest or the newest version of Python is version 3.7.4 or in other words, it is Python 3.
Note: The python version 3.7.4 cannot be used on Windows XP or earlier devices.

Before you start with the installation process of Python. First, you need to know about your System Requirements. Based on your system type i.e. operating system and based processor, you must download the python version. My system type is a Windows 64-bit operating system. So the steps below are to install python version 3.7.4 on Windows 7 device or to install Python 3. Download the Python Cheatsheet here.The steps on how to install Python on Windows 10, 8 and 7 are divided into 4 parts to help understand better.

Download the Correct version into the system

Step 1: Go to the official site to download and install python using Google Chrome or any other web browser. OR Click on the following link: https://www.python.org

![image](https://github.com/YashwanthChintala/Sign-language-recognition/assets/143099149/75ffd981-a43e-4a7b-b183-b8ed52d6c52d)

 
Now, check for the latest and the correct version for your operating system.
Step 2: Click on the Download Tab.
 ![image](https://github.com/YashwanthChintala/Sign-language-recognition/assets/143099149/24493c0f-1712-4c1d-b6fb-b02e82f63fe1)

Step 3: You can either select the Download Python for windows 3.7.4 button in Yellow Color or you can scroll further down and click on download with respective to their version. Here, we are downloading the most recent python version for windows 3.7.4
 ![image](https://github.com/YashwanthChintala/Sign-language-recognition/assets/143099149/3f06e39d-8af3-4cc1-a103-5b1fa83105a3)

Step 4: Scroll down the page until you find the Files option.

Step 5: Here you see a different version of python along with the operating system.
![image](https://github.com/YashwanthChintala/Sign-language-recognition/assets/143099149/fec6f07e-b8b0-444d-876e-2fa98001f57c)

 
•	To download Windows 32-bit python, you can select any one from the three options: Windows x86 embeddable zip file, Windows x86 executable installer or Windows x86 web-based installer.
•	To download Windows 64-bit python, you can select any one from the three options: Windows x86-64 embeddable zip file, Windows x86-64 executable installer or Windows x86-64 web-based installer.

Here we will install Windows x86-64 web-based installer. Here your first part regarding which version of python is to be downloaded is completed. Now we move ahead with the second part in installing python i.e. Installation
Note: To know the changes or updates that are made in the version you can click on the Release Note Option.
Installation of Python
Step 1: Go to Download and Open the downloaded python version to carry out the installation process.
 ![image](https://github.com/YashwanthChintala/Sign-language-recognition/assets/143099149/4ca09323-3c12-409c-861f-af2f4bea50f8)

Step 2: Before you click on Install Now, Make sure to put a tick on Add Python 3.7 to PATH.

 ![image](https://github.com/YashwanthChintala/Sign-language-recognition/assets/143099149/04737fb6-fc42-473a-8de6-f95740443275)


Step 3: Click on Install NOW After the installation is successful. Click on Close.
![image](https://github.com/YashwanthChintala/Sign-language-recognition/assets/143099149/b9ecc8e4-b73c-43d4-9a3b-205109d80c89)

 

With these above three steps on python installation, you have successfully and correctly installed Python. Now is the time to verify the installation.
Note: The installation process might take a couple of minutes.

Verify the Python Installation
Step 1: Click on Start
Step 2: In the Windows Run Command, type “cmd”

 ![image](https://github.com/YashwanthChintala/Sign-language-recognition/assets/143099149/48f3379b-9210-48bb-9efd-6c21ccf22199)

 
 
Step 3: Open the Command prompt option.
Step 4: Let us test whether the python is correctly installed. Type python –V and press Enter.
![image](https://github.com/YashwanthChintala/Sign-language-recognition/assets/143099149/887f23c5-1776-4a29-9e9c-d5ca1414a0e8)

 
Step 5: You will get the answer as 3.7.4
Note: If you have any of the earlier versions of Python already installed. You must first uninstall the earlier version and then install the new one. 

Check how the Python IDLE works
Step 1: Click on Start
Step 2: In the Windows Run command, type “python idle”
 ![image](https://github.com/YashwanthChintala/Sign-language-recognition/assets/143099149/0a89fc9a-1818-4523-8e94-a69e4b3a7280)

Step 3: Click on IDLE (Python 3.7 64-bit) and launch the program
Step 4: To go ahead with working in IDLE you must first save the file. Click on File > Click on Save
![image](https://github.com/YashwanthChintala/Sign-language-recognition/assets/143099149/39feb825-987d-4201-b5a3-8e0baf176ba7)

 
Step 5: Name the file and save as type should be Python files. Click on SAVE. Here I have named the files as Hey World.
Step 6: Now for e.g. enter print (“Hey World”) and Press Enter.
![image](https://github.com/YashwanthChintala/Sign-language-recognition/assets/143099149/06d0f292-fe93-466f-a0a3-319c411634c3)

 

You will see that the command given is launched. With this, we end our tutorial on how to install Python. You have learned how to download python for windows into your respective operating system.
Note: Unlike Java, Python doesn’t need semicolons at the end of the statements otherwise it won’t work. 
This stack that includes:
•	world.
Django – Design Philosophies
Django comes with the following design philosophies −
•	Loosely Coupled − Django aims to make each element of its stack independent of the others.
•	Less Coding − Less code so in turn a quick development.
•	Don't Repeat Yourself (DRY) − Everything should be developed only in exactly one place instead of repeating it again and again.
•	Fast Development − Django's philosophy is to do all it can to facilitate hyper-fast development.
•	Clean Design − Django strictly maintains a clean design throughout its own code and makes it easy to follow best web-development practices.
Advantages of Django
Here are few advantages of using Django which can be listed out here −
•	Object-Relational Mapping (ORM) Support − Django provides a bridge between the data model and the database engine, and supports a large set of database systems including MySQL, Oracle, Postgres, etc. Django also supports NoSQL database through Django-nonrel fork. For now, the only NoSQL databases supported are MongoDB and google app engine.
•	Multilingual Support − Django supports multilingual websites through its built-in internationalization system. So you can develop your website, which would support multiple languages.
•	Framework Support − Django has built-in support for Ajax, RSS, Caching and various other frameworks.
•	Administration GUI − Django provides a nice ready-to-use user interface for administrative activities.
•	Development Environment − Django comes with a lightweight web server to facilitate end-to-end application development and testing.
As you already know, Django is a Python web framework. And like most modern framework, Django supports the MVC pattern. First let's see what is the Model-View-Controller (MVC) pattern, and then we will look at Django’s specificity for the Model-View-Template (MVT) pattern.
MVC Pattern
When talking about applications that provides UI (web or desktop), we usually talk about MVC architecture. And as the name suggests, MVC pattern is based on three components: Model, View, and Controller. Check our MVC tutorial here to know more.
Django MVC – MVT Pattern
The Model-View-Template (MVT) is slightly different from MVC. In fact the main difference between the two patterns is that Django itself takes care of the Controller part (Software Code that controls the interactions between the Model and View), leaving us with the template. The template is a HTML file mixed with Django Template Language (DTL).
The following diagram illustrates how each of the components of the MVT pattern interacts with each other to serve a user request −
 
Fig 2.2: Django MVC – MVT Pattern

The developer provides the Model, the view and the template then just maps it to a URL and Django does the magic to serve it to the user.


Jupyter Notebook
The Jupyter Notebook is an open source web application that you can use to create and share documents that contain live code, equations, visualizations, and text. Jupyter Notebook is maintained by the people at Project Jupyter.
Jupyter Notebooks are a spin-off project from the IPython project, which used to have an IPython Notebook project itself. The name, Jupyter, comes from the core supported programming languages that it supports: Julia, Python, and R. Jupyter ships with the IPython kernel, which allows you to write your programs in Python, but there are currently over 100 other kernels that you can also use.
Anaconda  :-


What is Anaconda Python?

Together with a list of Python packages, tools like editors, Python distributions include the Python interpreter. Anaconda is one of several Python distributions. Anaconda is a new distribution of the Python and R data science package. It was formerly known as Continuum Analytics. Anaconda has more than 100 new packages.
This work environment, Anaconda is used for scientific computing, data science, statistical analysis, and machine learning. The latest version of Anaconda 5.0.1 is released in October 2017.
The released version 5.0.1 addresses some minor bugs and adds useful features, such as updated R language support. All of these features weren’t available in the original 5.0.0 release.
This package manager is also an environment manager, a Python distribution, and a collection of open source packages and contains more than 1000 R and Python Data Science Packages.
Why Anaconda for Python?
There’s no big reason to switch to Anaconda if you are completely happy with you regular python. But some people like data scientists who are not full-time developers, find anaconda much useful as it simplifies a lot of common problems a beginner runs into.
Anaconda can help with –
•	Installing Python on multiple platforms
•	Separating out different environments
•	Dealing with not having correct privileges and
•	Getting up and running with specific packages and libraries
How to Download Anaconda 5.0.1?






















3.3.1	HardwareRequirements:

•	Processor - Pentium–III
•	Speed – 2.4GHz
•	RAM - 512 MB(min)
•	Hard Disk - 20 GB
•	Floppy Drive - 1.44MB
•	Key Board - Standard Keyboard
•	Monitor – 15 VGAColour














UML DIAGRAMS:
3.4	DetailedDesign

 UML is an acronym that stands for Unified Modeling Language. Simply put, UML is a modern approach to modeling and documenting software. In fact, it’s one of the most popular business process modeling techniques.
It is based on diagrammatic representations of software components. As the old proverb says: “a picture is worth a thousand words”. By using visual representations, we are able to better understand possible flaws or errors in software or business processes.
UML was created as a result of the chaos revolving around software development and documentation. In the 1990s, there were several different ways to represent and document software systems. The need arose for a more unified way to visually represent those systems and as a result, in 1994-1996, the UML was developed by three software engineers working at Rational Software. It was later adopted as the standard in 1997 and has remained the standard ever since, receiving only a few updates.
GOALS:
The Primary goals in the design of the UML are as follows:
1.	Provide users a ready-to-use, expressive visual modeling Language so that they can develop and exchange meaningfulmodels.
2.	Provide extendibility and specialization mechanisms to extend the coreconcepts.
3.	Be independent of particular programming languages and developmentprocess.
4.	Provide a formal basis for understanding the modelinglanguage.
5.	Encourage the growth of OO toolsmarket.

          6    Support higher level development concepts such as collaborations, frameworks, patterns and components.
7	    Integrate best practices.


i.	USE CASEDIAGRAM:
A use case diagram in the Unified Modeling Language (UML) is a type of behavioral diagram defined by and created from a Use-case analysis. Its purpose is to present a graphical overview of the functionality provided by a system in terms of actors, their goals (represented as use cases), and any dependencies between those use cases. The main purpose of a use case diagram is to show what system functions are performed for which actor. Roles of the actors in the system can be depicted.

User use case Diagram
 







Class 
 



















Sequence

 











Collaboration
 




















Screen Shots:
In this project using CNN we are recognizing hand gesture movement and to train CNN we are using following images shown in below screen shots
 ![image](https://github.com/YashwanthChintala/Sign-language-recognition/assets/143099149/25b7829e-7b1e-4f03-a14c-430c2b53716f)

In above screen we can see we have 10 different types of hand gesture images and to see those images just go inside any folder
 ![image](https://github.com/YashwanthChintala/Sign-language-recognition/assets/143099149/92cd9942-3cb9-4628-b31e-17ad917fd1d8)

In above screen showing images from 0 folder and similarly you can see different images in different folders.
SCREEN SHOTS
To run project double click on run.bat file to get below screen
 ![image](https://github.com/YashwanthChintala/Sign-language-recognition/assets/143099149/b9daa2b5-41ae-4aee-a55b-eef3415e174b)

In above screen click on ‘Upload Hand Gesture Dataset’ button to upload dataset and to get below screen
 ![image](https://github.com/YashwanthChintala/Sign-language-recognition/assets/143099149/fda60df6-6284-4660-8cc2-5eec06ca281a)

In above screen selecting and uploading ‘Dataset’ folder and then click on ‘Select Folder’ button to load dataset and to get below screen
 ![image](https://github.com/YashwanthChintala/Sign-language-recognition/assets/143099149/1ac2c7f5-cbc5-42a1-afc6-a27dfc6f015d)

In above screen dataset loaded and now click on ‘Train CNN with Gesture Images’ button to trained CNN model and to get below screen
 ![image](https://github.com/YashwanthChintala/Sign-language-recognition/assets/143099149/48e2cf00-b316-4dfb-956e-58f5a06f2781)

In above screen CNN model trained on 2000 images and its prediction accuracy we got as 100% and now model is ready and now click on ‘Upload Test Image & Recognize Gesture’ button to upload image and to gesture recognition 
 ![image](https://github.com/YashwanthChintala/Sign-language-recognition/assets/143099149/c7cbe8d1-a337-45e1-a1f9-7c706818582b)

In above screen selecting and uploading ’14.png’ file and then click Open button to get below result
 ![image](https://github.com/YashwanthChintala/Sign-language-recognition/assets/143099149/f460e189-fdd5-4de7-9a62-440ea4ec1ab6)

In above screen gesture recognize as OK and similarly you can upload any image and get result and now click on ‘Recognize Gesture from Video’ button to upload video and get result
 ![image](https://github.com/YashwanthChintala/Sign-language-recognition/assets/143099149/4ef56cd0-4810-4420-8c14-be97b8f23f73)

In above screen selecting and uploading ‘video.avi’ file and then click on ‘Open’ button to get below result
 ![image](https://github.com/YashwanthChintala/Sign-language-recognition/assets/143099149/594d7349-5975-4431-996d-eb79507350dc)

 ![image](https://github.com/YashwanthChintala/Sign-language-recognition/assets/143099149/729de069-bc4b-4796-afd9-7b3b17cf77c5)

 ![image](https://github.com/YashwanthChintala/Sign-language-recognition/assets/143099149/17f4a090-02ab-47c8-a15b-1a85444c45e3)

In above screen as video play then  will get recognition result

















CONCLUSION
We developed a  CNN model for sign language recognition. Our model learns and extracts both spatial and temporal features by performing 3D convolutions. The developed deep architecture extracts multiple types of information from adjacent input frames and then performs convolution and subsampling separately. The final feature representation combines information from all channels. We use multilayer perceptron classifier to classify these feature representations. For comparison, we evaluate both CNN and GMM-HMM on the same dataset. The experimental results demonstrate the effectiveness of the proposed method.




















REFERENCES
[1] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton, “Imagenet classification with deep convolutional neural networks,” in Advances in neural information processing systems, 2012, pp. 1097–1105.
 [2] Andrej Karpathy, George Toderici, Sanketh Shetty, Thomas Leung, Rahul Sukthankar, and Li Fei-Fei, “Large-scale video classification with convolutional neural networks,” in CVPR, 2014. [3] Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick ´ Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 1998. 
[4] Hueihan Jhuang, Thomas Serre, Lior Wolf, and Tomaso Poggio, “A biologically inspired system for action recognition,” in Computer Vision, 2007. ICCV 2007. IEEE 11th International Conference on. Ieee, 2007, pp. 1–8. [5] Shuiwang Ji, Wei Xu, Ming Yang, and Kai Yu, “3D convolutional neural networks for human action recognition,” IEEE TPAMI, vol. 35, no. 1, pp. 221–231, 2013. 
[6] Kirsti Grobel and Marcell Assan, “Isolated sign language recognition using hidden markov models,” in Systems, Man, and Cybernetics, 1997. Computational Cybernetics and Simulation., 1997 IEEE International Conference on. IEEE, 1997, vol. 1, pp. 162–167.
 [7] Thad Starner, Joshua Weaver, and Alex Pentland, “Realtime american sign language recognition using desk and wearable computer based video,” IEEE TPAMI, vol. 20, no. 12, pp. 1371–1375, 1998. 
[8] Christian Vogler and Dimitris Metaxas, “Parallel hidden markov models for american sign language recognition,” in Computer Vision, 1999. The Proceedings of the Seventh IEEE International Conference on. IEEE, 1999, vol. 1, pp. 116–122.
 [9] Kouichi Murakami and Hitomi Taguchi, “Gesture recognition using recurrent neural networks,” in Proceedings of the SIGCHI conference on Human factors in computing systems. ACM, 1991, pp. 237–242. 
[10] Chung-Lin Huang and Wen-Yi Huang, “Sign language recognition using model-based tracking and a 3D hopfield neural network,” Machine vision and applications, vol. 10, no. 5-6, pp. 292–307, 1998. 
[11] Jong-Sung Kim, Won Jang, and Zeungnam Bien, “A dynamic gesture recognition system for the korean sign language (ksl),” Systems, Man, and Cybernetics, Part B: Cybernetics, IEEE Transactions on, vol. 26, no. 2, pp. 354–359, 1996. 
[12] Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik, “Rich feature hierarchies for accurate object detection and semantic segmentation,” arXiv preprint arXiv:1311.2524, 2013. 
[13] Ronan Collobert and Jason Weston, “A unified architecture for natural language processing: Deep neural networks with multitask learning,” in ICML. ACM, 2008, pp. 160–167. 
[14] Clement Farabet, Camille Couprie, Laurent Najman, ´ and Yann LeCun, “Learning hierarchical features for scene labeling,” IEEE TPAMI, vol. 35, no. 8, pp. 1915– 1929, 2013. [15] Srinivas C Turaga, Joseph F Murray, Viren Jain, Fabian Roth, Moritz Helmstaedter, Kevin Briggman, Winfried Denk, and H Sebastian Seung, “Convolutional networks can learn to generate affinity graphs for image segmentation,” Neural Computation, vol. 22, no. 2, pp. 511– 538, 2010.
 [16] Ao Tang, Ke Lu, Yufei Wang, Jie Huang, and Houqiang Li, “A real-time hand posture recognition system using deep neural networks,” ACM Transactions on Intelligent Systems and Technology, 2014. 
[17] Moez Baccouche, Franck Mamalet, Christian Wolf, Christophe Garcia, and Atilla Baskurt, “Sequential deep learning for human action recognition,” in Human Behavior Understanding, pp. 29–39. Springer, 2011.

