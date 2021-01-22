---
layout: post
title: Recurrent Neural Networks&#58; LSTM, GRU
tags: [RNN, NLP]
image: /assets/rnn-images/rnnfront.png
description: Simplified&#58; Theory, Mathematics, and Flow behind RNN, LSTM and GRU
comments: true
---
{% include elements/figure.html image="/assets/rnn-images/rnntop.png" %}
## **Overview**
<br>
In this article, I have collated my learnings of Recurrent Neural Networks and its two other implementations - LSTM and GRU, in a simple to understand language, all at one place. This post breaks down - in simpler terms - the Theory, Mathematics, and Flow behind Recurrent Neural Networks and their complex equations. 
<br><br>

### **Table of Content**
<br> 

• [Recurrent Neural Network](#introduction-to-recurrent-nn)<br>
&emsp;&nbsp;• [Working of RNN](#working-of-recurrent-neural-network)<br>
&emsp;&nbsp;• [Why RNN Suffers?](#why-recurrent-neural-network-suffers)<br>

• [Long Short Term Memory (LSTM)](#introduction-to-long-short-term-memory-lstm)<br>
&emsp;&nbsp;• [Memory Cell](#memory-cell)<br>
&emsp;&nbsp;• [Gates](#gates)<br>
&emsp;&nbsp;• [Flow of Information in a LSTM Cell](#flow-of-information-in-a-lstm-cell)<br>
&emsp;&emsp;&nbsp;• [i. Inputs and Outputs](#i-inputs-and-outputs-in-lstm-cell)<br>
&emsp;&emsp;&nbsp;• [ii. Candidate Value](#ii-candidate-value-lstm)<br>
&emsp;&emsp;&nbsp;• [iii. Input, Forget, Output Gates](#iii-input-forget-output-gates-lstm)<br>
&emsp;&emsp;&nbsp;• [iv. New Cell State: Working of Forget, Input Gates](#iv-cell-state--working-of-forget-and-input-gates-lstm)<br>
&emsp;&emsp;&nbsp;• [v. New Hidden State: Working of Output Gate](#v-hidden-state--working-of-output-gate-lstm)<br>

• [Gated Recurrent Units (GRU)](#introduction-to-gated-recurrent-units-gru)<br>
&emsp;&nbsp;• [Flow of Information in a GRU cell](#flow-of-information-in-gru)<br>
&emsp;&emsp;&nbsp;• [i. Inputs and Outputs](#i-inputs-and-outputs-in-gru-cell)<br>
&emsp;&emsp;&nbsp;• [ii. Candidate Value](#ii-candidate-value-gru)<br>
&emsp;&emsp;&nbsp;• [iii. New Hidden State: Working of Update Gate](#iii-new-hidden-state--working-of-update-gate-gru)<br>
<br>

## **Introduction to Recurrent NN**
<br>
Recurrent Neural Network is a type of Neural Network where the output from the previous cell is fed into the next cell, and so on. This way the parameters learned from the previous cells are shared across the network. Whereas in Traditional Neural Networks, each Input (x) is mapped to its corresponding Output (y) and the learned parameters for the inputs are not shared across the network.

{% include elements/figure.html image="/assets/rnn-images/annvsrnn.png" caption="<b>Traditional Neural Network vs Recurrent Neural Network.</b>" %}

In simple terms:
* In the **Traditional Neural Network**, the parameters learned for one training example are **independent** of the parameters learned from the rest of the training examples.

* In the **Recurrent Neural Network**, the parameters learned for one training example are **dependent** on the parameters learned from the rest of the training examples.

RNNs are very useful in Natural Language Processing (NLP) and Natural Language Understanding (NLU) where the meaning of each word is understood based on the context from previous words. They have a wide variety of applications in Sequence Generation, Music Generation, Machine Language Translation, Sentiment Analysis, Image Captioning, and many more.

These kinds of networks are mostly used in learning features in textual data. 
<br><br>

### **Working of Recurrent Neural Network**
<br>
Let's take an example of Movie Review: In a review – “<b><span style="color:#f9cb9c">This </span><span style="color:#93c47d">movie </span><span style="color:#3c78d8">was </span><span style="color:#f1c232">brilliant</span></b>.”, the word “brilliant” is based on the previous words “This movie was”. So, a RNN learns that the word “brilliant” is used to describe the “movie”. Hence the model predicts it as a positive review.

{% include elements/figure.html image="/assets/rnn-images/rnn.png" caption="<b>Example: Recurrent Neural Network for Sentiment Analysis.</b>" %}

Understanding the above Figure:

* The review “The movie was brilliant.” is broken into words (This, movie, was, brilliant) and each word is treated as a timestamp (total 4 timestamps, 1 for each word
 
* The values – x1, x2, x3, x4 are the corresponding embedding vectors for the words – This, movie, was, brilliant
<p align="center">
	This&nbsp;&nbsp;&nbsp;&nbsp; → &nbsp;&nbsp;embedding vector x1<br>
	Movie &nbsp;→ &nbsp;&nbsp;embedding vector x2<br>
	Was &nbsp;&nbsp;&nbsp;&nbsp;→ &nbsp;&nbsp;embedding vector x3<br> 
	Brilliant&nbsp;→ &nbsp;&nbsp;embedding vector x4
<br>
  
</p>

* The values – h1, h2, h3, h4 are the hidden states at each timestamp. The exception is h0, which is initialized beforehand to be fed into the first timestamp.

	**Definition of Hidden State:** Hidden State contains the contextual information, in form of numbers, from current and past cells/timestamps.
	
	**Calculation of Hidden State**: The inputs h<t-1> and x<t> at every timestamp “t” multiplied with their weights and added with some bias, when passed through a non-linearity function returns what’s called an <b>activation value</b>, often called a hidden state.
	
	<p style="font-family: 'Times New Roman'; font-size:125%" align="center">h<sup>t</sup> = g(W<sub>hh</sub>* h<sup>t-1</sup> + W<sub>xh</sub>*x<sup>t</sup> + b)<p><br>
	
	Because the hidden state h<sup>t</sup> uses both h<sup>t-1</sup> and x<sup>t</sup> as inputs, thus it contains the information from the previous RNN cells as well as the current RNN cell. In the  example, the hidden state h<3> uses h\<2> as well as x\<3> as inputs, thus it contains the information up until the 3rd cell i.e. up until “This movie was”. This hidden state h<3> is then fed into the 4th RNN cell, where the information for word “brilliant” is incorporated and a new hidden state h<4> is returned.

* Hidden State h\<4> from the last RNN cell is passed through a Sigmoid Function to get “ŷ” and based on a threshold of 0.5, the review will be either classified as positive or negative.


* Difference between *ŷ* and *y* is is called Error, which is used to calculate Loss. Loss is then backpropagation through the entire RNN through time, and each parameter is updated based on the gradients calculated. This goes on for multiple epochs.
<br><br>

### **Why Recurrent Neural Network Suffers?**
<br>
The major drawback of a RNN is that it cannot remember long term information. This happens because the derivatives / gradients from the past cells become smaller and smaller as the distance increases. This phenomenon is known as Vanishing Gradients.

**How does the Gradients Vanish?**
During the Forward Propagation, every time a hidden state is calculated, some new information from the current timestamp is baked into the hidden state. To make room for this new information, the old information has to be shrinked down. As it happens, proportion of the information from the distant cells/ timestamps shrinks down to very small values.

During the Backward Propagation, the loss is back propagated and gradients are calculated through time. At each timestamp, the information from distant timestamps was already shrunk down to very small numbers, hence the calculated gradients get way smaller. At any particular timestamp, the gradients for distant timestamps eventually reaches almost 0, and when it happens the information from them is almost completely lost.

{% include elements/figure.html image="/assets/rnn-images/rnn-gradient.png" caption="Visualizing Vanishing Gradients in Recurrent Neural Network." %}

**Lack of Control over Information**:
One of the other drawbacks of RNN, is that it does not offer any control over information passed on from the different timestamps/cells. There is no finer control over what relevant information to preserve and what relevant information to discard. Even the irrelevant information is passed onto the next cell, which takes up the space which could have been used for new relevant information.
<br><br>
## **Introduction to Long Short Term Memory (LSTM)**
<br>
Recurrent Neural Network suffers from the problem of vanishing gradients and lack of control over information passed on. This makes them bad at learning long term connections.

A more robust implementation of recurrent networks is called LSTM, which is very effective at dealing with the issues faced by Traditional RNN. LSTM outperforms it in terms of learning both the long term and the short term dependencies, hence the name **Long-Short Term Memory**. It introduces two new concepts viz. Memory Cell and Gates.

{% include elements/figure.html image="/assets/rnn-images/lstm-layer.png" caption="<b>A LSTM Layer consisting of 3 Cells/Timestamps.</b>" %}

### **Memory Cell**
<br>
Firstly, the LSTM introduces the concept of Memory Cell (denoted by the letter “c”). The Memory Cell is the memory of a LSTM cell. Exactly like humans have memory through which we can remember details from way back in life, LSTM has a memory cell that allows it to remember information from very distant timestamps.

Compared to a RNN cell which outputs only the “hidden state”, a LSTM cell outputs both “hidden state” as well as “cell state”.
* Hidden State acts more like a short-term memory of the network.

* Whereas, Cell State acts like a long-term memory of the network.
<br><br>

### **Gates**
<br>
Secondly, the LSTM introduces the concept of Gates. Gates can be thought of exactly like the doors we have in houses, only certain people who are relevant will be allowed to enter through the door and rest are kept outside. Similarly, Gates allow LSTM to control what information is to be kept/withheld and what information is to be forgotten.

As we saw in Traditional RNN, entire information – whether relevant or not – was passed to the next RNN cell/timestamp. But Gates regulate the flow of information into and out of the cell, and only the relevant information is passed through.
<br><br>

### **Flow of Information in a LSTM Cell**
<br>
LSTM involves some complex mathematical equations. Best way to understand the intuition and meaning of those equations is to look at the flow of information in a LSTM network.

{% include elements/figure.html image="/assets/rnn-images/inside-lstm.png" caption="Inside a LSTM Cell/Timestamp." %}
<br>

Understanding the LSTM Cell in above figure:

#### **i. Inputs and Outputs in LSTM cell**
<br>
First, we need to know what are the Inputs and what are the Outputs of a LSTM cell/timestamp.

{% include elements/figure.html image="/assets/rnn-images/lstm-io.png" caption="<b>LSTM Cell: Inputs and Outputs.</b>" %}

**Inputs**:
* c\<t-1>: “cell state” from previous LSTM cell/timestamp

* h\<t-1>: “hidden state” from previous LSTM cell/timestamp

* x\<t>: embedding vector for the word at timestamp “t”

**Outputs**:
* c\<t>: “cell state” of LSTM cell/timestamp

* h\<t>: “hidden state” of LSTM cell/timestamp
<br><br>

#### **ii. Candidate Value (LSTM)**
<br>
A candidate value contains new information to be baked into the outputs of a cell.

However, the extent to which this value will be used in the outputs of the cell will be decided by the (Input) Gate. It may be possible that none of this information is used or vice versa. This makes it just a candidate for replacing the values in the outputs of the cell. And because of this reason, this value is called a “candidate” value. It is denoted by c̃ (pronounced as c tilde).

{% include elements/figure.html image="/assets/rnn-images/lstm-ctilde.png" caption="<b>LSTM Cell: Candidate Value.</b>" %}

It is calculated using the “previous hidden state” h<t-1> and the “word” x<t>. Given by the equation:
<br><br>

<p style="font-family: 'Times New Roman'; font-size:125%" align="center">c̃<sup>&nbsp;&nbsp;t</sup> = tanh(W<sub>ch</sub>* h<sup>t-1</sup> + W<sub>cx</sub>*x<sup>t</sup> + b<sub>c</sub>)</p><br>

**Components of the equation**:
* “previous hidden state” h<sup>t-1</sup> contains information (mostly short-term) from previous timestamps.
 
* “word” x<sup>t</sup> contains new information about word at the current timestamp.

* W<sub>ch</sub> and W<sub>cx</sub> are the weight parameters which will be learned by the cell at current cell/timestamp for h<sup>t-1</sup> and x<sup>t</sup> respectively.

* b<sub>c</sub> is the bias parameter

So the Candidate Value contains the **“Information from previous hidden state”** + **“Information from current word”** + Some **“Bias”**. This is then passed through the **“tanh”** function which controls the vanishing and exploding gradients problem by returning values between -1 and 1.
<br><br>

#### **iii. Input, Forget, Output Gates (LSTM)**
<br>
Gates allow the network to control what information is to be kept and upto what extent. They are used to control the flow of information in outputs of a cell i.e. in “cell state” and “hidden state”. There are three gates, each plays a different role. They return values between 0 and 1. Each gate can be thought of like knobs, if turned to 0 it will “discard everything” and if turned to 1 it will “keep everything”, but usually the values are somewhere in between 0 and 1 so that some proportion of information is kept.

A typical gate is given by the following equation:
<br><br>

<p style="font-family: 'Times New Roman'; font-size:125%" align="center">𝚪 = Sigmoid(W* h<sup>t-1</sup> + W* x<sup>t</sup> + b)</p><br>
The above equation looks very similar to the equation for candidate value. The difference is the Sigmoid Function, which returns values between 0 and 1. These values when multiplied with information (such as candidate value, etc), decides how much information is to be used i.e. it dictates the flow of information in and out of a LSTM cell.

Equation for Gates viz Forget, Input, and Output:
<br><br>
<p style="font-family: 'Times New Roman'; font-size:125%" align="center">
𝚪<sub>f</sub> = Sigmoid(W<sub>fh</sub>* h<sup>t-1</sup> + W<sub>fx</sub>* x<sup>t</sup> + b<sub>f</sub>)<br><br>
𝚪<sub>i</sub> = Sigmoid(W<sub>ih</sub>* h<sup>t-1</sup> + W<sub>ix</sub>* x<sup>t</sup> + b<sub>i</sub>)<br><br>
𝚪<sub>o</sub> = Sigmoid(W<sub>oh</sub>* h<sup>t-1</sup> + W<sub>ox</sub>* x<sup>t</sup> + b<sub>o</sub>)
</p><br>
|Gates|Controls flow of Information to|
|--|--|
|“Input Gate” and “Forget Gate”|“cell state c<t>”|
|“Output Gate”|“hidden state h<t>”|
<br><br>

#### **iv. Cell State // Working of Forget and Input Gates (LSTM)**
<br>
Cell State is the memory of the LSTM cell. It stores useful information from the previous timestamps, as well as the current timestamp. It is given by the equation:
<br>
<p style="font-family: 'Times New Roman'; font-size:125%" align="center">
c<sup>t</sup> = 𝚪<sub>i</sub>* c̃<sup>&nbsp;&nbsp;t</sup> + 𝚪<sub>f</sub>* c<sup>t-1</sup></p>
{% include elements/figure.html image="/assets/rnn-images/lstm-cs.png" caption="<b>LSTM Cell: Inputs and Outputs.</b>" %}

In the “cell state” at each cell, two things happen:

1. **Old Information from distant cells may be used in the outputs of the current cell**: In this case, once that part of old information is used, it is no longer required to be stored in memory.

	**Forget Gate** outputs a value between 0 and 1 to determine how much old information from previous memory c<t-1> is to be forgotten. 0 means “completely get rid of that information”, 1 means “keep the entire information”. Any value in between 0 and 1, means keep that proportion of existing information intact.

2. **New Information from the current cell is added to the cell state**: In this case, some new information is to be stored in memory for future use.

	**Input Gate** outputs a value between 0 and 1 to determine how much new information from “candidate value c tilde <t>” is to be inserted in the memory.

The new memory c<sup>t</sup> comprises “**new information from candidate value c̃**” + “**information from previous memory c<sup>t-1</sup> which is still useful**”.
<br><br>

#### **v. Hidden State // Working of Output Gate (LSTM)**
<br>
Hidden State contains the **information which is to be disclosed to the next LSTM cell/timestamp**. The cell state c<sup>t</sup> is used along with the Output Gate which determines what extent of information from memory cell is to be revealed in the hidden state h<sup>t</sup>.
     
The Hidden State uses the “tanh” function to avoid the problem of exploding and vanishing gradients by converting the values between -1 and 1.
<br>
<p style="font-family: 'Times New Roman'; font-size:125%" align="center">
h<sup>t</sup> = 𝚪<sub>o</sub> * tanh(c<sup>t</sup>)</p>
{% include elements/figure.html image="/assets/rnn-images/lstm-hs.png" caption="<b>LSTM Cell: Inputs and Outputs.</b>" %}

**In conclusion**, 
* a LSTM cell gets as input: “previous cell state”, “previous hidden state” and “new word embedding/any other input value” at each timestamp. Based on these 3 inputs, candidate value, 3 gates are calculated. Next, 

* The Forget Gate is multiplied with “previous cell state” and to find out what **old unnecessary information** is to be **forgotten**.

* The Insert Gate is multiplied with “candidate value” to find out what **new information** is to be **inserted**. 

* This gives us the “new cell state”.

* The Output Gate is multiplied with the "new cell state" to find out what information is to be dicslosed in the "new hidden state".

* These two outputs are then fed onto the next LSTM **cell**. This goes on for all the timestamps. And when it's done, the output of the LSTM **layer**, can then be fed to the next Layer in the Neural Network.

To Remember: Based on the timestamps in our dataset, the number of **Cells** in a LSTM **Layer** are determined. This is the reason why each input is padded to be of the same size in the preprocessing stage.
<br><br>

## **Introduction to Gated Recurrent Units (GRU)**
<br>
In Long Short Term Memory, the concept of Memory Cell was introduced, which acts like a conveyor belt throughout the network storing information from distant timestamps. But LSTM can be computationally expensive. To deal with this issue, a new simpler implementation of LSTM was introduced.

Instead of using a seperate cell state and hidden state, GRU combines the cell state with the hidden state. This simplifies the model and performance wise it does not take that big a hit. Though not as powerful as a LSTM network, it trades off some performance to save time/cost. The concept of Gates still exists in GRU.

**Notation-Wise**:
* In Simple RNN, there is just one output: hidden state h<sup>t</sup>.

* In LSTM, there are two outputs: hidden state h<sup>t</sup> and cell state c<sup>t</sup>.

* In GRU, there is just one output, which combines both long term and short term information, it is denoted by c<sup>t</sup> (instead of denoting it by just h<sup>t</sup> like in Simple RNN).
<br><br>

### **Flow of Information in GRU**
<br>
There are only a few differences in GRU when compared to LSTM. One of the differences is in output, and the other is in gate.

Unlike a dedicated Forget and Insert Gate in LSTM. GRU only has one gate – Update Gate, taking care of both tasks of updating new information and forgetting old unnecessary information.

Lets understand the flow of GRU in more depth.
<br><br>

#### **i. Inputs and Outputs in GRU cell**
<br>
Input at every GRU cell/timestamp “t”:
* c\<t-1>: old “hidden state”

* x\<t>: embedding vector for the word at timestamp “t”

Output at every LSTM cell/timestamp “t”:
* c\<t>: new “hidden state”
<br><br>

#### **ii. Candidate Value (GRU)**
<br>
A candidate value contains new information to be baked into the outputs of a cell. A candidate is a potential alternate to replace information in ct. It is denoted by c̃ (pronounced as c tilde). Given by the equation:
<br><br>

<p style="font-family: 'Times New Roman'; font-size:125%" align="center">
c<sup>~ t</sup> = tanh(Wc<sub>cc</sub><sup>t-1</sup> + W<sub>cx</sub>*x<sup>t</sup> + b<sub>c</sub>)
</p><br>
The equation is very similar to one in LSTM. It is calculated using the “previous hidden state” c<t-1> (containing also the memory cell information) and the “word/any value” x<t>.

Components of the equation:
* “previous hidden state” c<sup>t-1</sup> contains both long and short-term information.

* “word” x<sup>t</sup> contains new information about “word” at the current timestamp.

* W<sub>cc</sub> and W<sub>cx</sub> are the weight parameters which will be learned by the cell at current cell/timestamp for c<sup>t-1</sup> and x<sup>t</sup> respectively.

* b<sub>c</sub> is the bias parameter.

So the Candidate Value contains the “**Information from previous hidden state**” + “**Information from current word**” + Some “**Bias**”. This is then passed through the “tanh” function which controls the exploding gradients problem by returning values between -1 and 1.
<br><br>

#### **iii. New Hidden State // Working of Update Gate (GRU)**
<br>
Update Gate determines how much of the information from the “candidate value” and “previous state” is to be updated in the hidden state. Its value lies between 0 and 1. It is given by the equation:
<br><br>

<p style="font-family: 'Times New Roman'; font-size:125%" align="center">
𝚪<sub>u</sub> = Sigmoid(W<sub>uc</sub>* c<sup>t-1</sup> + W<sub>ux</sub>* x<sup>t</sup> + b<sub>u</sub>)
</p><br>

Now in this final step, update gate is used along with “previous hidden state” and “candidate value” to determine what proportion of new information and old information is to be kept.

For example, Update gate of 0.4, would means:
* Keep 40% of information from “candidate value”. 

* Keep (1-0.4) i.e. 60% of the useful information from the “previous hidden state”, and forget the rest.
<br><br>

<p style="font-family: 'Times New Roman'; font-size:125%" align="center">
c<sup>t</sup> = 𝚪<sub>u</sub>* c<sup>~ t</sup> + (1-𝚪<sub>u</sub>)* c<sup>t-1</sup>
</p><br>

**In conclusion**, a GRU cell gets as input: “previous hidden state” and “new word embedding/any other input value” at each timestamp. Based on these two inputs, candidate value and update gate is calculated. Next, the Update Gate is multiplied with “previous hidden state” and “candidate value” to find out what proportion of information is to be kept from both of them. This gives us the “new hidden state” which is then fed onto the next GRU cell. This goes on for all the timestamps. And when it's done, the output of the GRU layer, can then be fed to the next Layer in the Neural Network.




