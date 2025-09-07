# Installing Python

Whether you're a beginner stepping into programming or a data enthusiast diving into automation, Python is a great language to start with. In this post, we’ll explore various ways to install Python, both locally on your computer and through online platforms, so you can choose the one that works best for you. 

## Installing Python Locally

Let's explore steps to install Python on various Operating Systems. Let's start by downloading the latest package by either hovering on Downloads and downloading package or by visiting [https://www.python.org/downloads/](https://www.python.org/downloads/)

![Python Download Page](/assets/images/1.png)

### Installing python on Windows 
Start by downloading latest .exe installer from [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)
Once downloaded, double click on it to open installer 

![Python Installer](/assets/images/2.png)

After checking both boxes, click on Customize Installation and checkmark everything

![Python Customize Installation](/assets/images/3.png)

click on Next , then checkmark as per following screenshots

![Python Advanced Options](/assets/images/4.png)

and then click on Install. After installation is done, you can open Command Prompt and confirm if installation is done by typing

```bash
python --version
```

and if you see following result, it means your installation is done ! 

![Python Version Check](/assets/images/5.png)

### Installing python on Ubuntu 

Open terminal and enter following command

```bash
sudo apt install python3
```

enter your password.

![Ubuntu Install Python](/assets/images/6.png)

after installing python3, we also need to install pip which manages the packages in python. To install pip, execute following command

```bash
sudo apt install python3-pip
```

You can check if the installations are done correctly by typing 

```bash
python3 --version #to check python installation
pip3 --version # to check pip installation
```
![Ubuntu Version Check](/assets/images/7.png)

### Installing python on MacOS

Installing Python on MacOS X is similar to Windows, you can download the installer and following the commands 

![Python Download Page](/assets/images/8.png)

once the installation is done, check on terminal

![MacOS Version Check](/assets/images/9.png)

## Hosted Python Environments (No Installation Required!)

If you don’t want to install anything yet, you can run Python in the cloud. Ideal for learning or quick testing.

### Google Colab

- URL: [colab.research.google.com](colab.research.google.com)
- Free, cloud-based Jupyter notebooks with access to GPU
- Great for data science and ML

### Replit

- URL: [replit.com](replit.com)
- Supports Python and many other languages
- Includes a file manager, debugger, and terminal

### Jupyter Notebook (via Binder)

- URL: [mybinder.org](mybinder.org)
- Turn any GitHub repo into an executable notebook


