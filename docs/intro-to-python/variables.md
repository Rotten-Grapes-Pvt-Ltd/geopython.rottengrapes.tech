# Variables 

Variables in simple language are the names assigned to value, so that instead of the value you can just refer to this variable name throughout your code. e.g. If we want to refer to my age in code, instead of writing the value 30, I can just assign this value to a variable

```python
my_age = 30
```

and now, I can use this variable **my_age** in my code instead of the value 30. You might ask, why do we do this? Imagine you are writing a code where you are going to use this value at 10 different places, If you don't define the variable and directly use the value, then if you want to change the value from 30 to 31, you'll have to make changes at all 10 places, which make the probability of making error more. But if we use the variable in our python file, then we can just reassign new value to **my_age** and then this new value will be used everywhere. 

Here is an example of this

```python
# Python code for various checks on age with variable

my_age = 30

if my_age < 25: 
    print('you can not drink')
if my_age < 18: 
    print('you can not vote')
if my_age < 16: 
    print('you can not drive')
if my_age < 14: 
    print('you can not work')
if my_age < 10: 
    print('you can not stay awake after 10 PM')
```

now in above case, if I want to check for any age, I can simply change the value of age and run the code. But if I directly use the value instead of variable, I might make mistake

```python hl_lines="4"
# Python code for various checks on age without variable
if 30 < 25: 
    print('you can not drink')
if 10 < 18:  #  Accidentally wrote 10 instead of 30  
    print('you can not vote')
if 30 < 16: 
    print('you can not drive')
if 30 < 14: 
    print('you can not work')
if 30 < 10: 
    print('you can not stay awake after 10 PM')
```

## Defining variables

In theory, variables can be anything from sensible words like *fruit_name* , *title* , *is_available* to big sentences with either camel xasing *vehicleNumer* , *mathScore* or snake casing *valid_driver_license_numer* , *total_test_score* . It might contain numbers like *is_18* , *more_than_30* , or it can start with `_` such as *_5_fruits_names* , *_3subjectAvg*. It can be completely random as *dsdfudfgsfs* , *fdFDSFS_232* , etc. 

Here are few rules that we need to follow

- Variable name can not start with number
- Variable name can not have special characters 

## Reserve keywords

While variables can be anything, python does not allow few words as variable as they are used internally by python. e.g. `True`, `def` , `class` ,etc. Here is a [list](https://www.w3schools.com/python/python_ref_keywords.asp) of keywords that we can not use. 